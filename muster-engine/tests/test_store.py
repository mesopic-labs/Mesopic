"""What the SQLite store must guarantee: idempotence, durability, and no pixels.

Three of these tests are load-bearing rather than descriptive:

* the natural-key upsert must be idempotent for **camera-wide** metrics too, whose scope
  is `None` — the case where a nullable key column would silently stop de-duplicating;
* a camera dropped from config must not take its history with it, because every metric
  table cascades from `cameras`;
* no column anywhere may hold a pixel, asserted against the live schema rather than
  against the DDL file, so a migration cannot add one behind the audit's back.

Red-first for P2.5.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from muster.config.schema import MusterConfig
from muster.errors import StoreError
from muster.store.store import BASELINE_SCHEMA_VERSION, Store
from muster.types import (
    CameraId,
    EventKind,
    FrameTs,
    LineId,
    MetricName,
    MetricRow,
    MinuteBucket,
    RawEvent,
    ScopeId,
    TrackId,
    ZoneId,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_CONFIG = REPO_ROOT / "examples" / "muster.yaml"

FRONT_DOOR = CameraId("front-door")
DOOR_LINE = ScopeId("door-count")
BUCKET = MinuteBucket(datetime(2026, 8, 16, 9, 30, tzinfo=UTC))
LATER = MinuteBucket(datetime(2026, 8, 16, 9, 31, tzinfo=UTC))


@pytest.fixture
def config() -> MusterConfig:
    """The worked example — the same site every other test suite describes."""
    parsed = yaml.safe_load(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    return MusterConfig.model_validate(parsed)


@pytest.fixture
def store(tmp_path: Path, config: MusterConfig) -> Iterator[Store]:
    """A migrated store with the worked example's cameras, zones and lines in it."""
    with Store(tmp_path / "muster.db") as store:
        store.migrate()
        store.apply_config(config)
        yield store


def _metric(
    value: float,
    *,
    scope: ScopeId | None = DOOR_LINE,
    bucket: MinuteBucket = BUCKET,
    metric: MetricName = MetricName.FOOTFALL,
) -> MetricRow:
    return MetricRow(
        camera_id=FRONT_DOOR,
        bucket=bucket,
        metric=metric,
        scope_id=scope,
        value=value,
        sample_count=1,
    )


def _event(track: int = 1) -> RawEvent:
    return RawEvent(
        camera_id=FRONT_DOOR,
        ts=FrameTs(datetime(2026, 8, 16, 9, 30, 15, tzinfo=UTC)),
        kind=EventKind.LINE_CROSS,
        track_id=TrackId(track),
        line_id=LineId("door-count"),
        direction=1,
    )


# --- First run: the file, its pragmas, and its shape -------------------------


def test_the_schema_is_created_on_first_run(tmp_path: Path) -> None:
    """A fresh box has no database; starting the engine is what makes one."""
    path = tmp_path / "muster.db"
    store = Store(path)

    store.migrate()

    tables = {
        row[0]
        for row in store._connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    assert {
        "cameras",
        "zones",
        "lines",
        "events",
        "metrics_minute",
        "heatmap_minute",
        "schema_meta",
    } <= tables


def test_the_database_is_opened_in_wal_mode(store: Store) -> None:
    """WAL is what lets the dashboard, exporters and sync client read while we write."""
    (mode,) = store._connection.execute("PRAGMA journal_mode").fetchone()

    assert mode.lower() == "wal"


def test_foreign_keys_are_enforced_on_every_connection(store: Store) -> None:
    """`PRAGMA foreign_keys` is per-connection and defaults OFF — a real SQLite footgun.

    Setting it in the DDL file would enforce it exactly once, on the connection that
    created the schema, and never again.
    """
    (enabled,) = store._connection.execute("PRAGMA foreign_keys").fetchone()

    assert enabled == 1


def test_the_tables_are_strict_about_types(store: Store) -> None:
    """STRICT is why a string in an integer column is an error and not a silent cast."""
    with pytest.raises(sqlite3.IntegrityError):
        store._connection.execute(
            "INSERT INTO events (camera_id, ts, kind, track_id) VALUES (?, ?, ?, ?)",
            (FRONT_DOOR, "2026-08-16T09:30:00+00:00", "line_cross", "not-an-integer"),
        )


# --- Config-derived tables --------------------------------------------------


def test_apply_config_materializes_cameras_zones_and_lines(store: Store) -> None:
    """These tables exist so events and metrics can foreign-key to stable ids."""
    connection = store._connection

    assert [row[0] for row in connection.execute("SELECT camera_id FROM cameras ORDER BY 1")] == [
        "front-door",
        "till",
    ]
    assert [row[0] for row in connection.execute("SELECT zone_id FROM zones ORDER BY 1")] == [
        "behind-counter",
        "queue-till",
        "shop-floor",
    ]
    assert [row[0] for row in connection.execute("SELECT line_id FROM lines")] == ["door-count"]


def test_applying_a_changed_config_updates_in_place(store: Store, config: MusterConfig) -> None:
    """Config wins on conflict — a renamed camera must not become a second row."""
    renamed = config.model_copy(
        update={"cameras": [config.cameras[0].model_copy(update={"name": "Front entrance"})]}
    )

    store.apply_config(renamed)

    assert store._connection.execute(
        "SELECT name FROM cameras WHERE camera_id = ?", (FRONT_DOOR,)
    ).fetchone() == ("Front entrance",)


def test_dropping_a_camera_from_config_does_not_delete_its_history(
    store: Store, config: MusterConfig
) -> None:
    """Every metric table cascades from `cameras`, so a DELETE here is a data-loss bug.

    Editing a camera out of `muster.yaml` is a routine thing to do — retiring a lens,
    fixing a typo in a list. It must never be the thing that silently destroys months of
    counts, so `apply_config` upserts and never removes.
    """
    store.upsert_metrics([_metric(7.0)])
    without_front_door = config.model_copy(update={"cameras": [config.cameras[1]]})

    store.apply_config(without_front_door)

    assert store.unsynced_metrics(limit=10) == [_metric(7.0)]


# --- The idempotent upsert --------------------------------------------------


def test_upserting_the_same_bucket_twice_does_not_double_count(store: Store) -> None:
    """Replaying a bucket after a crash is the normal case, not the exceptional one."""
    store.upsert_metrics([_metric(3.0)])
    store.upsert_metrics([_metric(5.0)])

    rows = store.unsynced_metrics(limit=10)

    assert [row.value for row in rows] == [5.0]


def test_a_camera_wide_metric_survives_the_same_upsert_twice(store: Store) -> None:
    """The scope-less case, which a nullable key column would silently fail to dedupe.

    `MetricRow.scope_id` is `None` for camera-wide metrics, and SQL `NULL` never equals
    `NULL` — so an `ON CONFLICT` key containing one matches nothing and every replay
    appends. Occupancy would double on every restart.
    """
    camera_wide = _metric(4.0, scope=None, metric=MetricName.OCCUPANCY)
    store.upsert_metrics([camera_wide])
    store.upsert_metrics([camera_wide])

    rows = store.unsynced_metrics(limit=10)

    assert rows == [camera_wide]
    assert rows[0].scope_id is None


def test_a_scoped_and_a_camera_wide_metric_are_different_rows(store: Store) -> None:
    """Whatever `None` is stored as must not collide with a real scope id."""
    store.upsert_metrics(
        [
            _metric(1.0, scope=None, metric=MetricName.OCCUPANCY),
            _metric(2.0, scope=DOOR_LINE, metric=MetricName.OCCUPANCY),
        ]
    )

    assert len(store.unsynced_metrics(limit=10)) == 2


def test_metric_rows_round_trip_unchanged(store: Store) -> None:
    """Timestamps are UTC on the way in and UTC on the way out, not naive local time."""
    row = MetricRow(
        camera_id=FRONT_DOOR,
        bucket=BUCKET,
        metric=MetricName.DWELL_SECONDS,
        scope_id=ScopeId("shop-floor"),
        value=12.5,
        staff_value=2.5,
        sample_count=17,
    )

    store.upsert_metrics([row])

    assert store.unsynced_metrics(limit=10) == [row]


# --- The sync cursor --------------------------------------------------------


def test_unsynced_metrics_comes_back_in_bucket_order(store: Store) -> None:
    """The sync client ships oldest-first so a partial drain leaves a contiguous tail."""
    store.upsert_metrics([_metric(2.0, bucket=LATER), _metric(1.0, bucket=BUCKET)])

    assert [row.bucket for row in store.unsynced_metrics(limit=10)] == [BUCKET, LATER]


def test_unsynced_metrics_respects_its_limit(store: Store) -> None:
    """An unbounded read of a long offline backlog is how the sync client runs out of RAM."""
    store.upsert_metrics([_metric(2.0, bucket=LATER), _metric(1.0, bucket=BUCKET)])

    assert len(store.unsynced_metrics(limit=1)) == 1


def test_marking_rows_synced_takes_them_out_of_the_queue(store: Store) -> None:
    store.upsert_metrics([_metric(1.0, bucket=BUCKET), _metric(2.0, bucket=LATER)])
    first = store.unsynced_metrics(limit=1)

    store.mark_synced(first, synced_at=datetime(2026, 8, 16, 9, 40, tzinfo=UTC))

    assert [row.bucket for row in store.unsynced_metrics(limit=10)] == [LATER]


def test_a_resynced_row_is_not_stamped_twice(store: Store) -> None:
    """A lost 2xx makes the client retry; stamping is therefore idempotent by design."""
    store.upsert_metrics([_metric(1.0)])
    rows = store.unsynced_metrics(limit=10)
    stamped_at = datetime(2026, 8, 16, 9, 40, tzinfo=UTC)

    store.mark_synced(rows, synced_at=stamped_at)
    store.mark_synced(rows, synced_at=datetime(2026, 8, 16, 9, 45, tzinfo=UTC))

    (kept,) = store._connection.execute("SELECT synced_at FROM metrics_minute").fetchone()
    assert kept == stamped_at.isoformat()


def test_an_upsert_reopens_a_bucket_for_sync(store: Store) -> None:
    """A corrected value has to ship again, or the cloud keeps the number we replaced."""
    store.upsert_metrics([_metric(1.0)])
    store.mark_synced(store.unsynced_metrics(limit=10), synced_at=datetime(2026, 8, 16, tzinfo=UTC))

    store.upsert_metrics([_metric(9.0)])

    assert [row.value for row in store.unsynced_metrics(limit=10)] == [9.0]


# --- The event log ----------------------------------------------------------


def test_append_events_writes_the_raw_event_log(store: Store) -> None:
    store.append_events([_event(1), _event(2)])

    (count,) = store._connection.execute("SELECT count(*) FROM events").fetchone()
    assert count == 2


def test_an_event_for_an_unknown_camera_is_refused(store: Store) -> None:
    """Referential integrity is the whole reason the config-derived tables exist."""
    orphan = RawEvent(
        camera_id=CameraId("no-such-camera"),
        ts=FrameTs(datetime(2026, 8, 16, 9, 30, tzinfo=UTC)),
        kind=EventKind.ZONE_ENTER,
        track_id=TrackId(1),
        zone_id=ZoneId("shop-floor"),
    )

    with pytest.raises(StoreError):
        store.append_events([orphan])


def test_an_event_belonging_to_no_track_is_refused(store: Store) -> None:
    """The raw log is a log of per-track facts; `events.track_id` is `NOT NULL`.

    An occupancy sample names a zone and nobody in it, and the table has no column for
    the count or the interval it carries — so appending one would write a row that says
    only "something happened in this zone", once per zone per tick, forever. Refuse
    loudly: the supervisor has to choose what it logs rather than discover the gap in a
    disk-usage graph (ADR-0016).
    """
    sample = RawEvent(
        camera_id=CameraId("front-door"),
        ts=FrameTs(datetime(2026, 8, 16, 9, 30, tzinfo=UTC)),
        kind=EventKind.OCCUPANCY_SAMPLE,
        track_id=None,
        zone_id=ZoneId("shop-floor"),
        value=3.0,
        dt_s=0.4,
    )

    with pytest.raises(StoreError, match="occupancy_sample"):
        store.append_events([sample])


# --- Forward-only migrations ------------------------------------------------


def test_a_fresh_database_is_stamped_at_the_baseline_version(tmp_path: Path) -> None:
    """`schema.sql` is the current shape, so a new file starts current, not at zero."""
    store = Store(tmp_path / "muster.db")

    store.migrate()

    assert store.schema_version() == BASELINE_SCHEMA_VERSION


def test_a_migration_bumps_the_schema_version(tmp_path: Path) -> None:
    """The mechanism, driven by a real file, without inventing a migration we do not need."""
    migrations = tmp_path / "migrations"
    migrations.mkdir()
    (migrations / "0002_add_a_column.sql").write_text(
        "ALTER TABLE schema_meta ADD COLUMN note TEXT;", encoding="utf-8"
    )
    store = Store(tmp_path / "muster.db", migrations_dir=migrations)

    store.migrate()

    assert store.schema_version() == 2
    assert "note" in {row[1] for row in store._connection.execute("PRAGMA table_info(schema_meta)")}


def test_migrating_twice_applies_each_migration_once(tmp_path: Path) -> None:
    """Every start-up calls `migrate`; a second pass must be a no-op, not an error."""
    migrations = tmp_path / "migrations"
    migrations.mkdir()
    (migrations / "0002_add_a_column.sql").write_text(
        "ALTER TABLE schema_meta ADD COLUMN note TEXT;", encoding="utf-8"
    )
    store = Store(tmp_path / "muster.db", migrations_dir=migrations)
    store.migrate()

    store.migrate()

    assert store.schema_version() == 2


def test_migrations_apply_in_numeric_order(tmp_path: Path) -> None:
    """`0010` sorts before `0009` as text — ordering has to be numeric, not lexicographic."""
    migrations = tmp_path / "migrations"
    migrations.mkdir()
    (migrations / "0002_first.sql").write_text(
        "CREATE TABLE ordering (step TEXT);INSERT INTO ordering VALUES ('first');",
        encoding="utf-8",
    )
    (migrations / "0010_second.sql").write_text(
        "INSERT INTO ordering VALUES ('second');", encoding="utf-8"
    )
    store = Store(tmp_path / "muster.db", migrations_dir=migrations)

    store.migrate()

    assert [row[0] for row in store._connection.execute("SELECT step FROM ordering")] == [
        "first",
        "second",
    ]
    assert store.schema_version() == 10


def test_a_failed_migration_leaves_the_version_where_it_was(tmp_path: Path) -> None:
    """Half-applied is the worst outcome: the next start would skip the rest of the file."""
    migrations = tmp_path / "migrations"
    migrations.mkdir()
    (migrations / "0002_broken.sql").write_text(
        "CREATE TABLE fine (a TEXT); THIS IS NOT SQL;", encoding="utf-8"
    )
    store = Store(tmp_path / "muster.db", migrations_dir=migrations)

    with pytest.raises(StoreError):
        store.migrate()

    assert store.schema_version() == BASELINE_SCHEMA_VERSION
    assert store._connection.execute(
        "SELECT count(*) FROM sqlite_master WHERE name = 'fine'"
    ).fetchone() == (0,)


def test_a_database_written_by_a_newer_engine_is_refused(tmp_path: Path) -> None:
    """Forward-only means an older binary cannot downgrade a file — it must decline to try."""
    path = tmp_path / "muster.db"
    store = Store(path)
    store.migrate()
    store._connection.execute(
        "UPDATE schema_meta SET value = ? WHERE key = 'schema_version'", ("99",)
    )
    store._connection.commit()

    with pytest.raises(StoreError, match="newer"):
        store.migrate()


# --- The privacy audit ------------------------------------------------------


@pytest.mark.privacy
def _table_flags(store: Store) -> dict[str, tuple[bool, bool]]:
    """Per table: is it `WITHOUT ROWID`, and is it `STRICT`, as SQLite itself reports.

    `PRAGMA table_list` rather than a substring search of `sqlite_master.sql`, which was
    the first attempt and was WRONG: SQLite stores the original CREATE text comments
    included, so a table whose comment merely mentions "strict" passed a text check while
    having no STRICT clause at all. Verified by mutation — the text version did not fire.
    """
    return {
        name: (bool(without_rowid), bool(strict))
        for _schema, name, kind, _ncol, without_rowid, strict in store._connection.execute(
            "PRAGMA table_list"
        ).fetchall()
        if kind == "table" and not name.startswith("sqlite_")
    }


def test_no_column_in_the_live_schema_can_hold_a_pixel(store: Store) -> None:
    """Asserted against the database as built, so a migration cannot slip one past.

    `heatmap_minute.counts` is the one blob, and it is a per-zone density grid in
    deciseconds — a number per cell, not a picture. Everything else that could carry
    image bytes is absent by construction (ADR-0005).
    """
    forbidden = ("image", "frame", "crop", "pixel", "jpeg", "jpg", "png", "thumbnail", "snapshot")
    allowed_blobs = {("heatmap_minute", "counts")}

    for (table,) in store._connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall():
        for column in store._connection.execute(f"PRAGMA table_info({table})").fetchall():
            name, declared_type = column[1], column[2].upper()
            assert not any(word in name.lower() for word in forbidden), (
                f"{table}.{name} names something that sounds like image data"
            )
            if declared_type == "BLOB":
                assert (table, name) in allowed_blobs, f"{table}.{name} is an unaudited BLOB column"


def test_every_table_in_the_live_schema_is_strict(store: Store) -> None:
    """Every table, not just the one an INSERT happens to exercise.

    `test_the_tables_are_strict_about_types` above proves the *semantics* are real, by
    watching `events` refuse a string in an integer column. This proves the *coverage*:
    read off the built schema, so a migration that adds a lax table fails here even
    though no test inserts into it yet. Without STRICT, SQLite stores the string and the
    bug surfaces weeks later as a metric that reads wrong.
    """
    lax = [table for table, (_wr, strict) in _table_flags(store).items() if not strict]

    assert lax == [], f"not STRICT: {', '.join(lax)}"


def test_the_composite_key_time_series_tables_are_without_rowid(store: Store) -> None:
    """`WITHOUT ROWID` clusters storage by the natural key these tables range-scan by.

    Named explicitly rather than inferred: these two are the hot time series, and a
    migration that rebuilds one without the clause would shrink no file and slow every
    bucket read, silently.
    """
    clustered = {"metrics_minute", "heatmap_minute"}
    flags = _table_flags(store)

    unclustered = [table for table in sorted(clustered) if not flags[table][0]]

    assert unclustered == [], f"lost WITHOUT ROWID: {', '.join(unclustered)}"


def test_the_metrics_table_columns_are_frozen(store: Store) -> None:
    """A column added here without updating this list is a column nothing audited.

    `metrics_minute` is the durable, syncable series: its shape is the edge half of the
    sync contract (ADR-0010), so a column appearing on one side and not the other is a
    wire-format break rather than a local detail. Deliberately a change-detector.
    """
    expected = {
        "camera_id",
        "bucket",
        "metric",
        "scope_id",
        "value",
        "staff_value",
        "sample_count",
        "synced_at",
    }

    columns = {
        row[1] for row in store._connection.execute("PRAGMA table_info(metrics_minute)").fetchall()
    }

    assert columns == expected
