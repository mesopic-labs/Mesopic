"""The narrow store API over one SQLite file.

WAL mode: many readers (dashboard, exporters, sync client) concurrent with the one
writer (aggregator, in the supervisor process). `STRICT` tables catch type bugs at the
boundary; `WITHOUT ROWID` on the composite-key time series clusters storage by the key we
always range-scan.

There is no method here that reads or writes an image, a crop, or a bbox pixel — the
privacy guarantee is made structural rather than promised (ADR-0005).

Two SQLite facts this module is built around, both of which bite silently:

* **`PRAGMA foreign_keys` is per-connection and defaults off.** Setting it in the DDL
  file would arm it once, on the connection that created the schema, and never again —
  so every connection is configured here instead, and the DDL file is pure DDL.
* **A `STRICT` table's primary-key columns are implicitly `NOT NULL`**, and `NULL` never
  equals `NULL` in an `ON CONFLICT` key anyway. A camera-wide metric has no scope, so
  `scope_id` is stored as `''` and mapped back to `None` at this boundary. Left as a
  literal `NULL` it would defeat the idempotent upsert and double-count occupancy on
  every restart.

Implements P2.5.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType

from muster.config.schema import MusterConfig
from muster.errors import StoreError
from muster.types import (
    CameraId,
    MetricName,
    MetricRow,
    MinuteBucket,
    RawEvent,
    ScopeId,
)

BASELINE_SCHEMA_VERSION = 1
"""The version `schema.sql` describes. Migrations are numbered from here upwards."""

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"
MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"

BUSY_TIMEOUT_MS = 5000
"""How long a reader waits behind the writer before giving up. WAL makes this rare."""

_CAMERA_WIDE = ""
"""How "this metric has no scope" is spelled in a key column that cannot hold `NULL`."""


class Store:
    """Owns the SQLite file. One instance, in the supervisor process."""

    def __init__(self, path: Path, *, migrations_dir: Path | None = None) -> None:
        """Open (and create, if absent) the database. Call `migrate` before using it.

        `migrations_dir` is a seam, not a setting: it lets the migration mechanism be
        exercised without shipping a migration the schema does not need yet.
        """
        self._path = path
        self._migrations_dir = MIGRATIONS_DIR if migrations_dir is None else migrations_dir
        try:
            self._connection = sqlite3.connect(path, isolation_level=None)
        except sqlite3.Error:
            msg = f"cannot open the store at {path}"
            raise StoreError(msg) from None
        self._configure()

    # --- Lifecycle ----------------------------------------------------------

    @property
    def path(self) -> Path:
        """Where this database lives. `/healthz` measures free space on *its* filesystem,
        which on an appliance with a separate data volume is not the root's."""
        return self._path

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _configure(self) -> None:
        """Per-connection pragmas. `journal_mode` persists in the file; the rest do not."""
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA synchronous = NORMAL")
        self._connection.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        """One write, one transaction. `IMMEDIATE` takes the write lock up front."""
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            yield self._connection
        except BaseException:
            self._connection.execute("ROLLBACK")
            raise
        self._connection.execute("COMMIT")

    # --- Schema -------------------------------------------------------------

    def migrate(self) -> None:
        """Apply forward-only migrations and stamp `schema_meta.schema_version`."""
        if not self._schema_exists():
            self._create_baseline()
        current = self.schema_version()
        pending = [(n, p) for n, p in self._migration_files() if n > current]
        latest = max([n for n, _ in self._migration_files()], default=BASELINE_SCHEMA_VERSION)
        if current > latest:
            msg = (
                f"the store at {self._path} is at schema version {current}, newer than this "
                f"engine's {latest} — migrations are forward-only, so it will not be opened"
            )
            raise StoreError(msg)
        for number, path in pending:
            self._apply(number, path)

    def schema_version(self) -> int:
        row = self._connection.execute(
            "SELECT value FROM schema_meta WHERE key = 'schema_version'"
        ).fetchone()
        return int(row[0]) if row else 0

    def _schema_exists(self) -> bool:
        (count,) = self._connection.execute(
            "SELECT count(*) FROM sqlite_master WHERE type = 'table' AND name = 'schema_meta'"
        ).fetchone()
        return int(count) > 0

    def _create_baseline(self) -> None:
        """`schema.sql` is the current shape, so a new file starts current, not at zero."""
        try:
            self._connection.executescript(f"BEGIN;\n{SCHEMA_PATH.read_text(encoding='utf-8')}")
            self._connection.execute(
                "INSERT INTO schema_meta (key, value) VALUES ('schema_version', ?)",
                (str(BASELINE_SCHEMA_VERSION),),
            )
            self._connection.execute("COMMIT")
        except (OSError, sqlite3.Error):
            self._rollback()
            msg = f"cannot create the schema in {self._path}"
            raise StoreError(msg) from None

    def _migration_files(self) -> list[tuple[int, Path]]:
        """`0010_x.sql` sorts before `0009_x.sql` as text, so order numerically."""
        if not self._migrations_dir.is_dir():
            return []
        found: list[tuple[int, Path]] = []
        for path in self._migrations_dir.glob("*.sql"):
            number, separator, _ = path.stem.partition("_")
            if not separator or not number.isdigit():
                msg = f"migration {path.name} is not named <number>_<description>.sql"
                raise StoreError(msg)
            found.append((int(number), path))
        return sorted(found)

    def _apply(self, number: int, path: Path) -> None:
        """One migration, one transaction — including its own version stamp.

        The `BEGIN` rides inside the script and the `COMMIT` does not, so the stamp joins
        the same transaction as the migration body. Half-applied is the outcome worth
        engineering against: the next start-up would skip the rest of the file.
        """
        try:
            self._connection.executescript(f"BEGIN;\n{path.read_text(encoding='utf-8')}")
            self._connection.execute(
                "UPDATE schema_meta SET value = ? WHERE key = 'schema_version'", (str(number),)
            )
            self._connection.execute("COMMIT")
        except (OSError, sqlite3.Error):
            self._rollback()
            msg = f"migration {path.name} failed; the store is still at version {
                self.schema_version()
            }"
            raise StoreError(msg) from None

    def _rollback(self) -> None:
        """Best-effort: a failure that already aborted the transaction is not an error."""
        with suppress(sqlite3.Error):
            self._connection.execute("ROLLBACK")

    # --- Config-derived tables ----------------------------------------------

    def apply_config(self, config: MusterConfig) -> None:
        """Materialize config's cameras, zones and lines so events can reference them.

        Upsert only, never delete. Every metric table cascades from `cameras`, so
        removing a camera here because it left `muster.yaml` would silently destroy its
        history — and editing a camera out of a config file is a routine thing to do.
        """
        now = datetime.now(UTC).isoformat()
        try:
            with self._transaction() as connection:
                connection.executemany(
                    """INSERT INTO cameras
                           (camera_id, name, source_kind, enabled, ref_width, ref_height,
                            updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(camera_id) DO UPDATE SET
                           name = excluded.name,
                           source_kind = excluded.source_kind,
                           enabled = excluded.enabled,
                           ref_width = excluded.ref_width,
                           ref_height = excluded.ref_height,
                           updated_at = excluded.updated_at""",
                    [
                        (
                            camera.camera_id,
                            camera.name,
                            camera.source.kind.value,
                            int(camera.enabled),
                            camera.reference_resolution[0],
                            camera.reference_resolution[1],
                            now,
                        )
                        for camera in config.cameras
                    ],
                )
                connection.executemany(
                    """INSERT INTO zones (zone_id, camera_id, role, polygon, updated_at)
                       VALUES (?, ?, ?, ?, ?)
                       ON CONFLICT(zone_id) DO UPDATE SET
                           camera_id = excluded.camera_id,
                           role = excluded.role,
                           polygon = excluded.polygon,
                           updated_at = excluded.updated_at""",
                    [
                        (
                            zone.zone_id,
                            zone.camera_id,
                            zone.role.value,
                            json.dumps(zone.polygon),
                            now,
                        )
                        for zone in config.zones
                    ],
                )
                connection.executemany(
                    """INSERT INTO lines
                           (line_id, camera_id, ax, ay, bx, by, positive_dir, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(line_id) DO UPDATE SET
                           camera_id = excluded.camera_id,
                           ax = excluded.ax, ay = excluded.ay,
                           bx = excluded.bx, by = excluded.by,
                           positive_dir = excluded.positive_dir,
                           updated_at = excluded.updated_at""",
                    [
                        (
                            line.line_id,
                            line.camera_id,
                            line.a[0],
                            line.a[1],
                            line.b[0],
                            line.b[1],
                            line.positive_dir.value,
                            now,
                        )
                        for line in config.lines
                    ],
                )
        except sqlite3.Error:
            msg = "cannot write the config-derived tables"
            raise StoreError(msg) from None

    # --- Events -------------------------------------------------------------

    def append_events(self, events: Sequence[RawEvent]) -> None:
        """Write to the short-retention raw event log. Never synced.

        Every row is a fact about one track, which `events.track_id NOT NULL` enforces.
        A trackless event — an occupancy sample, which counts a zone rather than anyone
        in it — is refused rather than coerced, because the table has no column for what
        it actually carries and there is one of them per zone per tick (ADR-0016).
        """
        rows = [_event_row(event) for event in events]
        try:
            with self._transaction() as connection:
                connection.executemany(
                    """INSERT INTO events
                           (camera_id, ts, kind, track_id, line_id, zone_id, direction, is_staff)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    rows,
                )
        except sqlite3.Error:
            msg = "cannot append to the event log"
            raise StoreError(msg) from None

    # --- Metrics ------------------------------------------------------------

    def upsert_metrics(self, rows: Sequence[MetricRow]) -> None:
        """Idempotent upsert on `(camera_id, metric, scope_id, bucket)`.

        A re-written bucket is un-synced again: a corrected value has to ship, or the
        cloud keeps the number this one replaced.
        """
        try:
            with self._transaction() as connection:
                connection.executemany(
                    """INSERT INTO metrics_minute
                           (camera_id, bucket, metric, scope_id, value, staff_value,
                            sample_count, synced_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, NULL)
                       ON CONFLICT(camera_id, metric, scope_id, bucket) DO UPDATE SET
                           value = excluded.value,
                           staff_value = excluded.staff_value,
                           sample_count = excluded.sample_count,
                           synced_at = NULL""",
                    [
                        (
                            row.camera_id,
                            row.bucket.isoformat(),
                            row.metric.value,
                            _CAMERA_WIDE if row.scope_id is None else row.scope_id,
                            row.value,
                            row.staff_value,
                            row.sample_count,
                        )
                        for row in rows
                    ],
                )
        except sqlite3.Error:
            msg = "cannot write metric rows"
            raise StoreError(msg) from None

    def metrics_between(
        self,
        *,
        start: datetime,
        end: datetime,
        limit: int,
        camera_id: CameraId | None = None,
        metrics: Sequence[MetricName] | None = None,
    ) -> list[MetricRow]:
        """The series a dashboard plots: `[start, end)`, oldest first, capped at `limit`.

        The window is half-open for the reason `trim`'s is — a minute belongs to exactly
        one window, so consecutive requests neither double-count nor drop a bucket.

        Ordering and filtering live here rather than in the caller because
        engine-architecture.md §13 makes the API a reader: handlers query the store and
        render, they do not compute. `limit` has no default on purpose — a year of
        buckets is over half a million rows, and the caller has to have said what it can
        hold.
        """
        clauses = ["bucket >= ?", "bucket < ?"]
        parameters: list[object] = [start.isoformat(), end.isoformat()]
        if camera_id is not None:
            clauses.append("camera_id = ?")
            parameters.append(camera_id)
        if metrics is not None:
            clauses.append(f"metric IN ({', '.join('?' * len(metrics))})")
            parameters.extend(metric.value for metric in metrics)
        parameters.append(limit)
        cursor = self._connection.execute(
            f"""SELECT camera_id, bucket, metric, scope_id, value, staff_value, sample_count
                FROM metrics_minute
                WHERE {" AND ".join(clauses)}
                ORDER BY bucket, camera_id, metric, scope_id
                LIMIT ?""",  # noqa: S608 - every clause is a fixed string; values are bound
            parameters,
        )
        return [_row_from(record) for record in cursor.fetchall()]

    def unsynced_count(self) -> int:
        """How many rows have never reached the cloud. Reported by `/healthz` (§15)."""
        (count,) = self._connection.execute(
            "SELECT count(*) FROM metrics_minute WHERE synced_at IS NULL"
        ).fetchone()
        return int(count)

    def unsynced_metrics(self, limit: int) -> list[MetricRow]:
        """Rows awaiting cloud sync, in bucket order. Drives the sync cursor."""
        cursor = self._connection.execute(
            """SELECT camera_id, bucket, metric, scope_id, value, staff_value, sample_count
               FROM metrics_minute
               WHERE synced_at IS NULL
               ORDER BY bucket, camera_id, metric, scope_id
               LIMIT ?""",
            (limit,),
        )
        return [_row_from(record) for record in cursor.fetchall()]

    def mark_synced(self, rows: Sequence[MetricRow], synced_at: datetime) -> None:
        """Stamp `synced_at` after a confirmed 2xx from the cloud.

        Only rows still unstamped are touched, so a retry after a lost 2xx re-stamps
        nothing and the first delivery time is the one that survives.
        """
        stamp = synced_at.isoformat()
        try:
            with self._transaction() as connection:
                connection.executemany(
                    """UPDATE metrics_minute SET synced_at = ?
                       WHERE camera_id = ? AND metric = ? AND scope_id = ? AND bucket = ?
                         AND synced_at IS NULL""",
                    [
                        (
                            stamp,
                            row.camera_id,
                            row.metric.value,
                            _CAMERA_WIDE if row.scope_id is None else row.scope_id,
                            row.bucket.isoformat(),
                        )
                        for row in rows
                    ],
                )
        except sqlite3.Error:
            msg = "cannot stamp rows as synced"
            raise StoreError(msg) from None

    def trim(self, *, before: datetime) -> int:
        """Drop raw events captured before `before`. Returns how many went.

        The window is half-open — an event exactly at the cutoff is kept — and the
        caller supplies the cutoff rather than the retention window itself. The store
        has no clock for the same reason the aggregator has none: a component that reads
        the time decides, by itself and invisibly, which data exists. `before` is
        `now - timedelta(hours=config.storage.event_retention_hours)`.

        Only `events` is trimmed. `metrics_minute` is the durable, syncable series, and
        a row whose `synced_at` is still `NULL` has never reached the cloud — deleting
        one destroys the only copy. The disk-bounded metric cap engine-architecture.md
        §11 also names is deliberately not implemented here: §11 leaves its default an
        open question and §14 puts disk-full handling in M6.
        """
        cutoff = before.isoformat()
        try:
            with self._transaction() as connection:
                cursor = connection.execute("DELETE FROM events WHERE ts < ?", (cutoff,))
                return cursor.rowcount
        except sqlite3.Error:
            msg = "cannot trim the event log"
            raise StoreError(msg) from None


_EventRow = tuple[str, str, str, int, str | None, str | None, int | None, int]


def _event_row(event: RawEvent) -> _EventRow:
    """One raw event as the `events` table wants it, or a refusal.

    The guard lives here rather than in a loop of its own so the narrowing is the same
    expression as the write — a check that can drift away from the thing it protects is
    the check that eventually does.
    """
    if event.track_id is None:
        msg = f"event kind {event.kind.value!r} belongs to no track and cannot be logged"
        raise StoreError(msg)
    return (
        event.camera_id,
        event.ts.isoformat(),
        event.kind.value,
        int(event.track_id),
        event.line_id,
        event.zone_id,
        event.direction,
        int(event.is_staff),
    )


def _row_from(record: tuple[str, str, str, str, float, float | None, int]) -> MetricRow:
    camera_id, bucket, metric, scope_id, value, staff_value, sample_count = record
    return MetricRow(
        camera_id=CameraId(camera_id),
        bucket=MinuteBucket(datetime.fromisoformat(bucket)),
        metric=MetricName(metric),
        scope_id=None if scope_id == _CAMERA_WIDE else ScopeId(scope_id),
        value=value,
        staff_value=staff_value,
        sample_count=sample_count,
    )
