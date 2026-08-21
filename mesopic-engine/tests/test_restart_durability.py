"""What a process restart must not cost, and what it is allowed to cost.

Restart safety is not a mechanism of its own here: it is the visible consequence of two
properties that already exist. P2.5's upsert is idempotent on
`(camera_id, metric, scope_id, bucket)` and P2.6's fold is deterministic over the events
a bucket holds, so "restart does not double-count" follows. What these tests add is the
restart itself — and one test asserts the *loss* engine-architecture.md §10 accepts, so a
later reader finds it recorded as a decision rather than filing it as a bug.

The kill is a real `SIGKILL` of a real subprocess rather than a `close()` and reopen.
Only the hard kill leaves a WAL behind with no orderly commit after it, which is the
state a restart actually has to recover from; a clean close proves nothing about it.

Red-first for P2.8.
"""

from __future__ import annotations

import signal
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml

from mesopic.aggregator.aggregator import Aggregator
from mesopic.analytics.metrics import build_registry
from mesopic.analytics.site_geometry import SiteGeometry
from mesopic.config.schema import MesopicConfig
from mesopic.store.store import Store
from mesopic.types import (
    CameraId,
    EventKind,
    FrameTs,
    LineId,
    MetricName,
    MetricRow,
    MinuteBucket,
    RawEvent,
    TrackId,
    ZoneId,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_CONFIG = REPO_ROOT / "examples" / "mesopic.yaml"

FRONT_DOOR = CameraId("front-door")
DOOR_LINE = LineId("door-count")
SHOP_FLOOR = ZoneId("shop-floor")
BUCKET = MinuteBucket(datetime(2026, 8, 16, 9, 30, tzinfo=UTC))

_WRITE_THEN_DIE = """
import os
import signal
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml

from mesopic.config.schema import MesopicConfig
from mesopic.store.store import Store
from mesopic.types import CameraId, MetricName, MetricRow, MinuteBucket, ScopeId

db_path, config_path = Path(sys.argv[1]), Path(sys.argv[2])
config = MesopicConfig.model_validate(
    yaml.safe_load(config_path.read_text(encoding="utf-8"))
)

store = Store(db_path)
store.migrate()
store.apply_config(config)
store.upsert_metrics(
    [
        MetricRow(
            camera_id=CameraId("front-door"),
            bucket=MinuteBucket(datetime(2026, 8, 16, 9, 30, tzinfo=UTC)),
            metric=MetricName.FOOTFALL,
            scope_id=ScopeId("door-count"),
            value=7.0,
            sample_count=1,
        )
    ]
)

# No close(), no atexit, no chance to flush anything: the process dies where it stands.
os.kill(os.getpid(), signal.SIGKILL)
"""


@pytest.fixture
def config() -> MesopicConfig:
    parsed = yaml.safe_load(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    return MesopicConfig.model_validate(parsed)


@pytest.fixture
def store(tmp_path: Path, config: MesopicConfig) -> Iterator[Store]:
    with Store(tmp_path / "mesopic.db") as store:
        store.migrate()
        store.apply_config(config)
        yield store


def _aggregator(config: MesopicConfig) -> Aggregator:
    """A real registry over the worked example — the fold under test is the real one."""
    return Aggregator(
        build_registry(SiteGeometry.compile(config)),
        dwell_min_s=config.thresholds.dwell_min_s,
    )


def _folded(config: MesopicConfig, events: list[RawEvent]) -> list[MetricRow]:
    """One minute's events through a *fresh* aggregator — a restarted process's fold."""
    aggregator = _aggregator(config)
    for event in events:
        aggregator.ingest(event)
    return aggregator.close_bucket(BUCKET)


def _crossing(second: int, track: int = 1) -> RawEvent:
    return RawEvent(
        camera_id=FRONT_DOOR,
        ts=FrameTs(datetime(2026, 8, 16, 9, 30, second, tzinfo=UTC)),
        kind=EventKind.LINE_CROSS,
        track_id=TrackId(track),
        line_id=DOOR_LINE,
        direction=1,
    )


def _zone_event(kind: EventKind, second: int, track: int = 1) -> RawEvent:
    return RawEvent(
        camera_id=FRONT_DOOR,
        ts=FrameTs(datetime(2026, 8, 16, 9, 30, second, tzinfo=UTC)),
        kind=kind,
        track_id=TrackId(track),
        zone_id=SHOP_FLOOR,
    )


# --- Durability across a hard kill ------------------------------------------


def test_metrics_written_before_a_kill_survive_it(tmp_path: Path) -> None:
    """`SIGKILL` mid-run, then reopen: the committed minute rows must still be there.

    This is what `synchronous = NORMAL` in WAL mode buys. The pages are handed to the OS
    on commit, so they outlive the process even though they may not yet be on the
    platter — process death is survived; only a power cut is not.
    """
    db_path = tmp_path / "mesopic.db"

    completed = subprocess.run(  # noqa: S603 - list argv, our own interpreter
        [sys.executable, "-c", _WRITE_THEN_DIE, str(db_path), str(EXAMPLE_CONFIG)],
        capture_output=True,
        check=False,
    )

    assert completed.returncode == -signal.SIGKILL, completed.stderr.decode()
    with Store(db_path) as reopened:
        reopened.migrate()
        rows = reopened.unsynced_metrics(limit=10)
    assert [row.value for row in rows] == [7.0]


def test_refolding_a_bucket_after_a_restart_does_not_double_count(
    tmp_path: Path, config: MesopicConfig
) -> None:
    """The restart-safety property, exercised across an actual reopen of the file.

    A supervisor that crashed mid-minute re-reads its events and closes the bucket
    again. The fold is deterministic and the upsert is keyed, so the second pass must
    overwrite the first rather than add to it.
    """
    db_path = tmp_path / "mesopic.db"
    events = [_crossing(10), _crossing(20, track=2)]

    with Store(db_path) as before:
        before.migrate()
        before.apply_config(config)
        before.upsert_metrics(_folded(config, events))

    with Store(db_path) as after:
        after.migrate()
        after.upsert_metrics(_folded(config, events))
        rows = after.unsynced_metrics(limit=10)

    footfall = [row for row in rows if row.metric is MetricName.FOOTFALL]
    assert [row.value for row in footfall] == [2.0]


def test_an_open_dwell_does_not_survive_a_restart_but_closed_minutes_do(
    tmp_path: Path, config: MesopicConfig
) -> None:
    """The accepted loss, asserted so it reads as a decision (§10), not an oversight.

    In-memory dwell state is gone on restart; a track mid-dwell loses its open interval.
    What must not be lost is anything already folded into a minute row.
    """
    db_path = tmp_path / "mesopic.db"
    aggregator = _aggregator(config)
    aggregator.ingest(_zone_event(EventKind.ZONE_ENTER, 5))

    with Store(db_path) as before:
        before.migrate()
        before.apply_config(config)
        before.upsert_metrics(_folded(config, [_crossing(10)]))
    assert aggregator.open_dwells() == 1

    restarted = _aggregator(config)
    with Store(db_path) as after:
        after.migrate()
        rows = after.unsynced_metrics(limit=10)

    assert restarted.open_dwells() == 0
    assert [row.value for row in rows if row.metric is MetricName.FOOTFALL] == [1.0]


# --- Event retention --------------------------------------------------------


def _event_count(store: Store) -> int:
    (count,) = store._connection.execute("SELECT count(*) FROM events").fetchone()
    return int(count)


def test_trim_deletes_events_older_than_the_cutoff(store: Store) -> None:
    store.append_events([_crossing(10), _crossing(20, track=2)])

    store.trim(before=datetime(2026, 8, 16, 9, 30, 15, tzinfo=UTC))

    assert _event_count(store) == 1


def test_trim_keeps_an_event_exactly_at_the_cutoff(store: Store) -> None:
    """The window is half-open: an event at the cutoff is inside it, not outside."""
    store.append_events([_crossing(10)])

    store.trim(before=datetime(2026, 8, 16, 9, 30, 10, tzinfo=UTC))

    assert _event_count(store) == 1


def test_trim_reports_how_many_events_it_deleted(store: Store) -> None:
    """The retention job logs a number; a silent trim is indistinguishable from a no-op."""
    store.append_events([_crossing(10), _crossing(20, track=2)])

    deleted = store.trim(before=datetime(2026, 8, 16, 9, 31, tzinfo=UTC))

    assert deleted == 2


def test_trim_leaves_metric_rows_alone(store: Store, config: MesopicConfig) -> None:
    """`events` is the short-retention log. `metrics_minute` is the durable series."""
    store.append_events([_crossing(10)])
    store.upsert_metrics(_folded(config, [_crossing(10)]))

    store.trim(before=datetime(2026, 8, 16, 9, 31, tzinfo=UTC))

    assert _event_count(store) == 0
    assert store.unsynced_metrics(limit=10) != []


def test_the_configured_window_is_applied_in_hours(store: Store, config: MesopicConfig) -> None:
    """A cutoff derived from the config key must span hours, not minutes or days.

    Computed the way a caller has to compute it, against the default 72-hour window, so
    a unit mix-up fails here rather than silently on someone's disk. 71 hours old is
    inside the window and 73 is outside it — a window read as minutes or days puts both
    on the same side.
    """
    now = FrameTs(datetime(2026, 8, 19, 9, 30, tzinfo=UTC))
    window = timedelta(hours=config.storage.event_retention_hours)
    store.append_events(
        [
            replace(_crossing(10), ts=FrameTs(now - timedelta(hours=71))),
            replace(_crossing(20, track=2), ts=FrameTs(now - timedelta(hours=73))),
        ]
    )

    store.trim(before=now - window)

    assert _event_count(store) == 1
