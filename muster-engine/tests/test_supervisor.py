"""The supervisor's tick: drain, aggregate, close on a lag, and keep workers alive.

`tick()` is public and these tests drive it directly rather than racing `run()`. A test
that sleeps to let a loop catch up is a test that fails on a loaded CI box; a test that
advances the clock itself fails only when the logic is wrong.

Both clocks are injected, for the reason the aggregator has none: the lag this file is
mostly about is a relationship between two timestamps, and a real clock would turn it
into a relationship between two timestamps and a machine's load.

The config here gives **every** camera its own line, which the worked example does not —
`door-count` belongs to `front-door` alone. Sharing one line across cameras would make a
misattribution invisible, because every row would still look right.

Red-first for P2.7.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import fields
from datetime import UTC, datetime, timedelta
from functools import partial
from pathlib import Path
from typing import Any

import pytest
import yaml
from scripted_worker import emit_occupancy_sample, emit_then_die, emit_then_exit, line_of

from muster.aggregator.aggregator import EXIT_GRACE_S
from muster.config.schema import MusterConfig
from muster.store.store import Store
from muster.supervisor.handle import WorkerEntry
from muster.supervisor.supervisor import BUCKET_S, CLOSE_LAG_S, Supervisor
from muster.types import MetricName, ScopeId

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_CONFIG = REPO_ROOT / "examples" / "muster.yaml"

BUCKET_START = datetime(2026, 8, 16, 9, 30, tzinfo=UTC)


class FakeClock:
    """A wall clock the test winds forward by hand."""

    def __init__(self, start: datetime) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


@pytest.fixture
def config() -> MusterConfig:
    """The worked example, with a counting line per camera and a zone per camera."""
    parsed: dict[str, Any] = yaml.safe_load(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    cameras = [camera["camera_id"] for camera in parsed["cameras"]]
    parsed["lines"] = [
        {
            "line_id": f"{camera}-line",
            "camera_id": camera,
            "a": [0.10, 0.80],
            "b": [0.90, 0.80],
            "positive_dir": "in",
            "metrics": ["line_cross", "footfall"],
        }
        for camera in cameras
    ]
    parsed["zones"] = [
        {
            "zone_id": f"{camera}-zone",
            "camera_id": camera,
            "role": "area",
            "polygon": [[0.05, 0.30], [0.95, 0.30], [0.95, 0.95], [0.05, 0.95]],
            "metrics": ["occupancy"],
        }
        for camera in cameras
    ]
    return MusterConfig.model_validate(parsed)


@pytest.fixture
def store(tmp_path: Path, config: MusterConfig) -> Iterator[Store]:
    with Store(tmp_path / "muster.db") as store:
        store.migrate()
        store.apply_config(config)
        yield store


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock(BUCKET_START + timedelta(seconds=5))


def _supervisor(
    config: MusterConfig, store: Store, clock: FakeClock, entry: WorkerEntry
) -> Supervisor:
    return Supervisor(config, store=store, entry=entry, now=clock)


def _footfall(store: Store) -> list[float]:
    rows = store.unsynced_metrics(limit=50)
    return sorted(row.value for row in rows if row.metric is MetricName.FOOTFALL)


def _logged_events(store: Store) -> int:
    (count,) = store._connection.execute("SELECT count(*) FROM events").fetchone()
    return int(count)


# --- Events become metrics --------------------------------------------------


async def test_worker_events_become_metric_rows(
    config: MusterConfig, store: Store, clock: FakeClock
) -> None:
    """The whole path, end to end: a worker's events land as minute rows in SQLite."""
    supervisor = _supervisor(config, store, clock, partial(emit_then_exit, crossings=3))

    await supervisor.start()
    await supervisor.drain_once(timeout=5.0, expected=3 * len(config.cameras))
    clock.advance(60 + CLOSE_LAG_S + 1)
    await supervisor.tick()
    await supervisor.stop()

    assert _footfall(store) == [3.0] * len(config.cameras)


async def test_each_camera_is_attributed_its_own_events(
    config: MusterConfig, store: Store, clock: FakeClock
) -> None:
    """A TrackId is unique per camera per run only, so attribution cannot be inferred."""
    supervisor = _supervisor(config, store, clock, partial(emit_then_exit, crossings=2))

    await supervisor.start()
    await supervisor.drain_once(timeout=5.0, expected=2 * len(config.cameras))
    clock.advance(60 + CLOSE_LAG_S + 1)
    await supervisor.tick()
    await supervisor.stop()

    rows = store.unsynced_metrics(limit=50)
    scoped = {(row.camera_id, row.scope_id) for row in rows if row.metric is MetricName.FOOTFALL}
    expected = {(camera.camera_id, ScopeId(line_of(camera.camera_id))) for camera in config.cameras}
    assert scoped == expected


# --- The closing lag --------------------------------------------------------


def test_the_lag_exceeds_the_aggregators_exit_grace() -> None:
    """The two numbers must not drift apart; a lag under the grace is the bug below."""
    assert CLOSE_LAG_S > EXIT_GRACE_S


async def test_a_bucket_is_not_closed_before_its_lag_has_elapsed(
    config: MusterConfig, store: Store, clock: FakeClock
) -> None:
    """A dwell ending at 09:30:59 is only known once its grace lapses at 09:31:01.

    Closing 09:30 at 09:31:00 exactly would miss it. Re-folding a closed bucket is safe
    by design, but the lag is what stops us relying on that every single minute.
    """
    supervisor = _supervisor(config, store, clock, partial(emit_then_exit, crossings=1))

    await supervisor.start()
    await supervisor.drain_once(timeout=5.0, expected=len(config.cameras))
    # One second before the 09:30 bucket is due: the minute has ended, the lag has not.
    clock.now = BUCKET_START + timedelta(seconds=BUCKET_S + CLOSE_LAG_S - 1)
    await supervisor.tick()
    await supervisor.stop(flush=False)

    assert _footfall(store) == []


async def test_a_bucket_is_closed_once_the_lag_has_elapsed(
    config: MusterConfig, store: Store, clock: FakeClock
) -> None:
    supervisor = _supervisor(config, store, clock, partial(emit_then_exit, crossings=1))

    await supervisor.start()
    await supervisor.drain_once(timeout=5.0, expected=len(config.cameras))
    clock.advance(60 + CLOSE_LAG_S + 1)
    await supervisor.tick()
    await supervisor.stop()

    assert _footfall(store) == [1.0] * len(config.cameras)


async def test_a_closed_bucket_is_not_written_twice(
    config: MusterConfig, store: Store, clock: FakeClock
) -> None:
    """Closing is idempotent at the store, and the supervisor must also stop re-closing."""
    supervisor = _supervisor(config, store, clock, partial(emit_then_exit, crossings=1))

    await supervisor.start()
    await supervisor.drain_once(timeout=5.0, expected=len(config.cameras))
    clock.advance(60 + CLOSE_LAG_S + 1)
    await supervisor.tick()
    clock.advance(60)
    await supervisor.tick()
    await supervisor.stop()

    assert _footfall(store) == [1.0] * len(config.cameras)


# --- The raw event log ------------------------------------------------------


async def test_track_bearing_events_reach_the_raw_log(
    config: MusterConfig, store: Store, clock: FakeClock
) -> None:
    supervisor = _supervisor(config, store, clock, partial(emit_then_exit, crossings=2))

    await supervisor.start()
    await supervisor.drain_once(timeout=5.0, expected=2 * len(config.cameras))
    await supervisor.stop()

    assert _logged_events(store) == 2 * len(config.cameras)


async def test_an_occupancy_sample_is_counted_but_never_logged(
    config: MusterConfig, store: Store, clock: FakeClock
) -> None:
    """`events.track_id` is NOT NULL, and a trackless append raises (ADR-0016).

    The supervisor is the component that has to choose what it logs, so the choice is
    made visible: the sample still reaches the aggregator, and the skip is a number
    rather than a gap someone finds later in a disk-usage graph.
    """
    supervisor = _supervisor(config, store, clock, emit_occupancy_sample)

    await supervisor.start()
    await supervisor.drain_once(timeout=5.0, expected=len(config.cameras))
    await supervisor.stop()

    assert _logged_events(store) == 0
    assert supervisor.skipped_untracked_events == len(config.cameras)


# --- Keeping workers alive --------------------------------------------------


async def test_a_dead_worker_is_restarted_once_its_backoff_has_elapsed(
    config: MusterConfig, store: Store, clock: FakeClock
) -> None:
    """One camera crashing costs that camera's next frames and nothing else (§9)."""
    supervisor = _supervisor(config, store, clock, partial(emit_then_die, crossings=1))

    await supervisor.start()
    await supervisor.drain_once(timeout=5.0, expected=len(config.cameras))
    await supervisor.supervise(monotonic=0.0)
    before = supervisor.restarts
    await supervisor.supervise(monotonic=120.0)

    assert before == 0, "a worker must not be respawned in the instant it died"
    assert supervisor.restarts == len(config.cameras)
    await supervisor.stop()


async def test_a_crashed_workers_events_are_not_lost(
    config: MusterConfig, store: Store, clock: FakeClock
) -> None:
    supervisor = _supervisor(config, store, clock, partial(emit_then_die, crossings=2))

    await supervisor.start()
    await supervisor.drain_once(timeout=5.0, expected=2 * len(config.cameras))
    clock.advance(60 + CLOSE_LAG_S + 1)
    await supervisor.tick()
    await supervisor.stop()

    assert _footfall(store) == [2.0] * len(config.cameras)


# --- Shutdown ---------------------------------------------------------------


async def test_stopping_closes_the_buckets_still_open(
    config: MusterConfig, store: Store, clock: FakeClock
) -> None:
    """A clean shutdown must not throw away the minute in progress."""
    supervisor = _supervisor(config, store, clock, partial(emit_then_exit, crossings=4))

    await supervisor.start()
    await supervisor.drain_once(timeout=5.0, expected=4 * len(config.cameras))
    await supervisor.stop()

    assert _footfall(store) == [4.0] * len(config.cameras)


@pytest.mark.privacy
async def test_only_scalar_events_cross_the_process_boundary(
    config: MusterConfig, store: Store, clock: FakeClock
) -> None:
    """The frame guarantee is structural: a frame is not a thing this queue carries.

    §9's claim is that only small `RawEvent`s cross. This asserts the checkable half —
    nothing the supervisor received holds an array-shaped payload — so the day someone
    adds a convenient `crop` field to `RawEvent`, this fails rather than ships.
    """
    supervisor = _supervisor(config, store, clock, partial(emit_then_exit, crossings=2))

    await supervisor.start()
    received = await supervisor.drain_once(timeout=5.0, expected=2 * len(config.cameras))
    await supervisor.stop()

    assert received
    for event in received:
        for field in fields(event):
            value = getattr(event, field.name)
            assert not hasattr(value, "shape"), f"{field.name} carries an array"
            assert not isinstance(value, bytes | bytearray | memoryview)


async def test_an_event_for_a_forgotten_bucket_is_dropped_and_counted(
    config: MusterConfig, store: Store, clock: FakeClock
) -> None:
    """A closed-and-forgotten bucket must not be re-opened by a straggler.

    Memory is bounded by forgetting, so re-folding a forgotten bucket folds only the
    late event — and the store's upsert would write that partial value *over* the
    correct one. A bucket closed on 3 crossings would silently become 1.

    The counter is what discriminates here: the restarted worker replays the same
    events, so the re-folded value would coincidentally match. Verified by mutation —
    removing the `_is_late` guard turns this red.
    """
    supervisor = _supervisor(config, store, clock, partial(emit_then_die, crossings=3))
    cameras = len(config.cameras)

    await supervisor.start()
    await supervisor.drain_once(timeout=5.0, expected=3 * cameras)
    clock.advance(60 + CLOSE_LAG_S + 1)
    await supervisor.tick()

    await supervisor.supervise(monotonic=0.0)
    await supervisor.supervise(monotonic=120.0)
    await supervisor.drain_once(timeout=5.0, expected=3 * cameras)
    await supervisor.tick()
    await supervisor.stop()

    assert supervisor.late_events == 3 * cameras
    assert _footfall(store) == [3.0] * cameras
