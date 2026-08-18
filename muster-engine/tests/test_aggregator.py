"""Raw events become minute buckets, and stay correct when time misbehaves.

Three properties carry this module, and each has a failure that is silent rather than
loud:

* **capture time, never wall time** — a frame processed late after a reconnect belongs to
  the minute it was *captured* in, or every outage quietly rewrites history;
* **re-closing a bucket is safe** — a crash mid-minute means the bucket is folded twice,
  and the second fold must produce the same rows rather than doubled ones;
* **a dwell lands in the bucket it ended in** — not the one it started in, which for a
  long dwell is a different hour of the day.

Reducers are faked here on purpose. The core six are P2.4 and their arithmetic is checked
against labelled footage (MK.2); what this file tests is the machinery that decides
*which events reach which reducer for which minute*.

Red-first for P2.6.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml

from muster.aggregator.aggregator import EXIT_GRACE_S, Aggregator, bucket_of
from muster.analytics.metrics.registry import MetricRegistry
from muster.config.schema import MusterConfig
from muster.store.store import Store
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

CAMERA = CameraId("front-door")
DOOR = LineId("door-count")
FLOOR = ZoneId("shop-floor")

T0 = datetime(2026, 8, 16, 9, 30, 0, tzinfo=UTC)
BUCKET_0 = MinuteBucket(T0)
BUCKET_1 = MinuteBucket(T0 + timedelta(minutes=1))


def _at(second: float) -> FrameTs:
    return FrameTs(T0 + timedelta(seconds=second))


def _cross(second: float, *, track: int = 1, direction: int = 1) -> RawEvent:
    return RawEvent(
        camera_id=CAMERA,
        ts=_at(second),
        kind=EventKind.LINE_CROSS,
        track_id=TrackId(track),
        line_id=DOOR,
        direction=direction,
    )


def _zone(second: float, kind: EventKind, *, track: int = 1, zone: ZoneId = FLOOR) -> RawEvent:
    return RawEvent(
        camera_id=CAMERA,
        ts=_at(second),
        kind=kind,
        track_id=TrackId(track),
        zone_id=zone,
    )


@dataclass
class _CountingPlugin:
    """A stand-in reducer: counts the events it was handed, per scope.

    Deliberately trivial. Its job is to make visible *what the aggregator routed to it*,
    which is the thing under test — not to be a plausible metric.
    """

    names: frozenset[MetricName] = field(default_factory=lambda: frozenset({MetricName.LINE_CROSS}))
    kinds: frozenset[EventKind] = field(default_factory=lambda: frozenset({EventKind.LINE_CROSS}))
    seen: list[tuple[MinuteBucket, int]] = field(default_factory=list)

    def reduce(self, events: list[RawEvent], bucket: MinuteBucket) -> list[MetricRow]:
        relevant = [event for event in events if event.kind in self.kinds]
        self.seen.append((bucket, len(relevant)))
        if not relevant:
            return []
        return [
            MetricRow(
                camera_id=relevant[0].camera_id,
                bucket=bucket,
                metric=next(iter(sorted(self.names))),
                scope_id=ScopeId(str(relevant[0].line_id or relevant[0].zone_id)),
                value=float(len(relevant)),
                sample_count=len(relevant),
            )
        ]


@dataclass
class _DwellPlugin(_CountingPlugin):
    """Reports the durations it was given, so bucket attribution is visible."""

    names: frozenset[MetricName] = field(
        default_factory=lambda: frozenset({MetricName.DWELL_SECONDS})
    )
    kinds: frozenset[EventKind] = field(default_factory=lambda: frozenset({EventKind.DWELL_SAMPLE}))

    def reduce(self, events: list[RawEvent], bucket: MinuteBucket) -> list[MetricRow]:
        durations = [e.value for e in events if e.kind in self.kinds and e.value is not None]
        self.seen.append((bucket, len(durations)))
        return [
            MetricRow(
                camera_id=CAMERA,
                bucket=bucket,
                metric=MetricName.DWELL_SECONDS,
                scope_id=ScopeId(str(FLOOR)),
                value=duration,
                sample_count=1,
            )
            for duration in durations
        ]


def _aggregator(
    *plugins: _CountingPlugin,
    dwell_min_s: float = 3.0,
    exit_grace_s: float = EXIT_GRACE_S,
) -> Aggregator:
    registry = MetricRegistry()
    for plugin in plugins or (_CountingPlugin(),):
        registry.register(plugin)
    return Aggregator(registry, dwell_min_s=dwell_min_s, exit_grace_s=exit_grace_s)


# --- Bucketing by capture time ----------------------------------------------


def test_a_timestamp_floors_to_its_minute() -> None:
    assert bucket_of(_at(59.999)) == BUCKET_0
    assert bucket_of(_at(60.0)) == BUCKET_1


def test_events_land_in_the_bucket_they_were_captured_in() -> None:
    plugin = _CountingPlugin()
    aggregator = _aggregator(plugin)

    aggregator.ingest(_cross(5))
    aggregator.ingest(_cross(70))

    assert [row.value for row in aggregator.close_bucket(BUCKET_0)] == [1.0]
    assert [row.value for row in aggregator.close_bucket(BUCKET_1)] == [1.0]


def test_an_event_processed_late_still_lands_in_its_historical_bucket() -> None:
    """The reconnect case: a backlog drains long after the minute it belongs to.

    Bucketing on arrival would smear an outage's whole backlog into the minute the
    connection came back, which is the shape of a metric that quietly lies.
    """
    plugin = _CountingPlugin()
    aggregator = _aggregator(plugin)

    aggregator.ingest(_cross(70))
    aggregator.ingest(_cross(5))

    assert [row.value for row in aggregator.close_bucket(BUCKET_0)] == [1.0]


def test_the_pending_buckets_are_the_ones_holding_events() -> None:
    """The supervisor needs to know what is closable without guessing at the clock."""
    aggregator = _aggregator()

    aggregator.ingest(_cross(70))
    aggregator.ingest(_cross(5))

    assert aggregator.pending_buckets() == [BUCKET_0, BUCKET_1]


def test_a_bucket_with_no_events_produces_no_rows() -> None:
    aggregator = _aggregator()

    assert aggregator.close_bucket(BUCKET_0) == []


# --- Replay safety ----------------------------------------------------------


def test_closing_the_same_bucket_twice_produces_the_same_rows() -> None:
    """A crash mid-minute means the fold runs twice. It must not accumulate."""
    aggregator = _aggregator()
    aggregator.ingest(_cross(5))
    aggregator.ingest(_cross(6, track=2))

    first = aggregator.close_bucket(BUCKET_0)
    second = aggregator.close_bucket(BUCKET_0)

    assert first == second
    assert [row.value for row in first] == [2.0]


def test_replaying_a_bucket_through_the_store_does_not_double_count(tmp_path: Path) -> None:
    """The end-to-end property P2.6 exists to guarantee, against a real database.

    The aggregator being replay-safe and the store's upsert being idempotent are two
    separate mechanisms; this is the one test that proves they compose.
    """
    config = MusterConfig.model_validate(yaml.safe_load(EXAMPLE_CONFIG.read_text("utf-8")))
    aggregator = _aggregator()
    aggregator.ingest(_cross(5))
    aggregator.ingest(_cross(6, track=2))

    with Store(tmp_path / "muster.db") as store:
        store.migrate()
        store.apply_config(config)
        store.upsert_metrics(aggregator.close_bucket(BUCKET_0))
        store.upsert_metrics(aggregator.close_bucket(BUCKET_0))

        rows = store.unsynced_metrics(limit=10)

    assert [row.value for row in rows] == [2.0]


def test_a_late_event_reopens_a_bucket_that_was_already_closed() -> None:
    """Closing is not sealing. The upsert downstream is what makes a re-fold safe."""
    aggregator = _aggregator()
    aggregator.ingest(_cross(5))
    aggregator.close_bucket(BUCKET_0)

    aggregator.ingest(_cross(6, track=2))

    assert [row.value for row in aggregator.close_bucket(BUCKET_0)] == [2.0]


def test_forgetting_old_buckets_bounds_the_memory() -> None:
    """Events are kept so a bucket can be re-folded; something has to end that."""
    aggregator = _aggregator()
    aggregator.ingest(_cross(5))
    aggregator.ingest(_cross(70))

    aggregator.forget_before(BUCKET_1)

    assert aggregator.pending_buckets() == [BUCKET_1]
    assert aggregator.close_bucket(BUCKET_0) == []


# --- Routing to reducers ----------------------------------------------------


def test_every_registered_plugin_sees_the_bucket() -> None:
    crossings = _CountingPlugin()
    dwells = _DwellPlugin()
    aggregator = _aggregator(crossings, dwells)

    aggregator.ingest(_cross(5))
    aggregator.close_bucket(BUCKET_0)

    assert crossings.seen == [(BUCKET_0, 1)]
    assert dwells.seen == [(BUCKET_0, 0)]


def test_registering_the_same_metric_twice_is_an_error() -> None:
    """Two reducers writing one metric name would race on the natural key."""
    registry = MetricRegistry()
    registry.register(_CountingPlugin())

    with pytest.raises(ValueError, match="line_cross"):
        registry.register(_CountingPlugin())


def test_a_plugin_may_declare_more_than_one_metric() -> None:
    """Occupancy is peak *and* mean, from one fold over one sample stream (§6.1).

    Splitting that across two plugins would make them read the same events twice and
    leave the raw/confirmed pair free to drift apart.
    """
    registry = MetricRegistry()
    both = frozenset({MetricName.OCCUPANCY, MetricName.OCCUPANCY_RAW})

    registry.register(_CountingPlugin(names=both))

    with pytest.raises(ValueError, match="occupancy_raw"):
        registry.register(_CountingPlugin(names=frozenset({MetricName.OCCUPANCY_RAW})))


def test_an_occupancy_sample_does_not_open_a_dwell() -> None:
    """A sample names a zone but no track, and the dwell machine is keyed by track."""
    aggregator = _aggregator()

    aggregator.ingest(
        RawEvent(
            camera_id=CAMERA,
            ts=_at(1.0),
            kind=EventKind.OCCUPANCY_SAMPLE,
            track_id=None,
            zone_id=FLOOR,
            value=3.0,
            dt_s=0.5,
        )
    )

    assert aggregator.open_dwells() == 0


# --- Dwell: the state machine algorithms.md §7 specifies ---------------------


def test_a_completed_dwell_lands_in_the_bucket_it_ended_in() -> None:
    """A dwell that spans a minute boundary belongs to the minute it closed in."""
    dwells = _DwellPlugin()
    aggregator = _aggregator(dwells, exit_grace_s=2.0)

    aggregator.ingest(_zone(10, EventKind.ZONE_ENTER))
    aggregator.ingest(_zone(70, EventKind.ZONE_EXIT))
    aggregator.flush(_at(80))

    assert [row.value for row in aggregator.close_bucket(BUCKET_1)] == [60.0]
    assert aggregator.close_bucket(BUCKET_0) == []


def test_a_dwell_shorter_than_the_minimum_produces_no_record() -> None:
    """Someone clipping the corner of a zone is not a dwell (§7 entry debounce)."""
    dwells = _DwellPlugin()
    aggregator = _aggregator(dwells, dwell_min_s=3.0, exit_grace_s=2.0)

    aggregator.ingest(_zone(10, EventKind.ZONE_ENTER))
    aggregator.ingest(_zone(11, EventKind.ZONE_EXIT))
    aggregator.flush(_at(20))

    assert aggregator.close_bucket(BUCKET_0) == []


def test_a_re_entry_inside_the_grace_window_bridges_one_dwell() -> None:
    """Walking behind a pillar is one dwell, not two — the temporal analog of track_buffer."""
    dwells = _DwellPlugin()
    aggregator = _aggregator(dwells, exit_grace_s=2.0)

    aggregator.ingest(_zone(10, EventKind.ZONE_ENTER))
    aggregator.ingest(_zone(20, EventKind.ZONE_EXIT))
    aggregator.ingest(_zone(21, EventKind.ZONE_ENTER))
    aggregator.ingest(_zone(40, EventKind.ZONE_EXIT))
    aggregator.flush(_at(50))

    assert [row.value for row in aggregator.close_bucket(BUCKET_0)] == [30.0]


def test_a_re_entry_after_the_grace_window_is_a_second_dwell() -> None:
    dwells = _DwellPlugin()
    aggregator = _aggregator(dwells, exit_grace_s=2.0)

    aggregator.ingest(_zone(10, EventKind.ZONE_ENTER))
    aggregator.ingest(_zone(20, EventKind.ZONE_EXIT))
    aggregator.ingest(_zone(30, EventKind.ZONE_ENTER))
    aggregator.ingest(_zone(45, EventKind.ZONE_EXIT))
    aggregator.flush(_at(50))

    assert sorted(row.value for row in aggregator.close_bucket(BUCKET_0)) == [10.0, 15.0]


def test_a_duplicate_enter_does_not_truncate_an_open_dwell() -> None:
    """§7 calls this out as a silent defect: the output is merely wrong, never loud.

    Boundary flicker can produce a second `zone_enter` for a track already inside. Taking
    it as a fresh start would reset the clock and report a fraction of the real dwell.
    """
    dwells = _DwellPlugin()
    aggregator = _aggregator(dwells, exit_grace_s=2.0)

    aggregator.ingest(_zone(10, EventKind.ZONE_ENTER))
    aggregator.ingest(_zone(30, EventKind.ZONE_ENTER))
    aggregator.ingest(_zone(40, EventKind.ZONE_EXIT))
    aggregator.flush(_at(50))

    assert [row.value for row in aggregator.close_bucket(BUCKET_0)] == [30.0]


def test_dwells_in_overlapping_zones_are_independent() -> None:
    dwells = _DwellPlugin()
    aggregator = _aggregator(dwells, exit_grace_s=2.0)
    queue = ZoneId("queue-till")

    aggregator.ingest(_zone(10, EventKind.ZONE_ENTER))
    aggregator.ingest(_zone(12, EventKind.ZONE_ENTER, zone=queue))
    aggregator.ingest(_zone(30, EventKind.ZONE_EXIT, zone=queue))
    aggregator.ingest(_zone(40, EventKind.ZONE_EXIT))
    aggregator.flush(_at(50))

    assert sorted(row.value for row in aggregator.close_bucket(BUCKET_0)) == [18.0, 30.0]


def test_two_tracks_in_one_zone_dwell_independently() -> None:
    dwells = _DwellPlugin()
    aggregator = _aggregator(dwells, exit_grace_s=2.0)

    aggregator.ingest(_zone(10, EventKind.ZONE_ENTER, track=1))
    aggregator.ingest(_zone(12, EventKind.ZONE_ENTER, track=2))
    aggregator.ingest(_zone(30, EventKind.ZONE_EXIT, track=1))
    aggregator.ingest(_zone(42, EventKind.ZONE_EXIT, track=2))
    aggregator.flush(_at(50))

    assert sorted(row.value for row in aggregator.close_bucket(BUCKET_0)) == [20.0, 30.0]


def test_an_unresolved_dwell_stays_open() -> None:
    """A dwell is only known when it ends; nothing may be emitted before the grace lapses."""
    dwells = _DwellPlugin()
    aggregator = _aggregator(dwells, exit_grace_s=2.0)

    aggregator.ingest(_zone(10, EventKind.ZONE_ENTER))
    aggregator.ingest(_zone(40, EventKind.ZONE_EXIT))
    aggregator.flush(_at(41))

    assert aggregator.close_bucket(BUCKET_0) == []
    assert aggregator.open_dwells() == 1


def test_an_exit_with_no_open_dwell_is_ignored() -> None:
    """Restart loses the open-dwell state (§10), so the first exits after one are orphans."""
    dwells = _DwellPlugin()
    aggregator = _aggregator(dwells, exit_grace_s=2.0)

    aggregator.ingest(_zone(40, EventKind.ZONE_EXIT))
    aggregator.flush(_at(50))

    assert aggregator.close_bucket(BUCKET_0) == []
    assert aggregator.open_dwells() == 0
