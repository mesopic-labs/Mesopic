"""Occupancy, queue length, and dwell — the metrics folded from samples and durations.

The `Δt`-weighting convention (algorithms.md §0.6) is the whole risk here, and it is not
a rounding matter. The adaptive sampler lowers fps when the box is under load, and the
box is under load when the scene is busy — so the sampling interval is *correlated with
the quantity being measured*, and a mean over samples under-weights the rush by whatever
the fps ratio happens to be. `test_the_mean_is_weighted_by_time_not_by_sample_count`
exists to make the two answers visibly different rather than plausibly close.

The peak/mean split is the other thing worth pinning: peak reads the raw count and mean
reads the confirmed one (§6.1), because the `dwell_min_s` confirmation lag suppresses
exactly the fast-turnover moments that create peaks.

Red-first for P2.4.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from muster.analytics.metrics import build_registry
from muster.analytics.metrics.dwell import DwellPlugin
from muster.analytics.metrics.occupancy import OccupancyPlugin, QueueLengthPlugin
from muster.analytics.site_geometry import SiteGeometry
from muster.config.schema import MusterConfig
from muster.types import (
    CameraId,
    EventKind,
    FrameTs,
    MetricName,
    MetricRow,
    MinuteBucket,
    RawEvent,
    ScopeId,
    TrackId,
    ZoneId,
)

CAMERA = CameraId("front-door")
FLOOR = ZoneId("floor")
TILL_QUEUE = ZoneId("till-queue")

T0 = datetime(2026, 8, 16, 9, 30, tzinfo=UTC)
BUCKET = MinuteBucket(T0)

SQUARE = [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]]


def _config(zones: list[dict[str, Any]]) -> MusterConfig:
    return MusterConfig.model_validate(
        {
            "site": {"site_id": "test-site"},
            "cameras": [
                {
                    "camera_id": CAMERA,
                    "name": "Front door",
                    "source": {"kind": "rtsp", "url_env": "MUSTER_TEST_RTSP"},
                    "reference_resolution": [1920, 1080],
                }
            ],
            "zones": zones,
        }
    )


def _zone(
    zone_id: str = FLOOR,
    *,
    role: str = "area",
    metrics: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "zone_id": zone_id,
        "camera_id": CAMERA,
        "role": role,
        "polygon": SQUARE,
        "metrics": ["occupancy", "dwell_seconds"] if metrics is None else metrics,
    }


def _geometry(*zones: dict[str, Any]) -> SiteGeometry:
    return SiteGeometry.compile(_config(list(zones or (_zone(),))))


def _sample(
    *, raw: float, confirmed: float, dt_s: float, second: float = 0.0, zone: ZoneId = FLOOR
) -> RawEvent:
    """One tick's count of a zone. `value` is the raw count; confirmed rides alongside."""
    return RawEvent(
        camera_id=CAMERA,
        ts=FrameTs(T0 + timedelta(seconds=second)),
        kind=EventKind.OCCUPANCY_SAMPLE,
        track_id=None,
        zone_id=zone,
        value=raw,
        confirmed_value=confirmed,
        dt_s=dt_s,
    )


def _dwell(seconds: float, *, track: int = 1, zone: ZoneId = FLOOR) -> RawEvent:
    return RawEvent(
        camera_id=CAMERA,
        ts=FrameTs(T0 + timedelta(seconds=30)),
        kind=EventKind.DWELL_SAMPLE,
        track_id=TrackId(track),
        zone_id=zone,
        value=seconds,
    )


def _by_metric(rows: list[MetricRow]) -> dict[tuple[MetricName, ScopeId | None], float]:
    return {(row.metric, row.scope_id): row.value for row in rows}


# --- Occupancy: the two series ----------------------------------------------


def test_the_mean_is_weighted_by_time_not_by_sample_count() -> None:
    """The §0.6 bias, made visible.

    One person for four seconds, sampled forty times because the box was idle. Then
    eleven people for four seconds, sampled ten times because inference got slower as the
    scene filled — which is the shape the adaptive sampler actually produces, not a
    contrived one.

    Half the minute held one person and half held eleven, so the honest average is 6.0.
    Averaging over samples answers 3.0, because the quiet stretch contributed four times
    as many of them. Both are "average occupancy for the minute"; one of them halves the
    busy period.
    """
    quiet = [_sample(raw=1.0, confirmed=1.0, dt_s=0.1, second=i * 0.1) for i in range(40)]
    rush = [_sample(raw=11.0, confirmed=11.0, dt_s=0.4, second=4.0 + i * 0.4) for i in range(10)]
    plugin = OccupancyPlugin(_geometry())

    rows = plugin.reduce(quiet + rush, BUCKET)

    sample_mean = (40 * 1.0 + 10 * 11.0) / 50
    assert sample_mean == 3.0
    assert _by_metric(rows)[(MetricName.OCCUPANCY, ScopeId(FLOOR))] == pytest.approx(6.0)


def test_the_peak_reads_the_raw_count_not_the_confirmed_one() -> None:
    """Confirmation lags by `dwell_min_s`, and peaks form inside that lag (§6.1)."""
    plugin = OccupancyPlugin(_geometry())

    rows = plugin.reduce(
        [
            _sample(raw=2.0, confirmed=2.0, dt_s=0.5, second=0.0),
            _sample(raw=9.0, confirmed=2.0, dt_s=0.5, second=0.5),
        ],
        BUCKET,
    )

    assert _by_metric(rows)[(MetricName.OCCUPANCY_RAW, ScopeId(FLOOR))] == 9.0


def test_the_mean_reads_the_confirmed_count_not_the_raw_one() -> None:
    """Pass-through traffic must not inflate the average (`min_dwell_to_count`, §6)."""
    plugin = OccupancyPlugin(_geometry())

    rows = plugin.reduce(
        [
            _sample(raw=9.0, confirmed=2.0, dt_s=0.5, second=0.0),
            _sample(raw=9.0, confirmed=2.0, dt_s=0.5, second=0.5),
        ],
        BUCKET,
    )

    assert _by_metric(rows)[(MetricName.OCCUPANCY, ScopeId(FLOOR))] == 2.0


def test_both_series_report_how_many_samples_backed_them() -> None:
    """A thin bucket should read as thin rather than as confident (§6.2)."""
    plugin = OccupancyPlugin(_geometry())

    rows = plugin.reduce(
        [
            _sample(raw=1.0, confirmed=1.0, dt_s=0.5, second=0.0),
            _sample(raw=3.0, confirmed=3.0, dt_s=0.5, second=0.5),
        ],
        BUCKET,
    )

    assert [row.sample_count for row in rows] == [2, 2]


def test_one_plugin_owns_both_occupancy_series() -> None:
    """They come from one fold over one sample stream, so they cannot drift apart."""
    assert OccupancyPlugin(_geometry()).names == frozenset(
        {MetricName.OCCUPANCY, MetricName.OCCUPANCY_RAW}
    )


def test_a_zone_with_no_samples_produces_no_row() -> None:
    """No observation is not an observation of zero."""
    plugin = OccupancyPlugin(_geometry())

    assert plugin.reduce([_dwell(10.0)], BUCKET) == []


def test_a_zone_that_does_not_ask_for_occupancy_produces_no_row() -> None:
    plugin = OccupancyPlugin(_geometry(_zone(metrics=["dwell_seconds"])))

    assert plugin.reduce([_sample(raw=1.0, confirmed=1.0, dt_s=0.5)], BUCKET) == []


def test_a_zero_width_sample_cannot_divide_the_mean_by_zero() -> None:
    """Geometry refuses to emit one, so this is a contract with a future emitter."""
    plugin = OccupancyPlugin(_geometry())

    assert plugin.reduce([_sample(raw=4.0, confirmed=4.0, dt_s=0.0)], BUCKET) == []


# --- Queue length: the same fold on a queue zone -----------------------------


def test_queue_length_is_occupancy_of_a_queue_zone() -> None:
    plugin = QueueLengthPlugin(_geometry(_zone(TILL_QUEUE, role="queue", metrics=["queue_len"])))

    rows = plugin.reduce(
        [
            _sample(raw=5.0, confirmed=4.0, dt_s=0.5, second=0.0, zone=TILL_QUEUE),
            _sample(raw=5.0, confirmed=4.0, dt_s=0.5, second=0.5, zone=TILL_QUEUE),
        ],
        BUCKET,
    )

    assert _by_metric(rows) == {
        (MetricName.QUEUE_LEN, ScopeId(TILL_QUEUE)): 4.0,
        (MetricName.QUEUE_LEN_RAW, ScopeId(TILL_QUEUE)): 5.0,
    }


def test_an_area_zone_is_not_a_queue() -> None:
    """Role decides. A shop floor is not a till queue however many people are on it."""
    plugin = QueueLengthPlugin(_geometry(_zone(FLOOR, metrics=["queue_len"])))

    assert plugin.reduce([_sample(raw=5.0, confirmed=5.0, dt_s=0.5)], BUCKET) == []


def test_a_queue_zone_is_not_counted_as_general_occupancy() -> None:
    """Otherwise the till queue is added to the shop floor and the site double-counts."""
    plugin = OccupancyPlugin(_geometry(_zone(TILL_QUEUE, role="queue", metrics=["occupancy"])))

    assert plugin.reduce([_sample(raw=5.0, confirmed=5.0, dt_s=0.5, zone=TILL_QUEUE)], BUCKET) == []


# --- Dwell ------------------------------------------------------------------


def test_dwell_reports_the_mean_of_the_durations_that_closed() -> None:
    plugin = DwellPlugin(_geometry())

    rows = plugin.reduce([_dwell(10.0), _dwell(20.0, track=2)], BUCKET)

    assert _by_metric(rows) == {(MetricName.DWELL_SECONDS, ScopeId(FLOOR)): 15.0}


def test_dwell_reports_how_many_stays_made_the_mean() -> None:
    """A mean of one is a data point; the tail is where dwell's signal lives (§7)."""
    plugin = DwellPlugin(_geometry())

    rows = plugin.reduce([_dwell(10.0), _dwell(20.0, track=2)], BUCKET)

    assert [row.sample_count for row in rows] == [2]


def test_a_bucket_with_no_completed_dwell_produces_no_row() -> None:
    """Dwells attribute to the minute they *ended* in, so most minutes have none."""
    plugin = DwellPlugin(_geometry())

    assert plugin.reduce([_sample(raw=3.0, confirmed=3.0, dt_s=0.5)], BUCKET) == []


def test_a_zone_that_does_not_ask_for_dwell_produces_no_row() -> None:
    plugin = DwellPlugin(_geometry(_zone(metrics=["occupancy"])))

    assert plugin.reduce([_dwell(10.0)], BUCKET) == []


# --- Wiring -----------------------------------------------------------------


def test_the_registry_is_built_from_the_core_six_and_nothing_else() -> None:
    """The seam is only real if the core six go through it (engine-architecture.md §17).

    Heatmap is P4.1 and is absent rather than stubbed — a registered plugin that emits
    nothing would read as a working metric reporting no activity.
    """
    registry = build_registry(_geometry())

    assert registry.metrics() == frozenset(
        {
            MetricName.FOOTFALL,
            MetricName.LINE_CROSS,
            MetricName.OCCUPANCY,
            MetricName.OCCUPANCY_RAW,
            MetricName.QUEUE_LEN,
            MetricName.QUEUE_LEN_RAW,
            MetricName.DWELL_SECONDS,
            MetricName.CONVERSION,
        }
    )


def test_a_site_with_no_till_still_builds_and_simply_omits_conversion() -> None:
    """No POS configured is not an error; footfall is still produced (§9)."""
    registry = build_registry(_geometry())

    rows = registry.reduce_all([_dwell(10.0)], BUCKET)

    assert [row.metric for row in rows] == [MetricName.DWELL_SECONDS]
