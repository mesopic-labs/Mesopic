"""Zone heatmaps — the grid blob, and the `Δt` weighting that makes it a picture of the floor.

The whole risk in this metric is the accumulation unit, and it is not a rounding matter.
A `+= 1` per sampled frame renders *the load controller*: the adaptive sampler slows down
when the scene is busy, so a frame-counted grid is brightest during the quiet periods and
dimmest during the rush — exactly inverted from what the customer is being sold
(algorithms.md §0.6, §10). `test_the_grid_is_weighted_by_time_not_by_frame_count` drives
the same stationary person at two different sampling rates and asserts the two grids are
identical; a frame-counting implementation passes every other test in this file.

Two further properties are pinned here because both fail silently:

* **Accumulation is in float deciseconds, rounded once at pack time.** Rounding per hit
  loses everything at high frame rates — at 20 fps a hit is 0.5 ds and `round(0.5)` is
  `0` under banker's rounding, so the grid would stay empty while the code looked right.
* **The blob is explicitly little-endian.** It syncs to the cloud (ADR-0010), so native
  byte order would make the wire format depend on which box wrote it.

Red-first for P4.1.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from mesopic.analytics.geometry import GeometryAnalytics
from mesopic.analytics.metrics.heatmap import (
    MAX_CELL,
    HeatmapAccumulator,
    pack_counts,
    unpack_counts,
)
from mesopic.analytics.site_geometry import GRID_H, GRID_W, SiteGeometry, cell_of
from mesopic.config.schema import MesopicConfig
from mesopic.types import (
    CameraId,
    EventKind,
    FrameTs,
    HeatmapRow,
    MinuteBucket,
    RawEvent,
    Track,
    TrackId,
    ZoneId,
)

CAMERA = CameraId("front-door")
FLOOR = ZoneId("floor")
BACK = ZoneId("back-room")

T0 = datetime(2026, 8, 18, 9, 30, tzinfo=UTC)
BUCKET = MinuteBucket(T0)

SQUARE = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
STANDING = (0.5, 0.5)


# --- Fixtures ---------------------------------------------------------------


def _config(zones: list[dict[str, Any]]) -> MesopicConfig:
    return MesopicConfig.model_validate(
        {
            "site": {"site_id": "test-site"},
            "cameras": [
                {
                    "camera_id": CAMERA,
                    "name": "Front door",
                    "source": {"kind": "rtsp", "url_env": "MESOPIC_TEST_RTSP"},
                    "reference_resolution": [1920, 1080],
                }
            ],
            "zones": zones,
        }
    )


def _zone(zone_id: str = FLOOR, *, metrics: list[str] | None = None) -> dict[str, Any]:
    return {
        "zone_id": zone_id,
        "camera_id": CAMERA,
        "polygon": SQUARE,
        "role": "area",
        "metrics": ["heatmap"] if metrics is None else metrics,
    }


def _geometry(zones: list[dict[str, Any]] | None = None) -> SiteGeometry:
    return SiteGeometry.compile(_config([_zone()] if zones is None else zones))


def _track(
    foot: tuple[float, float] = STANDING,
    *,
    ts: datetime = T0,
    track_id: int = 1,
    coasted: bool = False,
) -> Track:
    return Track(
        camera_id=CAMERA,
        track_id=TrackId(track_id),
        ts=FrameTs(ts),
        foot_point=foot,
        score=0.9,
        time_since_update=1 if coasted else 0,
    )


def _hit(cell: tuple[int, int], dt_s: float, *, zone_id: ZoneId = FLOOR) -> RawEvent:
    return RawEvent(
        camera_id=CAMERA,
        ts=FrameTs(T0),
        kind=EventKind.HEATMAP_HIT,
        track_id=TrackId(1),
        zone_id=zone_id,
        cell=cell,
        dt_s=dt_s,
    )


def _cell_value(row: HeatmapRow, cell: tuple[int, int]) -> int:
    column, line = cell
    return unpack_counts(row.counts)[line * row.grid_w + column]


def _drive(analytics: GeometryAnalytics, *, fps: float, seconds: float) -> list[RawEvent]:
    """Stand one person still for `seconds`, sampled at `fps`. Returns every event."""
    events: list[RawEvent] = []
    step = 1.0 / fps
    ticks = round(seconds / step) + 1
    for index in range(ticks):
        ts = T0 + timedelta(seconds=index * step)
        events += analytics.on_tracks(CAMERA, [_track(ts=ts)], ts=FrameTs(ts))
    return events


# --- The grid ---------------------------------------------------------------


def test_a_foot_point_lands_in_the_cell_its_coordinates_name() -> None:
    assert cell_of((0.0, 0.0)) == (0, 0)
    assert cell_of((0.5, 0.5)) == (GRID_W // 2, GRID_H // 2)


def test_the_far_edge_stays_inside_the_grid() -> None:
    """`floor(1.0 * 32)` is 32, one past the last cell. A clamp, not an IndexError."""
    assert cell_of((1.0, 1.0)) == (GRID_W - 1, GRID_H - 1)


def test_the_blob_is_little_endian_whatever_the_box_is() -> None:
    """The blob syncs (ADR-0010), so native byte order would make the wire host-dependent."""
    assert pack_counts([1.0, 258.0], width=2, height=1) == b"\x01\x00\x02\x01"


def test_the_blob_round_trips() -> None:
    counts = [0.0, 7.4, 7.6, 0.0]
    assert unpack_counts(pack_counts(counts, width=2, height=2)) == (0, 7, 8, 0)


def test_a_cell_saturates_rather_than_wrapping() -> None:
    """65536 ds must read as "very hot", never as a cold cell (algorithms.md §10)."""
    assert unpack_counts(pack_counts([70000.0], width=1, height=1)) == (MAX_CELL,)


# --- The fold ---------------------------------------------------------------


def test_a_hit_deposits_deciseconds_of_presence() -> None:
    rows = HeatmapAccumulator(_geometry()).fold([_hit((4, 6), 2.0)], BUCKET)
    assert len(rows) == 1
    assert _cell_value(rows[0], (4, 6)) == 20  # 2.0 s == 20 ds


def test_deciseconds_accumulate_before_they_are_rounded() -> None:
    """Rounding each hit loses everything at high fps: `round(0.5)` is `0` (§10)."""
    hits = [_hit((4, 6), 0.05) for _ in range(20)]  # 20 Hz for one second
    rows = HeatmapAccumulator(_geometry()).fold(hits, BUCKET)
    assert _cell_value(rows[0], (4, 6)) == 10


def test_a_zone_without_the_heatmap_metric_produces_no_grid() -> None:
    geometry = _geometry([_zone(FLOOR, metrics=["occupancy"])])
    assert HeatmapAccumulator(geometry).fold([_hit((4, 6), 2.0)], BUCKET) == []


def test_a_zone_nobody_entered_produces_no_row_at_all() -> None:
    """An empty zone renders cold from a missing row. Writing 2 KB of zeros a minute
    per zone to say "nothing happened" is a cost with no reader."""
    geometry = _geometry([_zone(FLOOR), _zone(BACK)])
    rows = HeatmapAccumulator(geometry).fold([_hit((4, 6), 2.0)], BUCKET)
    assert [row.zone_id for row in rows] == [FLOOR]


def test_folding_a_bucket_twice_produces_the_same_grid() -> None:
    """Buckets are re-foldable (P2.6), so the grid must be replay-safe like the scalars."""
    accumulator = HeatmapAccumulator(_geometry())
    hits = [_hit((4, 6), 2.0)]
    assert accumulator.fold(hits, BUCKET)[0].counts == accumulator.fold(hits, BUCKET)[0].counts


# --- Emission ---------------------------------------------------------------


def test_a_track_inside_a_heatmap_zone_deposits_heat_at_its_cell() -> None:
    analytics = GeometryAnalytics(_geometry())
    analytics.on_tracks(CAMERA, [_track()], ts=FrameTs(T0))
    later = T0 + timedelta(seconds=1)
    events = analytics.on_tracks(CAMERA, [_track(ts=later)], ts=FrameTs(later))

    hits = [event for event in events if event.kind is EventKind.HEATMAP_HIT]
    assert len(hits) == 1
    assert hits[0].cell == cell_of(STANDING)
    assert hits[0].dt_s == pytest.approx(1.0)


def test_the_first_tick_of_a_run_deposits_no_heat() -> None:
    """There is no previous frame, so the interval it would stand for is unknown."""
    analytics = GeometryAnalytics(_geometry())
    events = analytics.on_tracks(CAMERA, [_track()], ts=FrameTs(T0))
    assert [event for event in events if event.kind is EventKind.HEATMAP_HIT] == []


def test_a_coasted_track_deposits_no_heat() -> None:
    """algorithms.md §3.4(2): an unobserved position produces no heatmap hit. Heat is
    evidence of a person having been seen somewhere, not of the filter's best guess."""
    analytics = GeometryAnalytics(_geometry())
    analytics.on_tracks(CAMERA, [_track()], ts=FrameTs(T0))
    later = T0 + timedelta(seconds=1)
    events = analytics.on_tracks(CAMERA, [_track(ts=later, coasted=True)], ts=FrameTs(later))
    assert [event for event in events if event.kind is EventKind.HEATMAP_HIT] == []


def test_the_grid_is_weighted_by_time_not_by_frame_count() -> None:
    """The load-bearing test. One person, standing still for the same 60 seconds, sampled
    at 1 fps and at 5 fps. A `+= 1` accumulator reads 5x hotter for the faster sampling —
    and since the sampler slows down exactly when the scene is busy, that grid is a
    picture of the load controller rather than of the floor (§0.6, §10)."""
    slow = _drive(GeometryAnalytics(_geometry()), fps=1.0, seconds=60.0)
    fast = _drive(GeometryAnalytics(_geometry()), fps=5.0, seconds=60.0)

    accumulator = HeatmapAccumulator(_geometry())
    slow_row = accumulator.fold(slow, BUCKET)[0]
    fast_row = accumulator.fold(fast, BUCKET)[0]

    assert _cell_value(slow_row, cell_of(STANDING)) == 600  # 60 s of presence
    assert slow_row.counts == fast_row.counts
