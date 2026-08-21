"""The metrics that count things: footfall, line crossings, and conversion.

All three are event counts rather than sampled states, so `Δt`-weighting does not apply
to any of them (algorithms.md §0.6) — what does apply is *uniqueness*. Footfall is
distinct entering tracks, not crossings, or one person pacing a doorway becomes a busy
morning.

The scope decision lives here and is load-bearing downstream: footfall rows are scoped to
the geometry that produced them — the line, or the zone where a camera has no line — and
never to the camera as a whole. Two doors sum correctly; a camera-wide row would be the
same quantity counted a second way, and `mesopic.truth`'s scorer refuses the mix rather
than double it.

Red-first for P2.4.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from mesopic.analytics.metrics.conversion import ConversionPlugin
from mesopic.analytics.metrics.footfall import FootfallPlugin
from mesopic.analytics.metrics.line_cross import LineCrossPlugin
from mesopic.analytics.site_geometry import SiteGeometry
from mesopic.config.schema import MesopicConfig
from mesopic.types import (
    CameraId,
    EventKind,
    FrameTs,
    LineId,
    MetricRow,
    MinuteBucket,
    RawEvent,
    ScopeId,
    TrackId,
    ZoneId,
)

CAMERA = CameraId("front-door")
SIDE = CameraId("side-door")

FRONT = LineId("front")
BACK = LineId("back")
FLOOR = ZoneId("floor")

T0 = datetime(2026, 8, 16, 9, 30, tzinfo=UTC)
BUCKET = MinuteBucket(T0)

SQUARE = [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]]


def _config(*, lines: list[dict[str, Any]], zones: list[dict[str, Any]]) -> MesopicConfig:
    return MesopicConfig.model_validate(
        {
            "site": {"site_id": "test-site"},
            "cameras": [
                {
                    "camera_id": camera,
                    "name": camera,
                    "source": {"kind": "rtsp", "url_env": "MESOPIC_TEST_RTSP"},
                    "reference_resolution": [1920, 1080],
                }
                for camera in (CAMERA, SIDE)
            ],
            "lines": lines,
            "zones": zones,
        }
    )


def _line(
    line_id: str = FRONT,
    *,
    camera: str = CAMERA,
    positive_dir: str = "in",
    metrics: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "line_id": line_id,
        "camera_id": camera,
        "a": [0.1, 0.5],
        "b": [0.9, 0.5],
        "positive_dir": positive_dir,
        "metrics": ["footfall", "line_cross"] if metrics is None else metrics,
    }


def _zone(
    zone_id: str = FLOOR, *, camera: str = CAMERA, metrics: list[str] | None = None
) -> dict[str, Any]:
    return {
        "zone_id": zone_id,
        "camera_id": camera,
        "polygon": SQUARE,
        "metrics": ["footfall"] if metrics is None else metrics,
    }


def _geometry(
    *, lines: list[dict[str, Any]] | None = None, zones: list[dict[str, Any]] | None = None
) -> SiteGeometry:
    return SiteGeometry.compile(
        _config(lines=[] if lines is None else lines, zones=[] if zones is None else zones)
    )


def _cross(
    *, track: int, line: LineId = FRONT, direction: int = 1, camera: CameraId = CAMERA
) -> RawEvent:
    return RawEvent(
        camera_id=camera,
        ts=FrameTs(T0 + timedelta(seconds=track)),
        kind=EventKind.LINE_CROSS,
        track_id=TrackId(track),
        line_id=line,
        direction=direction,
    )


def _confirmed(*, track: int, zone: ZoneId = FLOOR, camera: CameraId = CAMERA) -> RawEvent:
    return RawEvent(
        camera_id=camera,
        ts=FrameTs(T0 + timedelta(seconds=track)),
        kind=EventKind.ZONE_CONFIRMED,
        track_id=TrackId(track),
        zone_id=zone,
    )


def _sample(*, camera: CameraId = CAMERA, zone: ZoneId = FLOOR, inside: float = 0.0) -> RawEvent:
    """A tick of proof that the camera was alive, carrying nobody."""
    return RawEvent(
        camera_id=camera,
        ts=FrameTs(T0),
        kind=EventKind.OCCUPANCY_SAMPLE,
        track_id=None,
        zone_id=zone,
        value=inside,
        dt_s=0.5,
    )


def _by_scope(rows: list[MetricRow]) -> dict[ScopeId | None, float]:
    return {row.scope_id: row.value for row in rows}


# --- Footfall ---------------------------------------------------------------


def test_an_inbound_crossing_counts_as_one_visit() -> None:
    plugin = FootfallPlugin(_geometry(lines=[_line()]))

    rows = plugin.reduce([_cross(track=1)], BUCKET)

    assert _by_scope(rows) == {ScopeId(FRONT): 1.0}


def test_an_outbound_crossing_is_not_a_visit() -> None:
    """Footfall is people arriving. Counting departures would roughly double it."""
    plugin = FootfallPlugin(_geometry(lines=[_line()]))

    rows = plugin.reduce([_cross(track=1, direction=-1)], BUCKET)

    assert _by_scope(rows) == {ScopeId(FRONT): 0.0}


def test_a_line_drawn_the_other_way_round_counts_its_negative_crossing() -> None:
    """`positive_dir` labels the +1 sense; it does not decide which way is in."""
    plugin = FootfallPlugin(_geometry(lines=[_line(positive_dir="out")]))

    rows = plugin.reduce([_cross(track=1, direction=-1)], BUCKET)

    assert _by_scope(rows) == {ScopeId(FRONT): 1.0}


def test_one_person_crossing_twice_counts_once() -> None:
    """Distinct entering tracks, not crossings — a doorway loiterer is one visit."""
    plugin = FootfallPlugin(_geometry(lines=[_line()]))

    rows = plugin.reduce([_cross(track=1), _cross(track=1)], BUCKET)

    assert _by_scope(rows) == {ScopeId(FRONT): 1.0}


def test_two_doors_are_two_rows_that_sum() -> None:
    """The whole reason footfall is scoped: a two-entrance shop can still ask which."""
    plugin = FootfallPlugin(_geometry(lines=[_line(FRONT), _line(BACK)]))

    rows = plugin.reduce([_cross(track=1), _cross(track=2, line=BACK)], BUCKET)

    assert _by_scope(rows) == {ScopeId(FRONT): 1.0, ScopeId(BACK): 1.0}


def test_no_footfall_row_is_ever_camera_wide() -> None:
    """A camera-wide total is the same count a second way; summing both doubles it."""
    plugin = FootfallPlugin(_geometry(lines=[_line(FRONT), _line(BACK)]))

    rows = plugin.reduce([_cross(track=1), _cross(track=2, line=BACK)], BUCKET)

    assert all(row.scope_id is not None for row in rows)


def test_a_camera_with_no_line_counts_confirmed_zone_entries() -> None:
    """The open-shop-floor fallback: no clean doorway to draw a line across."""
    plugin = FootfallPlugin(_geometry(zones=[_zone()]))

    rows = plugin.reduce([_confirmed(track=1), _confirmed(track=2)], BUCKET)

    assert _by_scope(rows) == {ScopeId(FLOOR): 2.0}


def test_an_unconfirmed_zone_entry_is_not_a_visit() -> None:
    """Someone clipping the corner of the zone has not visited it (§6, §7)."""
    plugin = FootfallPlugin(_geometry(zones=[_zone()]))

    rows = plugin.reduce(
        [RawEvent(CAMERA, FrameTs(T0), EventKind.ZONE_ENTER, TrackId(1), zone_id=FLOOR)], BUCKET
    )

    assert _by_scope(rows) == {ScopeId(FLOOR): 0.0}


def test_a_camera_with_a_line_ignores_its_zones() -> None:
    """A doorway line and a doorway zone would otherwise count the same arrival twice."""
    plugin = FootfallPlugin(_geometry(lines=[_line()], zones=[_zone()]))

    rows = plugin.reduce([_cross(track=1), _confirmed(track=1)], BUCKET)

    assert _by_scope(rows) == {ScopeId(FRONT): 1.0}


def test_geometry_that_does_not_ask_for_footfall_produces_no_row() -> None:
    """Config declares which metrics a line or zone serves; nothing is implicit."""
    plugin = FootfallPlugin(_geometry(lines=[_line(metrics=["line_cross"])]))

    assert plugin.reduce([_cross(track=1)], BUCKET) == []


def test_a_live_camera_with_no_arrivals_reports_zero() -> None:
    """ "Nobody came" and "the camera was down" must not look the same downstream."""
    plugin = FootfallPlugin(_geometry(lines=[_line()]))

    rows = plugin.reduce([_sample()], BUCKET)

    assert _by_scope(rows) == {ScopeId(FRONT): 0.0}


def test_a_camera_absent_from_the_bucket_reports_nothing() -> None:
    """The other half of the same rule: a camera that sent nothing gets no zero either.

    A dead camera reporting a confident zero every minute is how an outage turns into a
    quiet trading day in the customer's chart.
    """
    plugin = FootfallPlugin(_geometry(lines=[_line(FRONT), _line(BACK, camera=SIDE)]))

    rows = plugin.reduce([_cross(track=1)], BUCKET)

    assert _by_scope(rows) == {ScopeId(FRONT): 1.0}


# --- Line crossings ---------------------------------------------------------


def test_crossings_reduce_to_a_signed_net() -> None:
    """Three in and one out is a net of two — the quantity §10 asks for."""
    plugin = LineCrossPlugin(_geometry(lines=[_line()]))

    rows = plugin.reduce(
        [_cross(track=1), _cross(track=2), _cross(track=3), _cross(track=4, direction=-1)],
        BUCKET,
    )

    assert _by_scope(rows) == {ScopeId(FRONT): 2.0}


def test_the_sample_count_is_how_many_crossings_made_the_net() -> None:
    """A net of zero from forty crossings is a busy doorway, not a quiet one."""
    plugin = LineCrossPlugin(_geometry(lines=[_line()]))

    rows = plugin.reduce([_cross(track=1), _cross(track=2, direction=-1)], BUCKET)

    assert [(row.value, row.sample_count) for row in rows] == [(0.0, 2)]


def test_a_line_that_does_not_ask_for_line_cross_produces_no_row() -> None:
    plugin = LineCrossPlugin(_geometry(lines=[_line(metrics=["footfall"])]))

    assert plugin.reduce([_cross(track=1)], BUCKET) == []


# --- Conversion -------------------------------------------------------------


def test_conversion_is_transactions_over_footfall() -> None:
    plugin = ConversionPlugin(
        FootfallPlugin(_geometry(lines=[_line()])), transactions={(CAMERA, BUCKET): 1}
    )

    rows = plugin.reduce(
        [_cross(track=1), _cross(track=2), _cross(track=3), _cross(track=4)], BUCKET
    )

    assert _by_scope(rows) == {None: 0.25}


def test_conversion_is_camera_wide() -> None:
    """Transactions arrive from a till, which belongs to no single door."""
    plugin = ConversionPlugin(
        FootfallPlugin(_geometry(lines=[_line(FRONT), _line(BACK)])),
        transactions={(CAMERA, BUCKET): 3},
    )

    rows = plugin.reduce([_cross(track=1), _cross(track=2, line=BACK)], BUCKET)

    assert _by_scope(rows) == {None: 1.5}


def test_a_missing_till_reading_produces_no_row() -> None:
    """A fabricated zero reads as "nobody bought anything" — a damaging lie (§9)."""
    plugin = ConversionPlugin(FootfallPlugin(_geometry(lines=[_line()])), transactions={})

    assert plugin.reduce([_cross(track=1)], BUCKET) == []


def test_zero_footfall_produces_no_row() -> None:
    """Usually a vision gap while the till ran. Undefined, not zero, not infinite."""
    plugin = ConversionPlugin(
        FootfallPlugin(_geometry(lines=[_line()])), transactions={(CAMERA, BUCKET): 2}
    )

    assert plugin.reduce([_sample()], BUCKET) == []


def test_conversion_above_one_is_reported_rather_than_clamped() -> None:
    """Group baskets and staff purchases make it real; clamping hides a footfall
    undercount that the number is supposed to reveal (§9)."""
    plugin = ConversionPlugin(
        FootfallPlugin(_geometry(lines=[_line()])), transactions={(CAMERA, BUCKET): 3}
    )

    rows = plugin.reduce([_cross(track=1), _cross(track=2)], BUCKET)

    assert _by_scope(rows) == {None: 1.5}
