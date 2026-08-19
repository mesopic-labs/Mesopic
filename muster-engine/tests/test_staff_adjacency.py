"""Staff-vs-customer tagging, and the sub-count every metric splits into (adjacency A2).

The whole feature is one polygon test done once, at the right moment, on the right
foot-point — and each of those three qualifiers is a way to get it wrong that still looks
like it works:

* **Once.** The tag is decided at birth and sticks. Re-deciding per tick would untag a
  staff member the moment they stepped onto the shop floor, which is most of their day.
* **At the right moment.** A track's first sighting may be dead reckoning, and a Kalman
  guess must not assign someone a role. The decision defers to the first observation.
* **On origin, not position.** A customer leaning over the counter is inside the staff
  zone. Testing where a track *is* rather than where it *came from* tags them staff —
  and it is exactly the case that makes the naive implementation look right in a demo,
  because staff really are behind the counter most of the time.

`value` remains the **total, staff included**, and `staff_value` is the staff portion, so
customers are `value - staff_value`. Decided 2026-08-18: the alternative silently changes
the meaning of every row already stored and of the ADR-0010 sync contract.

Red-first for P4.2.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from muster.aggregator.aggregator import Aggregator
from muster.analytics.geometry import GeometryAnalytics
from muster.analytics.metrics import build_registry
from muster.analytics.site_geometry import SiteGeometry
from muster.api.board import Cohort, tiles_for
from muster.config.schema import MusterConfig
from muster.types import (
    CameraId,
    EventKind,
    FrameTs,
    MetricName,
    MetricRow,
    MinuteBucket,
    RawEvent,
    Track,
    TrackId,
    ZoneId,
)

CAMERA = CameraId("front-door")
FLOOR = ZoneId("shop-floor")
BACK = ZoneId("behind-counter")
DOOR = "door-count"

T0 = datetime(2026, 8, 18, 9, 30, tzinfo=UTC)
BUCKET = MinuteBucket(T0)

# The floor is the left half, the staff area the right half. They do not overlap, so a
# foot-point is unambiguously in one or the other.
FLOOR_POLY = [[0.0, 0.0], [0.45, 0.0], [0.45, 1.0], [0.0, 1.0]]
BACK_POLY = [[0.55, 0.0], [1.0, 0.0], [1.0, 1.0], [0.55, 1.0]]

ON_FLOOR = (0.2, 0.5)
BEHIND_COUNTER = (0.8, 0.5)

# A vertical line between the two, crossed by walking from the counter onto the floor.
LINE_A = [0.5, 0.0]
LINE_B = [0.5, 1.0]


# --- Fixtures ---------------------------------------------------------------


def _config(*, staff_zone: bool = True) -> MusterConfig:
    zones: list[dict[str, Any]] = [
        {
            "zone_id": FLOOR,
            "camera_id": CAMERA,
            "role": "area",
            "polygon": FLOOR_POLY,
            "metrics": ["occupancy", "dwell_seconds"],
        }
    ]
    if staff_zone:
        zones.append(
            {
                "zone_id": BACK,
                "camera_id": CAMERA,
                "role": "staff",
                "polygon": BACK_POLY,
                "metrics": [],
            }
        )
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
            "lines": [
                {
                    "line_id": DOOR,
                    "camera_id": CAMERA,
                    "a": LINE_A,
                    "b": LINE_B,
                    "positive_dir": "in",
                    "metrics": ["line_cross", "footfall"],
                }
            ],
            "zones": zones,
        }
    )


def _analytics(*, staff_zone: bool = True) -> GeometryAnalytics:
    return GeometryAnalytics(SiteGeometry.compile(_config(staff_zone=staff_zone)))


def _track(
    foot: tuple[float, float],
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


def _walk(analytics: GeometryAnalytics, *steps: tuple[tuple[float, float], bool]) -> list[RawEvent]:
    """Walk one track through `(foot_point, observed)` steps, a second apart."""
    events: list[RawEvent] = []
    for index, (foot, observed) in enumerate(steps):
        ts = T0 + timedelta(seconds=index)
        events += analytics.on_tracks(
            CAMERA, [_track(foot, ts=ts, coasted=not observed)], ts=FrameTs(ts)
        )
    return events


# --- The tag ----------------------------------------------------------------


def test_a_track_born_in_a_staff_zone_is_tagged() -> None:
    assert _walk_tag(BEHIND_COUNTER) is True


def test_a_track_born_on_the_shop_floor_is_not_tagged() -> None:
    assert _walk_tag(ON_FLOOR) is False


def _walk_tag(origin: tuple[float, float]) -> bool:
    """The tag as the events report it: walk from `origin` onto the floor and read the
    zone enter, which is the first event a track on the floor produces."""
    analytics = _analytics()
    events = _walk(analytics, (origin, True), (ON_FLOOR, True))
    enters = [e for e in events if e.kind is EventKind.ZONE_ENTER and e.zone_id == FLOOR]
    assert len(enters) == 1, "the walk should enter the floor exactly once"
    return enters[0].is_staff


def test_the_tag_sticks_when_staff_walk_out_onto_the_floor() -> None:
    """Most of a shift is spent out from behind the counter. Re-deciding per tick would
    untag them the moment they step out, which is when they inflate the numbers."""
    analytics = _analytics()
    events = _walk(
        analytics,
        (BEHIND_COUNTER, True),
        (ON_FLOOR, True),
        (ON_FLOOR, True),
    )
    crossings = [e for e in events if e.kind is EventKind.LINE_CROSS]
    assert crossings, "walking out from behind the counter should cross the line"
    assert all(event.is_staff for event in crossings)


def test_a_customer_who_walks_into_a_staff_zone_is_never_tagged() -> None:
    """Origin, not position. Leaning over the counter puts a customer's foot-point in the
    staff zone, and this is the case a position test gets wrong while looking correct."""
    analytics = _analytics()
    events = _walk(analytics, (ON_FLOOR, True), (BEHIND_COUNTER, True))
    assert not any(event.is_staff for event in events)


def test_a_coasted_first_sighting_defers_the_decision() -> None:
    """A dead-reckoned position must not assign a role (algorithms.md §3.4). The track is
    first *observed* on the floor, so it is a customer however it was first guessed."""
    analytics = _analytics()
    events = _walk(analytics, (BEHIND_COUNTER, False), (ON_FLOOR, True), (ON_FLOOR, True))
    assert not any(event.is_staff for event in events)


def test_a_site_with_no_staff_zone_tags_nobody() -> None:
    analytics = _analytics(staff_zone=False)
    events = _walk(analytics, (BEHIND_COUNTER, True), (ON_FLOOR, True))
    assert not any(event.is_staff for event in events)


def test_a_track_id_reused_after_death_is_classified_afresh() -> None:
    """`TrackId` is unique per run, not forever — a dead track's state is dropped, so the
    next holder of the number must not inherit a role it never earned."""
    analytics = _analytics()
    _walk(analytics, (BEHIND_COUNTER, True))
    analytics.on_tracks(CAMERA, [], ts=FrameTs(T0 + timedelta(seconds=1)))

    later = T0 + timedelta(seconds=2)
    events = analytics.on_tracks(CAMERA, [_track(ON_FLOOR, ts=later)], ts=FrameTs(later))
    events += analytics.on_tracks(
        CAMERA,
        [_track(ON_FLOOR, ts=later + timedelta(seconds=1))],
        ts=FrameTs(later + timedelta(seconds=1)),
    )
    assert not any(event.is_staff for event in events)


def test_a_confirmed_zone_entry_carries_the_tag() -> None:
    """Zone-derived footfall counts *confirmed* entries, so a confirmation with no tag is a
    staff arrival counted as a customer on every camera that has no counting line. It is
    the one path `staff_only` cannot rescue: the flag it filters on was never set."""
    analytics = _analytics()

    events = _walk(analytics, (BEHIND_COUNTER, True), *([(ON_FLOOR, True)] * 5))

    confirmed = [e for e in events if e.kind is EventKind.ZONE_CONFIRMED and e.zone_id == FLOOR]
    assert confirmed, "staying on the floor past dwell_min_s should confirm the entry"
    assert all(event.is_staff for event in confirmed)


def test_a_customer_confirmation_is_left_untagged() -> None:
    """The other half of the same contract: confirming an entry must not invent a role."""
    analytics = _analytics()

    events = _walk(analytics, *([(ON_FLOOR, True)] * 6))

    confirmed = [e for e in events if e.kind is EventKind.ZONE_CONFIRMED and e.zone_id == FLOOR]
    assert confirmed, "staying on the floor past dwell_min_s should confirm the entry"
    assert not any(event.is_staff for event in confirmed)


# --- Sampled state ----------------------------------------------------------


def test_an_occupancy_sample_counts_its_staff_separately() -> None:
    """A sample names a zone and no track, so there is no per-track flag to filter on —
    the staff count has to ride the sample, as `confirmed_value` already does."""
    analytics = _analytics()
    staff = _track(BEHIND_COUNTER, track_id=1)
    customer = _track(ON_FLOOR, track_id=2)
    analytics.on_tracks(CAMERA, [staff, customer], ts=FrameTs(T0))

    later = T0 + timedelta(seconds=1)
    events = analytics.on_tracks(
        CAMERA,
        [_track(ON_FLOOR, ts=later, track_id=1), _track(ON_FLOOR, ts=later, track_id=2)],
        ts=FrameTs(later),
    )
    (sample,) = [
        event
        for event in events
        if event.kind is EventKind.OCCUPANCY_SAMPLE and event.zone_id == FLOOR
    ]
    assert sample.value == 2.0, "the total still counts everyone"
    assert sample.staff_value == 1.0


# --- The split --------------------------------------------------------------


def _rows(events: list[RawEvent]) -> dict[tuple[MetricName, str | None], MetricRow]:
    """Rows by `(metric, scope)`, with the scope as a plain string so a test can index it
    with the `ZoneId` and `LineId` constants above without casting at every call."""
    registry = build_registry(SiteGeometry.compile(_config()))
    return {
        (row.metric, None if row.scope_id is None else str(row.scope_id)): row
        for row in registry.reduce_all(events, BUCKET)
    }


def _crossing(*, is_staff: bool, track_id: int) -> RawEvent:
    return RawEvent(
        camera_id=CAMERA,
        ts=FrameTs(T0),
        kind=EventKind.LINE_CROSS,
        track_id=TrackId(track_id),
        line_id=DOOR,  # type: ignore[arg-type]
        direction=1,
        is_staff=is_staff,
    )


def test_a_count_reports_the_total_and_the_staff_portion() -> None:
    """`value` is everyone. Customers are the subtraction, done by the reader — the
    alternative changes what every stored row already means."""
    rows = _rows([_crossing(is_staff=True, track_id=1), _crossing(is_staff=False, track_id=2)])
    crossings = rows[(MetricName.LINE_CROSS, DOOR)]
    assert crossings.value == 2.0
    assert crossings.staff_value == 1.0


def test_footfall_splits_the_same_way() -> None:
    rows = _rows([_crossing(is_staff=True, track_id=1), _crossing(is_staff=False, track_id=2)])
    footfall = rows[(MetricName.FOOTFALL, DOOR)]
    assert footfall.value == 2.0
    assert footfall.staff_value == 1.0


def test_a_bucket_with_no_staff_reports_zero_not_none() -> None:
    """Zero staff is a measurement; `None` means the split was never computed, and the
    dashboard renders the two differently on purpose."""
    rows = _rows([_crossing(is_staff=False, track_id=1)])
    assert rows[(MetricName.LINE_CROSS, DOOR)].staff_value == 0.0


def test_conversion_carries_no_staff_split() -> None:
    """A ratio has no staff portion, and its denominator is still total footfall — that
    is P3.5's to revisit when the till ingress lands."""
    rows = _rows([_crossing(is_staff=True, track_id=1)])
    conversion = [row for (metric, _), row in rows.items() if metric is MetricName.CONVERSION]
    assert all(row.staff_value is None for row in conversion)


def test_occupancy_reports_a_staff_mean_and_a_staff_peak() -> None:
    samples = [
        RawEvent(
            camera_id=CAMERA,
            ts=FrameTs(T0 + timedelta(seconds=index)),
            kind=EventKind.OCCUPANCY_SAMPLE,
            track_id=None,
            zone_id=FLOOR,
            value=4.0,
            confirmed_value=4.0,
            staff_value=1.0,
            staff_confirmed_value=1.0,
            dt_s=1.0,
        )
        for index in range(3)
    ]
    rows = _rows(samples)
    assert rows[(MetricName.OCCUPANCY, FLOOR)].value == pytest.approx(4.0)
    assert rows[(MetricName.OCCUPANCY, FLOOR)].staff_value == pytest.approx(1.0)
    assert rows[(MetricName.OCCUPANCY_RAW, FLOOR)].staff_value == pytest.approx(1.0)


# --- Dwell ------------------------------------------------------------------


def test_a_completed_dwell_remembers_whose_it_was() -> None:
    """The dwell sample is built by the aggregator, not by geometry, so it has to carry
    the flag forward from the entry that opened it or every dwell reads as a customer."""
    aggregator = Aggregator(
        build_registry(SiteGeometry.compile(_config())), dwell_min_s=1.0, exit_grace_s=0.5
    )
    entered = RawEvent(
        camera_id=CAMERA,
        ts=FrameTs(T0),
        kind=EventKind.ZONE_ENTER,
        track_id=TrackId(1),
        zone_id=FLOOR,
        is_staff=True,
    )
    aggregator.ingest(entered)
    aggregator.ingest(
        RawEvent(
            camera_id=CAMERA,
            ts=FrameTs(T0 + timedelta(seconds=5)),
            kind=EventKind.ZONE_EXIT,
            track_id=TrackId(1),
            zone_id=FLOOR,
            is_staff=True,
        )
    )
    aggregator.flush(FrameTs(T0 + timedelta(seconds=10)))

    rows = {
        (row.metric, None if row.scope_id is None else str(row.scope_id)): row
        for row in aggregator.close_bucket(BUCKET)
    }
    dwell = rows[(MetricName.DWELL_SECONDS, FLOOR)]
    assert dwell.value == pytest.approx(5.0)
    assert dwell.staff_value == pytest.approx(5.0)


def test_a_zone_with_no_staff_dwell_reports_no_staff_mean_rather_than_zero() -> None:
    """A count of nobody is zero; the average of nothing is not. `0.0` here would claim
    staff stayed for no time at all, which is a different and wronger statement."""
    sample = RawEvent(
        camera_id=CAMERA,
        ts=FrameTs(T0),
        kind=EventKind.DWELL_SAMPLE,
        track_id=TrackId(1),
        zone_id=FLOOR,
        value=12.0,
        is_staff=False,
    )
    dwell = _rows([sample])[(MetricName.DWELL_SECONDS, FLOOR)]
    assert dwell.value == pytest.approx(12.0)
    assert dwell.staff_value is None


# --- Cameras that cannot measure staff at all (P4.7) -------------------------
#
# Every fixture above puts the staff zone on the only camera there is, so `value` and
# `staff_value` are always both measurable. A real site is mixed: the till camera sees the
# counter, the stockroom camera never does. A track on a camera with no `role: staff` zone
# can never be tagged, whoever they are — so a `0.0` there is not a count of no staff, it
# is a measurement that was not made, and ADR-0021's count-is-zero rule does not reach it.

OTHER = CameraId("side-door")
AISLE = ZoneId("aisle")
QUEUE = ZoneId("till-queue")
SIDE = "side-count"

AISLE_POLY = [[0.0, 0.0], [0.45, 0.0], [0.45, 1.0], [0.0, 1.0]]
QUEUE_POLY = [[0.55, 0.0], [1.0, 0.0], [1.0, 1.0], [0.55, 1.0]]


def _mixed_config() -> MusterConfig:
    """Two cameras, and only one of them can see the staff area.

    The configuration nothing else in the suite builds, and the only one in which
    "does this site have a staff zone" and "can this camera measure staff" differ.
    """
    base = _config().model_dump()
    base["cameras"].append(
        {
            "camera_id": OTHER,
            "name": "Side door",
            "source": {"kind": "rtsp", "url_env": "MUSTER_TEST_RTSP"},
            "reference_resolution": [1920, 1080],
        }
    )
    base["lines"].append(
        {
            "line_id": SIDE,
            "camera_id": OTHER,
            "a": LINE_A,
            "b": LINE_B,
            "positive_dir": "in",
            "metrics": ["line_cross", "footfall"],
        }
    )
    base["zones"] += [
        {
            "zone_id": AISLE,
            "camera_id": OTHER,
            "role": "area",
            "polygon": AISLE_POLY,
            "metrics": ["occupancy", "dwell_seconds"],
        },
        {
            "zone_id": QUEUE,
            "camera_id": OTHER,
            "role": "queue",
            "polygon": QUEUE_POLY,
            "metrics": ["queue_len"],
        },
    ]
    return MusterConfig.model_validate(base)


def _mixed_rows(events: list[RawEvent]) -> dict[tuple[str, MetricName, str | None], MetricRow]:
    """Rows by `(camera, metric, scope)` — the camera is what this section is about."""
    registry = build_registry(SiteGeometry.compile(_mixed_config()))
    return {
        (str(row.camera_id), row.metric, None if row.scope_id is None else str(row.scope_id)): row
        for row in registry.reduce_all(events, BUCKET)
    }


def _side_crossing(*, track_id: int) -> RawEvent:
    """A crossing on the camera with no staff zone. `is_staff` is False because nothing
    could ever have set it, which is exactly the point."""
    return RawEvent(
        camera_id=OTHER,
        ts=FrameTs(T0),
        kind=EventKind.LINE_CROSS,
        track_id=TrackId(track_id),
        line_id=SIDE,  # type: ignore[arg-type]
        direction=1,
        is_staff=False,
    )


def _sample(zone_id: ZoneId, *, index: int) -> RawEvent:
    """An occupancy sample as `GeometryAnalytics` emits it on a staff-less camera: a
    staff count of zero, because counting nobody is all it can do."""
    return RawEvent(
        camera_id=OTHER,
        ts=FrameTs(T0 + timedelta(seconds=index)),
        kind=EventKind.OCCUPANCY_SAMPLE,
        track_id=None,
        zone_id=zone_id,
        value=3.0,
        confirmed_value=3.0,
        staff_value=0.0,
        staff_confirmed_value=0.0,
        dt_s=1.0,
    )


def test_site_geometry_knows_which_cameras_can_measure_staff() -> None:
    geometry = SiteGeometry.compile(_mixed_config())
    assert geometry.has_staff_zone(CAMERA) is True
    assert geometry.has_staff_zone(OTHER) is False


def test_asking_an_unknown_camera_about_staff_raises() -> None:
    """The same contract every other accessor keeps (P2.2): a camera that silently has no
    geometry is a camera that silently stops counting, and answering `False` here would
    make an unknown camera indistinguishable from a stockroom."""
    geometry = SiteGeometry.compile(_mixed_config())
    with pytest.raises(KeyError):
        geometry.has_staff_zone(CameraId("no-such-camera"))


def test_a_camera_with_no_staff_zone_reports_no_footfall_split() -> None:
    rows = _mixed_rows([_side_crossing(track_id=1), _side_crossing(track_id=2)])
    footfall = rows[(OTHER, MetricName.FOOTFALL, SIDE)]
    assert footfall.value == 2.0, "the total is measured and unaffected"
    assert footfall.staff_value is None


def test_a_camera_with_no_staff_zone_reports_no_line_cross_split() -> None:
    rows = _mixed_rows([_side_crossing(track_id=1)])
    assert rows[(OTHER, MetricName.LINE_CROSS, SIDE)].staff_value is None


def test_a_camera_with_no_staff_zone_reports_no_occupancy_split() -> None:
    """Both series, because both are read off the same unmeasurable samples."""
    rows = _mixed_rows([_sample(AISLE, index=index) for index in range(3)])
    assert rows[(OTHER, MetricName.OCCUPANCY, AISLE)].value == pytest.approx(3.0)
    assert rows[(OTHER, MetricName.OCCUPANCY, AISLE)].staff_value is None
    assert rows[(OTHER, MetricName.OCCUPANCY_RAW, AISLE)].staff_value is None


def test_a_camera_with_no_staff_zone_reports_no_queue_split() -> None:
    rows = _mixed_rows([_sample(QUEUE, index=index) for index in range(3)])
    assert rows[(OTHER, MetricName.QUEUE_LEN, QUEUE)].staff_value is None
    assert rows[(OTHER, MetricName.QUEUE_LEN_RAW, QUEUE)].staff_value is None


def test_the_staffed_camera_on_the_same_site_still_splits() -> None:
    """The half that stops this being a blanket `None`. One site, one bucket, two cameras:
    the one that can see the counter still answers, and its zero is a real zero."""
    rows = _mixed_rows(
        [
            _crossing(is_staff=True, track_id=1),
            _crossing(is_staff=False, track_id=2),
            _side_crossing(track_id=3),
        ]
    )
    assert rows[(CAMERA, MetricName.LINE_CROSS, DOOR)].staff_value == 1.0
    assert rows[(OTHER, MetricName.LINE_CROSS, SIDE)].staff_value is None


def test_a_camera_with_no_staff_zone_reports_no_dwell_split() -> None:
    """Already true before P4.7, and pinned here so it stays true for the stated reason
    rather than by accident: dwell is a mean, so it reported `None` for want of a staff
    sample whether or not the camera could have produced one."""
    sample = RawEvent(
        camera_id=OTHER,
        ts=FrameTs(T0),
        kind=EventKind.DWELL_SAMPLE,
        track_id=TrackId(1),
        zone_id=AISLE,
        value=9.0,
        is_staff=False,
    )
    dwell = _mixed_rows([sample])[(OTHER, MetricName.DWELL_SECONDS, AISLE)]
    assert dwell.value == pytest.approx(9.0)
    assert dwell.staff_value is None


def test_a_mixed_site_renders_one_camera_split_and_the_other_absent() -> None:
    """The whole chain the card is about, end to end: two cameras, one bucket, one cohort.

    The staffed camera answers "how many customers" with a number. The staff-less one
    answers with an em dash, because it was never in a position to answer at all — and a
    `0.0` there would have been the tile confidently reporting a measurement nobody made.
    """
    config = _mixed_config()
    registry = build_registry(SiteGeometry.compile(config))
    rows = list(
        registry.reduce_all(
            [
                _crossing(is_staff=True, track_id=1),
                _crossing(is_staff=False, track_id=2),
                _side_crossing(track_id=3),
            ],
            BUCKET,
        )
    )
    tiles = {
        (str(tile.camera_id), tile.metric, str(tile.scope_id)): tile
        for tile in tiles_for(config, rows=rows, cohort=Cohort.CUSTOMERS)
    }
    assert tiles[(CAMERA, MetricName.FOOTFALL, DOOR)].value == pytest.approx(1.0)
    assert tiles[(OTHER, MetricName.FOOTFALL, SIDE)].value is None
