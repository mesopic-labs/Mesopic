"""Tracks plus geometry become raw events — and, more importantly, do not become noise.

Every test here feeds a hand-built `Track` sequence and reads the events back. There is no
camera, no model and no database in the loop, which is the property engine-architecture.md
§8 asks for: the whole module is a deterministic function of (tracks it has been shown so
far, geometry) with no I/O anywhere in it.

The interesting half is the suppressions. A doorway line will be crossed a few hundred
times a day and jittered across a few thousand, so most of these tests are about the
cases that must emit *nothing*: a graze exactly on the line, a walk past the end of the
segment, jitter inside the hysteresis band, and a segment dead-reckoned at both ends.

Red-first for P2.3.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from muster.analytics.geometry import GeometryAnalytics
from muster.analytics.site_geometry import SiteGeometry
from muster.config.schema import MusterConfig
from muster.types import (
    CameraId,
    EventKind,
    FrameTs,
    LineId,
    NormPoint,
    RawEvent,
    Track,
    TrackId,
    ZoneId,
)

CAMERA = CameraId("front-door")
OTHER_CAMERA = CameraId("till")
T0 = datetime(2026, 8, 16, 12, 0, tzinfo=UTC)

DOOR = LineId("door")
FLOOR = ZoneId("floor")

# A horizontal line across the middle. With y down, "below" (larger y) is the +1 side —
# hand-checked in test_site_geometry.py and relied on here.
LINE_A = [0.1, 0.5]
LINE_B = [0.9, 0.5]

SQUARE = [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]]


def _config(*, lines: list[dict[str, Any]], zones: list[dict[str, Any]]) -> MusterConfig:
    """Two cameras, so "this camera's geometry only" is testable at all."""
    return MusterConfig.model_validate(
        {
            "site": {"site_id": "test-site"},
            "cameras": [
                {
                    "camera_id": CAMERA,
                    "name": "Front door",
                    "source": {"kind": "rtsp", "url_env": "MUSTER_TEST_RTSP"},
                    "reference_resolution": [1920, 1080],
                },
                {
                    "camera_id": OTHER_CAMERA,
                    "name": "Till",
                    "source": {"kind": "frigate", "mqtt_topic": "frigate/till"},
                    "reference_resolution": [1920, 1080],
                },
            ],
            "lines": lines,
            "zones": zones,
        }
    )


def _line(line_id: str = DOOR, camera: str = CAMERA, positive_dir: str = "in") -> dict[str, Any]:
    return {
        "line_id": line_id,
        "camera_id": camera,
        "a": LINE_A,
        "b": LINE_B,
        "positive_dir": positive_dir,
    }


def _zone(zone_id: str = FLOOR, camera: str = CAMERA, polygon: Any = None) -> dict[str, Any]:
    return {
        "zone_id": zone_id,
        "camera_id": camera,
        "polygon": SQUARE if polygon is None else polygon,
    }


def _analytics(
    *,
    lines: list[dict[str, Any]] | None = None,
    zones: list[dict[str, Any]] | None = None,
    **kw: Any,
) -> GeometryAnalytics:
    config = _config(lines=[] if lines is None else lines, zones=[] if zones is None else zones)
    return GeometryAnalytics(SiteGeometry.compile(config), **kw)


def _track(
    point: NormPoint,
    *,
    second: float = 0.0,
    track_id: int = 1,
    coasted: bool = False,
    camera: CameraId = CAMERA,
    is_staff: bool = False,
) -> Track:
    return Track(
        camera_id=camera,
        track_id=TrackId(track_id),
        ts=FrameTs(T0 + timedelta(seconds=second)),
        foot_point=point,
        score=0.9,
        time_since_update=1 if coasted else 0,
        is_staff=is_staff,
    )


def _walk(analytics: GeometryAnalytics, *steps: Track) -> list[RawEvent]:
    """Feed one track per tick and collect everything that came out."""
    events: list[RawEvent] = []
    for step in steps:
        events += analytics.on_tracks(step.camera_id, [step])
    return events


def _kinds(events: list[RawEvent]) -> list[EventKind]:
    return [event.kind for event in events]


# --- Line crossing: what must be counted ------------------------------------


def test_crossing_the_line_emits_one_directional_event() -> None:
    """Above to below is the `+1` sense of a left-to-right segment with y pointing down."""
    analytics = _analytics(lines=[_line()])

    events = _walk(
        analytics,
        _track((0.5, 0.2), second=0),
        _track((0.5, 0.8), second=1),
    )

    assert _kinds(events) == [EventKind.LINE_CROSS]
    assert events[0].line_id == DOOR
    assert events[0].direction == 1
    assert events[0].camera_id == CAMERA
    assert events[0].track_id == TrackId(1)


def test_crossing_the_other_way_emits_the_opposite_direction() -> None:
    analytics = _analytics(lines=[_line()])

    events = _walk(
        analytics,
        _track((0.5, 0.8), second=0),
        _track((0.5, 0.2), second=1),
    )

    assert [event.direction for event in events] == [-1]


def test_the_event_is_stamped_with_capture_time_not_wall_clock() -> None:
    """Every bucket downstream is keyed by capture time (engine-architecture.md §10)."""
    analytics = _analytics(lines=[_line()])

    events = _walk(
        analytics,
        _track((0.5, 0.2), second=0),
        _track((0.5, 0.8), second=7),
    )

    assert events[0].ts == FrameTs(T0 + timedelta(seconds=7))


def test_positive_dir_labels_the_sign_but_does_not_change_it() -> None:
    """`direction` is geometric; mapping `+1` to "in" or "out" is the metric layer's job."""
    inward = _analytics(lines=[_line(positive_dir="in")])
    outward = _analytics(lines=[_line(positive_dir="out")])
    steps = (_track((0.5, 0.2), second=0), _track((0.5, 0.8), second=1))

    assert [event.direction for event in _walk(inward, *steps)] == [1]
    assert [event.direction for event in _walk(outward, *steps)] == [1]


def test_a_staff_track_carries_its_tag_onto_the_event() -> None:
    """The adjacency metric (A2) reduces on this flag; losing it here loses the metric."""
    analytics = _analytics(lines=[_line()])

    events = _walk(
        analytics,
        _track((0.5, 0.2), second=0, is_staff=True),
        _track((0.5, 0.8), second=1, is_staff=True),
    )

    assert events[0].is_staff is True


# --- Line crossing: what must NOT be counted --------------------------------


def test_a_single_sample_cannot_cross_anything() -> None:
    """A crossing needs two foot-points; a track born on the far side is not a crossing."""
    analytics = _analytics(lines=[_line()])

    assert _walk(analytics, _track((0.5, 0.8), second=0)) == []


def test_walking_past_the_end_of_the_line_is_not_a_crossing() -> None:
    """The segment test, not the infinite line: someone passing outside the doorway."""
    analytics = _analytics(lines=[_line()])

    events = _walk(
        analytics,
        _track((0.95, 0.2), second=0),
        _track((0.95, 0.8), second=1),
    )

    assert events == []


def test_a_foot_point_exactly_on_the_line_emits_nothing_and_holds_the_side() -> None:
    """A graze must not become two half-crossings, nor a crossing in the wrong direction.

    The sticky side ignores zero-sided samples entirely, so stepping onto the line and
    back off the same side is silent — and stepping across it afterwards still reads as
    one crossing in the right direction.
    """
    analytics = _analytics(lines=[_line()])

    events = _walk(
        analytics,
        _track((0.5, 0.2), second=0),
        _track((0.5, 0.5), second=1),
        _track((0.5, 0.2), second=2),
    )

    assert events == []


def test_jitter_across_the_line_counts_once_not_once_per_flip() -> None:
    """The hysteresis band is the primary defence: micro-jitter never re-triggers.

    Someone loitering on the threshold flips sides every frame. Without the band this
    emits a burst of alternating crossings and the day's footfall is detection noise.
    """
    analytics = _analytics(lines=[_line()])

    events = _walk(
        analytics,
        _track((0.5, 0.40), second=0),
        _track((0.5, 0.60), second=1),
        _track((0.5, 0.499), second=2),
        _track((0.5, 0.501), second=3),
        _track((0.5, 0.499), second=4),
        _track((0.5, 0.502), second=5),
    )

    assert _kinds(events) == [EventKind.LINE_CROSS]
    assert events[0].direction == 1


def test_a_real_re_crossing_counts_once_the_foot_point_clears_the_band() -> None:
    """Hysteresis suppresses jitter, not people. Walking back out must still count."""
    analytics = _analytics(lines=[_line()], crossing_debounce_s=0.0)

    events = _walk(
        analytics,
        _track((0.5, 0.40), second=0),
        _track((0.5, 0.60), second=1),
        _track((0.5, 0.40), second=2),
    )

    assert [event.direction for event in events] == [1, -1]


def test_the_debounce_timer_suppresses_an_immediate_re_cross() -> None:
    """The backstop for the case where the walker clears the band but does it instantly."""
    analytics = _analytics(lines=[_line()], crossing_debounce_s=5.0)

    events = _walk(
        analytics,
        _track((0.5, 0.40), second=0),
        _track((0.5, 0.60), second=1),
        _track((0.5, 0.40), second=2),
    )

    assert _kinds(events) == [EventKind.LINE_CROSS]


def test_the_debounce_expires() -> None:
    analytics = _analytics(lines=[_line()], crossing_debounce_s=5.0)

    events = _walk(
        analytics,
        _track((0.5, 0.40), second=0),
        _track((0.5, 0.60), second=1),
        _track((0.5, 0.40), second=30),
    )

    assert [event.direction for event in events] == [1, -1]


def test_two_tracks_debounce_independently() -> None:
    """The debounce is per `(track, line)`; one person's crossing cannot mute another's."""
    analytics = _analytics(lines=[_line()], crossing_debounce_s=5.0)

    analytics.on_tracks(CAMERA, [_track((0.5, 0.2), track_id=1), _track((0.6, 0.2), track_id=2)])
    events = analytics.on_tracks(
        CAMERA,
        [_track((0.5, 0.8), second=1, track_id=1), _track((0.6, 0.8), second=1, track_id=2)],
    )

    assert sorted(event.track_id for event in events) == [TrackId(1), TrackId(2)]


# --- The coasted-track policy (algorithms.md §3.4) ---------------------------


def test_a_segment_dead_reckoned_at_both_ends_emits_nothing() -> None:
    """Every count has to trace back to a real observation, not to the Kalman filter."""
    analytics = _analytics(lines=[_line()])

    events = _walk(
        analytics,
        _track((0.5, 0.2), second=0),
        _track((0.5, 0.3), second=1, coasted=True),
        _track((0.5, 0.8), second=2, coasted=True),
    )

    assert events == []


def test_a_crossing_hidden_by_an_occlusion_is_recovered_when_the_track_re_matches() -> None:
    """§3.4(4): re-evaluate against the re-found observation, not the coasted path.

    The person crosses while occluded, so both endpoints of the crossing segment are
    dead-reckoned and nothing may be emitted then. When the detector picks them up again
    on the far side, the crossing is emitted once, timestamped at the re-match — the
    alternative is losing the doorway event that matters most.
    """
    analytics = _analytics(lines=[_line()])

    events = _walk(
        analytics,
        _track((0.5, 0.2), second=0),
        _track((0.5, 0.45), second=1, coasted=True),
        _track((0.5, 0.55), second=2, coasted=True),
        _track((0.5, 0.8), second=3),
    )

    assert _kinds(events) == [EventKind.LINE_CROSS]
    assert events[0].direction == 1
    assert events[0].ts == FrameTs(T0 + timedelta(seconds=3))


# --- Zones ------------------------------------------------------------------


def test_entering_a_zone_emits_one_enter_event() -> None:
    analytics = _analytics(zones=[_zone()])

    events = _walk(
        analytics,
        _track((0.05, 0.05), second=0),
        _track((0.5, 0.5), second=1),
    )

    assert _kinds(events) == [EventKind.ZONE_ENTER]
    assert events[0].zone_id == FLOOR


def test_staying_inside_a_zone_emits_nothing_further() -> None:
    """Enter and exit are edges, not levels — the reducer counts membership changes."""
    analytics = _analytics(zones=[_zone()])

    events = _walk(
        analytics,
        _track((0.5, 0.5), second=0),
        _track((0.5, 0.51), second=1),
        _track((0.5, 0.52), second=2),
    )

    assert _kinds(events) == [EventKind.ZONE_ENTER]


def test_leaving_a_zone_emits_an_exit() -> None:
    analytics = _analytics(zones=[_zone()])

    events = _walk(
        analytics,
        _track((0.5, 0.5), second=0),
        _track((0.05, 0.05), second=1),
    )

    assert _kinds(events) == [EventKind.ZONE_ENTER, EventKind.ZONE_EXIT]


def test_a_track_that_dies_inside_a_zone_still_produces_an_exit() -> None:
    """A contract, not an accident: the dwell state machine leaks without it.

    algorithms.md §7 relies on the membership diff being the single owner of
    track-death-to-exit. Without it every mannequin holds an open dwell forever.
    """
    analytics = _analytics(zones=[_zone()])
    analytics.on_tracks(CAMERA, [_track((0.5, 0.5), second=0)])

    events = analytics.on_tracks(CAMERA, [])

    assert _kinds(events) == [EventKind.ZONE_EXIT]
    assert events[0].zone_id == FLOOR
    assert events[0].track_id == TrackId(1)


def test_a_coasted_track_keeps_its_zone_membership() -> None:
    """A dead-reckoned position may not create or destroy membership (§3.4(2))."""
    analytics = _analytics(zones=[_zone()])

    events = _walk(
        analytics,
        _track((0.5, 0.5), second=0),
        _track((0.05, 0.05), second=1, coasted=True),
    )

    assert _kinds(events) == [EventKind.ZONE_ENTER]


def test_overlapping_zones_each_get_their_own_event() -> None:
    """A queue drawn inside a shop floor is the ordinary case."""
    analytics = _analytics(
        zones=[
            _zone("floor", polygon=[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
            _zone("queue", polygon=SQUARE),
        ]
    )

    events = _walk(
        analytics,
        _track((0.9, 0.9), second=0),
        _track((0.5, 0.5), second=1),
    )

    assert [event.zone_id for event in events] == [ZoneId("floor"), ZoneId("queue")]


# --- Scoping ----------------------------------------------------------------


def test_a_camera_with_no_geometry_produces_no_events() -> None:
    analytics = _analytics(lines=[_line()], zones=[_zone()])

    events = analytics.on_tracks(OTHER_CAMERA, [_track((0.5, 0.5), camera=OTHER_CAMERA)])

    assert events == []


def test_another_cameras_geometry_is_never_consulted() -> None:
    """Normalized coordinates mean the same point exists on every camera at once."""
    analytics = _analytics(zones=[_zone("till-zone", camera=OTHER_CAMERA)])

    events = _walk(
        analytics,
        _track((0.05, 0.05), second=0),
        _track((0.5, 0.5), second=1),
    )

    assert events == []


def test_tracks_from_the_wrong_camera_are_refused() -> None:
    """Track ids are unique per camera per run, so mixing cameras corrupts the state map."""
    analytics = _analytics(zones=[_zone()])

    with pytest.raises(ValueError, match="camera"):
        analytics.on_tracks(CAMERA, [_track((0.5, 0.5), camera=OTHER_CAMERA)])
