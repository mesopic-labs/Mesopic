"""Sampled state reaches the reducer as its own events.

Occupancy and queue length are *sampled states*, not transitions. A reducer that sees
only `zone_enter`/`zone_exit` cannot compute either one: three people standing still
through a minute emit no events at all, so that minute would reduce to nothing rather
than to three. And algorithms.md §0.6 wants the mean weighted by the wall-clock gap
between sampled frames, which a transition carries no way to express.

So geometry emits one `occupancy_sample` per zone per tick, carrying the count inside and
the `Δt` it represents, plus a sparse `zone_confirmed` when a track's residency passes
`dwell_min_s` — the confirmation §7's zone-derived footfall counts and the sample's own
count cannot identify.

Red-first for P2.4 (ADR-0016).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from muster.analytics.geometry import GeometryAnalytics
from muster.analytics.site_geometry import SiteGeometry
from muster.config.schema import MusterConfig
from muster.types import (
    CameraId,
    EventKind,
    FrameTs,
    NormPoint,
    RawEvent,
    Track,
    TrackId,
    ZoneId,
)

CAMERA = CameraId("front-door")
OTHER_CAMERA = CameraId("till")
T0 = datetime(2026, 8, 16, 12, 0, tzinfo=UTC)

FLOOR = ZoneId("floor")
AISLE = ZoneId("aisle")
TILL_ZONE = ZoneId("till-area")

INSIDE: NormPoint = (0.5, 0.5)
OUTSIDE: NormPoint = (0.05, 0.05)

SQUARE = [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]]

DWELL_MIN_S = 3.0


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
                },
                {
                    "camera_id": OTHER_CAMERA,
                    "name": "Till",
                    "source": {"kind": "frigate", "mqtt_topic": "frigate/till"},
                    "reference_resolution": [1920, 1080],
                },
            ],
            "zones": zones,
        }
    )


def _zone(zone_id: str = FLOOR, camera: str = CAMERA) -> dict[str, Any]:
    return {"zone_id": zone_id, "camera_id": camera, "polygon": SQUARE}


def _analytics(*zones: dict[str, Any], dwell_min_s: float = DWELL_MIN_S) -> GeometryAnalytics:
    zones = zones or (_zone(),)
    return GeometryAnalytics(SiteGeometry.compile(_config(list(zones))), dwell_min_s=dwell_min_s)


def _track(
    point: NormPoint,
    *,
    second: float,
    track_id: int = 1,
    camera: CameraId = CAMERA,
) -> Track:
    return Track(
        camera_id=camera,
        track_id=TrackId(track_id),
        ts=FrameTs(T0 + timedelta(seconds=second)),
        foot_point=point,
        score=0.9,
    )


def _samples(events: list[RawEvent]) -> list[RawEvent]:
    return [event for event in events if event.kind is EventKind.OCCUPANCY_SAMPLE]


def _confirmations(events: list[RawEvent]) -> list[RawEvent]:
    return [event for event in events if event.kind is EventKind.ZONE_CONFIRMED]


# --- The sample itself ------------------------------------------------------


def test_a_tick_emits_one_occupancy_sample_per_zone_on_the_camera() -> None:
    analytics = _analytics(_zone(FLOOR), _zone(AISLE), _zone(TILL_ZONE, camera=OTHER_CAMERA))

    analytics.on_tracks(CAMERA, [_track(INSIDE, second=0.0)])
    events = analytics.on_tracks(CAMERA, [_track(INSIDE, second=0.5)])

    assert {sample.zone_id for sample in _samples(events)} == {FLOOR, AISLE}


def test_the_first_tick_emits_no_sample_because_it_spans_no_interval() -> None:
    """A `Δt`-weighted accumulator has nothing to weight on the very first observation."""
    analytics = _analytics()

    events = analytics.on_tracks(CAMERA, [_track(INSIDE, second=0.0)])

    assert _samples(events) == []


def test_a_sample_carries_the_wall_clock_gap_since_the_previous_tick() -> None:
    analytics = _analytics()

    analytics.on_tracks(CAMERA, [_track(INSIDE, second=0.0)])
    events = analytics.on_tracks(CAMERA, [_track(INSIDE, second=0.4)])

    assert [sample.dt_s for sample in _samples(events)] == [0.4]


def test_a_sample_counts_everyone_inside_including_the_unconfirmed() -> None:
    """`occupancy_raw` is every track in the zone — the confirmation lag is what makes it
    the honest input for a peak (algorithms.md §6.1)."""
    analytics = _analytics()

    analytics.on_tracks(
        CAMERA, [_track(INSIDE, second=0.0, track_id=1), _track(INSIDE, second=0.0, track_id=2)]
    )
    events = analytics.on_tracks(
        CAMERA, [_track(INSIDE, second=0.5, track_id=1), _track(INSIDE, second=0.5, track_id=2)]
    )

    assert [sample.value for sample in _samples(events)] == [2.0]


def test_a_sample_also_carries_how_many_of_those_are_confirmed() -> None:
    """Two counts, one sample, because §6.1 needs both and they must agree.

    Peak reads the raw count and the mean reads the confirmed one. Deriving the confirmed
    count in the reducer is not possible for the same reason the raw one is not: the
    events of one minute do not say who was already inside when it began.
    """
    analytics = _analytics()

    # Second 0 and 1: inside, not yet past dwell_min_s. Second 4: confirmed.
    events = []
    for second in (0.0, 1.0, 4.0):
        events += analytics.on_tracks(CAMERA, [_track(INSIDE, second=second)])

    assert [(s.value, s.confirmed_value) for s in _samples(events)] == [(1.0, 0.0), (1.0, 1.0)]


def test_a_sample_belongs_to_no_track() -> None:
    """A zone-level count is about the zone, not about anyone in it."""
    analytics = _analytics()

    analytics.on_tracks(CAMERA, [_track(INSIDE, second=0.0)])
    events = analytics.on_tracks(CAMERA, [_track(INSIDE, second=0.5)])

    assert [sample.track_id for sample in _samples(events)] == [None]


def test_an_empty_zone_still_reports_a_sample() -> None:
    """The quiet minute is the whole reason samples exist.

    Nobody is in the zone, so there are no transitions and no tracks to take a timestamp
    from — but "zero people were here" is a fact the minute must record, or a deserted
    hour is indistinguishable from a dead camera.
    """
    analytics = _analytics()

    analytics.on_tracks(CAMERA, [], ts=FrameTs(T0))
    events = analytics.on_tracks(CAMERA, [], ts=FrameTs(T0 + timedelta(seconds=0.5)))

    assert [(sample.value, sample.dt_s) for sample in _samples(events)] == [(0.0, 0.5)]


def test_a_tick_that_does_not_advance_the_clock_emits_no_sample() -> None:
    """A zero-width interval would inflate `sample_count` while weighting nothing."""
    analytics = _analytics()

    analytics.on_tracks(CAMERA, [_track(INSIDE, second=0.0)])
    events = analytics.on_tracks(CAMERA, [_track(INSIDE, second=0.0)])

    assert _samples(events) == []


def test_only_the_named_camera_is_sampled() -> None:
    analytics = _analytics(_zone(FLOOR), _zone(TILL_ZONE, camera=OTHER_CAMERA))

    analytics.on_tracks(CAMERA, [_track(INSIDE, second=0.0)])
    events = analytics.on_tracks(CAMERA, [_track(INSIDE, second=0.5)])

    assert [sample.zone_id for sample in _samples(events)] == [FLOOR]


# --- Confirmation -----------------------------------------------------------


def test_a_track_is_not_confirmed_before_the_minimum_dwell() -> None:
    analytics = _analytics()

    events = []
    for second in (0.0, 1.0, 2.0):
        events += analytics.on_tracks(CAMERA, [_track(INSIDE, second=second)])

    assert _confirmations(events) == []


def test_a_track_inside_for_the_minimum_dwell_is_confirmed() -> None:
    analytics = _analytics()

    events = []
    for second in (0.0, 1.5, 3.0):
        events += analytics.on_tracks(CAMERA, [_track(INSIDE, second=second)])

    confirmed = _confirmations(events)
    assert [(event.zone_id, event.track_id) for event in confirmed] == [(FLOOR, TrackId(1))]


def test_a_track_is_confirmed_only_once() -> None:
    """Footfall counts confirmations, so a repeat is a person counted twice."""
    analytics = _analytics()

    events = []
    for second in (0.0, 3.0, 4.0, 5.0):
        events += analytics.on_tracks(CAMERA, [_track(INSIDE, second=second)])

    assert len(_confirmations(events)) == 1


def test_leaving_the_zone_restarts_the_confirmation_clock() -> None:
    """§6's rule is *continuously* inside, so a round trip starts the count again."""
    analytics = _analytics()

    events = []
    for point, second in ((INSIDE, 0.0), (INSIDE, 2.0), (OUTSIDE, 2.5), (INSIDE, 3.0)):
        events += analytics.on_tracks(CAMERA, [_track(point, second=second)])
    assert _confirmations(events) == []

    events += analytics.on_tracks(CAMERA, [_track(INSIDE, second=6.0)])
    assert len(_confirmations(events)) == 1


def test_a_confirmation_is_stamped_when_it_was_reached_not_when_the_track_arrived() -> None:
    analytics = _analytics()

    events = []
    for second in (0.0, 3.5):
        events += analytics.on_tracks(CAMERA, [_track(INSIDE, second=second)])

    assert [event.ts for event in _confirmations(events)] == [FrameTs(T0 + timedelta(seconds=3.5))]
