"""Swapping geometry under a running camera without losing what was open.

`GeometryAnalytics` holds two pieces of state that outlive a frame: the sticky side per
`(track, line)`, which is what makes a crossing a crossing, and zone residency, which is
algorithms.md §7's open dwells. Both live inside the worker process.

The whole point of this module is that **rebuilding the analytics on a geometry edit
discards that state**, and discarding it is not neutral. §7 makes zone-membership diffing
the single owner of track-death → exit; a dwell that is dropped rather than closed never
emits its exit, so `open_dwells` leaks and the visit is never counted. So a reconfigure
closes what is open *first*, then swaps — a geometry edit looks like everyone left and
re-entered, which is honest and bounded.

Red-first for P3.8.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from mesopic.analytics.geometry import GeometryAnalytics
from mesopic.analytics.site_geometry import SiteGeometry
from mesopic.config.schema import MesopicConfig
from mesopic.types import (
    CameraId,
    EventKind,
    FrameTs,
    NormPoint,
    Track,
    TrackId,
    ZoneId,
)

CAMERA = CameraId("front-door")
OTHER_CAMERA = CameraId("till")
T0 = datetime(2026, 8, 16, 12, 0, tzinfo=UTC)

FLOOR = ZoneId("floor")
SQUARE = [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]]
NARROWER = [[0.3, 0.3], [0.7, 0.3], [0.7, 0.7], [0.3, 0.7]]

INSIDE: NormPoint = (0.5, 0.5)


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
                },
                {
                    "camera_id": OTHER_CAMERA,
                    "name": "Till",
                    "source": {"kind": "frigate", "mqtt_topic": "frigate/till"},
                    "reference_resolution": [1920, 1080],
                },
            ],
            "lines": [],
            "frigate": {"broker": "mosquitto"},
            "zones": zones,
        }
    )


def _zone(zone_id: str = FLOOR, camera: str = CAMERA, polygon: Any = None) -> dict[str, Any]:
    return {
        "zone_id": zone_id,
        "camera_id": camera,
        "polygon": SQUARE if polygon is None else polygon,
        "metrics": ["occupancy", "dwell_seconds"],
    }


def _track(point: NormPoint, *, second: float = 0.0, track_id: int = 1) -> Track:
    return Track(
        camera_id=CAMERA,
        track_id=TrackId(track_id),
        ts=FrameTs(T0 + timedelta(seconds=second)),
        foot_point=point,
        score=0.9,
        time_since_update=0,
    )


def _occupied() -> GeometryAnalytics:
    """One track standing inside `floor`, so there is an open dwell to lose."""
    analytics = GeometryAnalytics(SiteGeometry.compile(_config([_zone()])))
    analytics.on_tracks(CAMERA, [_track(INSIDE)])
    return analytics


def _kinds(events: list[Any]) -> list[EventKind]:
    return [event.kind for event in events]


# --- Closing what is open ---------------------------------------------------


def test_reconfigure_closes_an_open_dwell_rather_than_dropping_it() -> None:
    """The failure this prevents: the visit is never counted and `open_dwells` leaks.

    §7 makes zone-membership diffing the single owner of track-death → exit. A geometry
    swap removes the zone from the diff entirely, so the exit can only come from here.
    """
    analytics = _occupied()

    events = analytics.reconfigure(
        SiteGeometry.compile(_config([_zone(polygon=NARROWER)])),
        camera_id=CAMERA,
        ts=FrameTs(T0 + timedelta(seconds=5)),
    )

    assert EventKind.ZONE_EXIT in _kinds(events)


def test_the_closing_exit_names_the_zone_that_was_left() -> None:
    analytics = _occupied()

    events = analytics.reconfigure(
        SiteGeometry.compile(_config([_zone(polygon=NARROWER)])),
        camera_id=CAMERA,
        ts=FrameTs(T0 + timedelta(seconds=5)),
    )

    exits = [event for event in events if event.kind is EventKind.ZONE_EXIT]
    assert [event.zone_id for event in exits] == [FLOOR]


def test_a_reconfigure_with_nobody_inside_emits_nothing() -> None:
    """A geometry edit on an empty shop is not an event. Emitting one would put a phantom
    visit in the store every time the operator nudged a polygon."""
    analytics = GeometryAnalytics(SiteGeometry.compile(_config([_zone()])))

    events = analytics.reconfigure(
        SiteGeometry.compile(_config([_zone(polygon=NARROWER)])),
        camera_id=CAMERA,
        ts=FrameTs(T0),
    )

    assert events == []


def test_the_same_track_is_not_still_resident_after_the_swap() -> None:
    """Closed means closed: a re-entry after the swap has to start a new residency, or
    the dwell that was just closed is silently continued and counted twice."""
    analytics = _occupied()
    analytics.reconfigure(
        SiteGeometry.compile(_config([_zone(polygon=NARROWER)])),
        camera_id=CAMERA,
        ts=FrameTs(T0 + timedelta(seconds=5)),
    )

    events = analytics.on_tracks(CAMERA, [_track(INSIDE, second=6)])

    assert EventKind.ZONE_ENTER in _kinds(events)


def test_reconfigure_leaves_another_camera_alone() -> None:
    """A worker owns one camera. Closing another camera's dwells from here would emit
    exits nobody observed, in a process that cannot see that camera at all."""
    analytics = GeometryAnalytics(
        SiteGeometry.compile(_config([_zone(), _zone(zone_id="till-zone", camera=OTHER_CAMERA)]))
    )
    analytics.on_tracks(CAMERA, [_track(INSIDE)])

    events = analytics.reconfigure(
        SiteGeometry.compile(_config([_zone(), _zone(zone_id="till-zone", camera=OTHER_CAMERA)])),
        camera_id=OTHER_CAMERA,
        ts=FrameTs(T0 + timedelta(seconds=5)),
    )

    assert events == []


def test_the_new_geometry_is_in_force_after_a_reconfigure() -> None:
    """The swap has to actually happen — a `reconfigure` that only closed dwells would
    pass every test above and change nothing."""
    analytics = GeometryAnalytics(SiteGeometry.compile(_config([_zone()])))

    analytics.reconfigure(
        SiteGeometry.compile(_config([_zone(polygon=[[0.0, 0.0], [0.1, 0.0], [0.1, 0.1]])])),
        camera_id=CAMERA,
        ts=FrameTs(T0),
    )
    events = analytics.on_tracks(CAMERA, [_track(INSIDE, second=1)])

    assert EventKind.ZONE_ENTER not in _kinds(events)


def test_a_crossing_is_not_carried_across_a_reconfigure() -> None:
    """Sticky side is per `(track, line)` and describes geometry that may no longer
    exist. Carrying it makes the first frame after a swap compare against a line that
    moved, which is a crossing nobody walked."""
    analytics = _occupied()

    analytics.reconfigure(
        SiteGeometry.compile(_config([_zone(polygon=NARROWER)])),
        camera_id=CAMERA,
        ts=FrameTs(T0 + timedelta(seconds=5)),
    )

    assert analytics.open_zone_count(CAMERA) == 0
