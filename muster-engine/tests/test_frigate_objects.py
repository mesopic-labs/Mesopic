"""Frigate's object stream, adapted into the tick shape the rest of the engine speaks.

Frigate publishes **deltas** — one message per object as it appears, moves and ends.
`GeometryAnalytics.on_tracks` takes **states**: every track a camera can currently see,
diffed against the previous call. That mismatch is the whole of this module, and getting
it wrong does not raise — feed geometry one object at a time and every *other* person in
the shop appears to leave and re-enter on every message, shredding dwells and oscillating
occupancy while producing data that looks entirely plausible.

Three more failures that stay quiet:

* **A missed `end` haunts the zone forever.** `_zone_events` is the sole owner of
  track-death → exit, so an object Frigate stops mentioning never leaves, and
  `open_dwells` grows without bound. Objects expire on a TTL.
* **Nothing arrives when nothing moves.** For RTSP, frames keep coming, so occupancy
  samples keep flowing and a quiet minute is *measured* as quiet. Frigate says nothing at
  all, so the tick has to keep happening on its own or a quiet shop becomes an unmeasured
  one — which is the difference between a zero and a gap.
* **A bucket that goes backwards is dropped.** Capture times come from Frigate's clock,
  and the supervisor forgets closed buckets, so a tick timestamp that ever moves backwards
  silently discards a minute.

Everything here runs against hand-built payloads. **The wire format is taken from
Frigate's documentation and is not verified against a running Frigate** — see ADR-0022.

Red-first for P4.3.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from muster.ingest.frigate import DEFAULT_TTL_S, FrigateObjects
from muster.types import CameraId, TrackId

CAMERA = CameraId("till")
FRIGATE_CAMERA = "till_cam"

WIDTH = 1280
HEIGHT = 720

# Frigate's clock, in unix seconds. Deliberately not "now": a capture time is Frigate's,
# and nothing here may quietly substitute ours.
T0 = 1_755_500_000.0


def _payload(
    object_id: str = "1755500000.1-abc",
    *,
    kind: str = "new",
    box: tuple[int, int, int, int] = (600, 300, 700, 500),
    frame_time: float = T0,
    label: str = "person",
    camera: str = FRIGATE_CAMERA,
    score: float = 0.86,
    **extra: Any,
) -> bytes:
    after: dict[str, Any] = {
        "id": object_id,
        "camera": camera,
        "frame_time": frame_time,
        "label": label,
        "score": score,
        "box": list(box),
        # Fields Frigate really sends that we have no use for. They must not be rejected:
        # forbidding unknown keys in somebody else's wire format means breaking on their
        # next release, which is the opposite of the rule our own config follows.
        "area": 20000,
        "ratio": 0.5,
        "region": [500, 200, 800, 600],
        "current_zones": [],
        "has_snapshot": True,
        **extra,
    }
    return json.dumps({"type": kind, "before": after, "after": after}).encode()


def _objects(**kwargs: Any) -> FrigateObjects:
    return FrigateObjects(
        CAMERA,
        frigate_camera=FRIGATE_CAMERA,
        width=WIDTH,
        height=HEIGHT,
        **kwargs,
    )


# --- Deltas become states ---------------------------------------------------


def test_one_object_becomes_one_track() -> None:
    objects = _objects()
    assert objects.apply(_payload(), now=0.0) is True

    tracks = objects.tick(now=0.0)

    assert len(tracks) == 1
    assert tracks[0].camera_id == CAMERA
    assert tracks[0].score == pytest.approx(0.86)


def test_every_live_object_rides_every_tick() -> None:
    """The load-bearing one. Frigate names one object per message; geometry needs all of
    them every time, or the ones it is not told about read as having left."""
    objects = _objects()
    objects.apply(_payload("a", box=(100, 100, 200, 300)), now=0.0)
    objects.apply(_payload("b", box=(600, 100, 700, 300)), now=0.0)

    # A message about `a` alone must still produce a tick naming both.
    objects.apply(_payload("a", kind="update", box=(120, 100, 220, 300)), now=1.0)
    tracks = objects.tick(now=1.0)

    assert len(tracks) == 2


def test_an_end_event_removes_the_object() -> None:
    objects = _objects()
    objects.apply(_payload("a"), now=0.0)
    objects.apply(_payload("a", kind="end"), now=1.0)

    assert objects.tick(now=1.0) == []


def test_an_object_frigate_stops_mentioning_expires() -> None:
    """A dropped `end` would otherwise leave a resident who never leaves: geometry's
    membership diff is the only thing that closes a dwell."""
    objects = _objects()
    objects.apply(_payload("a"), now=0.0)

    assert objects.tick(now=DEFAULT_TTL_S - 0.1) != []
    assert objects.tick(now=DEFAULT_TTL_S + 0.1) == []


def test_a_tick_happens_even_when_nothing_arrives() -> None:
    """A quiet shop must read as measured-and-empty, not as unmeasured. Tick is callable
    with no messages at all and answers honestly."""
    assert _objects().tick(now=0.0) == []


# --- Identity ---------------------------------------------------------------


def test_an_object_keeps_one_track_id_across_updates() -> None:
    objects = _objects()
    objects.apply(_payload("a"), now=0.0)
    first = objects.tick(now=0.0)[0].track_id

    objects.apply(_payload("a", kind="update", box=(620, 300, 720, 500)), now=1.0)

    assert objects.tick(now=1.0)[0].track_id == first


def test_two_objects_never_share_a_track_id() -> None:
    objects = _objects()
    objects.apply(_payload("a"), now=0.0)
    objects.apply(_payload("b", box=(100, 100, 200, 300)), now=0.0)

    ids = {track.track_id for track in objects.tick(now=0.0)}

    assert len(ids) == 2


def test_a_track_id_is_not_reused_while_its_object_lives() -> None:
    """Frigate's ids are strings and ours are ints, so the mapping is ours to keep sane.
    Handing a live object's number to a new one merges two people into one track."""
    objects = _objects()
    objects.apply(_payload("a"), now=0.0)
    alive = objects.tick(now=0.0)[0].track_id
    objects.apply(_payload("b", box=(100, 100, 200, 300)), now=1.0)

    assert alive not in {t.track_id for t in objects.tick(now=1.0) if t.track_id != alive}
    assert objects.tick(now=1.0)[1].track_id != alive


def test_track_ids_start_from_a_stable_base() -> None:
    objects = _objects()
    objects.apply(_payload("a"), now=0.0)
    assert objects.tick(now=0.0)[0].track_id == TrackId(1)


# --- Geometry ---------------------------------------------------------------


def test_the_foot_point_is_the_bottom_centre_normalized() -> None:
    """Frigate boxes are detect-resolution pixels; everything downstream is `[0,1]`, and
    the foot-point convention is bottom-centre — the same one the tracker uses."""
    objects = _objects()
    objects.apply(_payload(box=(640, 0, 1280, 360)), now=0.0)

    (track,) = objects.tick(now=0.0)

    assert track.foot_point == pytest.approx((0.75, 0.5))


def test_a_box_beyond_the_frame_is_clamped_not_rejected() -> None:
    """Frigate's detect resolution and our `reference_resolution` are two config values a
    user can set inconsistently. A foot-point outside `[0,1]` would break every polygon
    test downstream, so it is clamped here and the geometry still runs."""
    objects = _objects()
    objects.apply(_payload(box=(1200, 600, 1400, 900)), now=0.0)

    (track,) = objects.tick(now=0.0)

    assert 0.0 <= track.foot_point[0] <= 1.0
    assert 0.0 <= track.foot_point[1] <= 1.0


def test_a_track_carries_frigates_capture_time_not_ours() -> None:
    """Bucketing is by capture time. Substituting our clock would put an event in the
    minute we processed it rather than the minute it happened in."""
    objects = _objects()
    objects.apply(_payload(frame_time=T0), now=999.0)

    (track,) = objects.tick(now=999.0)

    assert track.ts.timestamp() == pytest.approx(T0)


def test_a_track_is_never_coasted() -> None:
    """Every object Frigate reports is one it saw. There is no dead reckoning here, so
    geometry may use these positions for crossings and zone membership without reserve."""
    objects = _objects()
    objects.apply(_payload(), now=0.0)
    assert objects.tick(now=0.0)[0].time_since_update == 0


# --- The tick clock ---------------------------------------------------------


def test_the_tick_time_follows_frigates_newest_frame() -> None:
    objects = _objects()
    objects.apply(_payload("a", frame_time=T0), now=0.0)
    objects.apply(_payload("b", frame_time=T0 + 5, box=(100, 100, 200, 300)), now=0.0)

    assert objects.tick_ts(now=0.0).timestamp() == pytest.approx(T0 + 5)


def test_the_tick_time_never_runs_backwards() -> None:
    """The supervisor forgets closed buckets, so a timestamp that regresses does not
    produce an out-of-order metric — it produces a silently discarded minute."""
    objects = _objects()
    objects.apply(_payload("a", frame_time=T0 + 10), now=0.0)
    ahead = objects.tick_ts(now=0.0)

    objects.apply(_payload("b", frame_time=T0, box=(100, 100, 200, 300)), now=1.0)

    assert objects.tick_ts(now=1.0) >= ahead


# --- Untrusted input --------------------------------------------------------


def test_a_message_for_another_camera_is_ignored() -> None:
    """`frigate/events` carries every camera on the box, so the payload's own name is
    what says whose it is — not the topic it arrived on."""
    objects = _objects()
    assert objects.apply(_payload(camera="driveway"), now=0.0) is False
    assert objects.tick(now=0.0) == []


def test_only_people_are_counted() -> None:
    objects = _objects()
    assert objects.apply(_payload(label="car"), now=0.0) is False
    assert objects.tick(now=0.0) == []


def test_a_false_positive_is_ignored() -> None:
    objects = _objects()
    assert objects.apply(_payload(false_positive=True), now=0.0) is False


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"not json at all",
        b"[]",
        b'{"type": "new"}',
        b'{"type": "new", "after": {"id": "a"}}',
        b'{"type": "sideways", "after": {"id": "a", "camera": "till_cam", '
        b'"frame_time": 1.0, "label": "person", "box": [1, 2, 3, 4]}}',
        b'{"type": "new", "after": {"id": "a", "camera": "till_cam", '
        b'"frame_time": 1.0, "label": "person", "box": [1, 2]}}',
    ],
)
def test_a_malformed_message_is_dropped_and_nothing_else(raw: bytes) -> None:
    """MQTT is a trust boundary. A broker publishing junk — or a Frigate version we do
    not understand — must cost that message and not the camera."""
    objects = _objects()
    objects.apply(_payload("good"), now=0.0)

    assert objects.apply(raw, now=1.0) is False

    assert len(objects.tick(now=1.0)) == 1


def test_an_oversized_message_is_refused_before_it_is_parsed() -> None:
    """An explicit ceiling on an unbounded input, per the house rule. A broker is not
    necessarily friendly, and `json.loads` on an arbitrary payload is a decompression
    bomb's worth of work away from being a problem."""
    objects = _objects()
    assert objects.apply(b"x" * (1 << 20), now=0.0) is False


def test_an_idle_camera_still_advances_its_clock() -> None:
    """The reason the tick time is extrapolated rather than repeated.

    Frigate publishes on change, so a quiet room sends nothing and the anchor stops
    moving. A tick that repeated the last capture time would present a zero-width
    interval, which every sampled-state consumer drops — so the camera would go from
    "measured, empty" to "not measured at all", which is the one distinction occupancy
    exists to preserve.
    """
    objects = _objects()
    objects.apply(_payload(frame_time=T0), now=100.0)

    first = objects.tick_ts(now=100.0)
    later = objects.tick_ts(now=130.0)

    assert (later - first).total_seconds() == pytest.approx(30.0)


def test_a_camera_that_never_spoke_still_has_a_clock() -> None:
    """Before Frigate has said anything there is no anchor to extrapolate from, so the
    tick falls back to our own wall clock — the only case where it is used."""
    stamps = iter(
        [datetime(2026, 8, 18, 9, 30, tzinfo=UTC), datetime(2026, 8, 18, 9, 31, tzinfo=UTC)]
    )
    objects = _objects(clock=lambda: next(stamps))

    assert objects.tick_ts(now=0.0) < objects.tick_ts(now=1.0)


def test_an_out_of_order_message_does_not_rewind_the_clock() -> None:
    """Frigate can deliver late. Re-anchoring on an older capture time would drag the
    tick clock back with it, into buckets the supervisor has already closed."""
    objects = _objects()
    objects.apply(_payload("a", frame_time=T0 + 10), now=0.0)
    ahead = objects.tick_ts(now=0.0)

    objects.apply(_payload("b", frame_time=T0, box=(100, 100, 200, 300)), now=0.0)

    assert objects.tick_ts(now=0.0) >= ahead
