"""Frigate as an upstream detector, over MQTT (ADR-0006, ADR-0022).

This source decodes nothing. On a box already running Frigate we subscribe to its object
topic and adapt its detections to the same `Track` shape as every other path, skipping our
own decode+detect entirely — the largest free performance win available on the reference
box (engine-architecture.md §4, §7).

We talk to Frigate across an API boundary and neither bundle nor derive from its code:
that boundary is what keeps Frigate's AGPL away from the MIT engine (ADR-0008, ADR-0013).

The module is two units, split because only one of them needs a broker:

* **`FrigateObjects`** is the whole of the thinking and none of the plumbing. Given raw
  payloads and a clock it maintains the live object set and answers with ticks. No socket,
  no thread, no library — which is what lets every trap below be tested against hand-built
  messages.
* **`FrigateTrackSource`** is the plumbing: subscribe, buffer, hand payloads over, and
  yield the ticks that come back.

**Deltas in, states out.** Frigate publishes one message per object as it appears, moves
and ends. `GeometryAnalytics.on_tracks` takes every track a camera can currently see and
diffs zone membership against the previous call. Feeding it one object at a time is not a
degraded version of that — it makes every *other* person in the room appear to leave and
re-enter on every message, which shreds dwells and oscillates occupancy while producing
numbers that look entirely reasonable.

**The wire format is taken from Frigate's documentation and has not been verified against
a running Frigate.** Every assumption about it is isolated in `_read` and `_foot_point` so
correcting one is a line rather than an excavation. ADR-0022 records this.

Implements P4.3.
"""

from __future__ import annotations

import json
import queue
import time
from collections.abc import Callable, Iterator
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from paho.mqtt.client import Client
from paho.mqtt.enums import CallbackAPIVersion

from mesopic.types import CameraId, FrameTs, NormPoint, Track, TrackId

KEEPALIVE_S = 60

IDLE_TICK_S = 1.0
"""How long to wait for a message before ticking anyway. See `FrigateTrackSource.ticks`."""

INBOX_SIZE = 256
"""Payloads buffered from paho's thread before the oldest is dropped. A stall absorber,
not a store — the same rule and the same size as the worker's outbox."""

PERSON = "person"
"""The only label that means anything here. Mesopic counts people (engine §1)."""

DEFAULT_TTL_S = 60.0
"""How long an object survives without Frigate mentioning it again.

A missed `end` would otherwise leave a resident who never leaves: geometry's membership
diff is the single owner of track-death → exit, so nothing else would ever close that
dwell (algorithms.md §7).

Generous on purpose, and **this is the number most likely to need tuning against a real
Frigate**: Frigate publishes on change, so a person standing still may go unmentioned for
a long time, and expiring them early drops somebody who is plainly still there. Too long
leaks a phantom; too short blinks real people out. Sixty seconds is a starting guess, not
a measurement.
"""

MAX_PAYLOAD_BYTES = 64 * 1024
"""An explicit ceiling on an unbounded input, checked before anything parses it.

A broker is a trust boundary like any other: nothing downstream of it is trusted, and the
first thing an untrusted byte string gets is a length check rather than a JSON parser.
"""

_EVENT_KINDS = frozenset({"new", "update", "end"})
_BOX_VALUES = 4
_REQUIRED = ("id", "camera", "frame_time", "label")


@dataclass(frozen=True, slots=True)
class _Sighting:
    """One Frigate object as we last saw it."""

    track_id: TrackId
    foot_point: NormPoint
    score: float
    ts: FrameTs
    seen_at: float
    """Our own monotonic clock, for the TTL only. Never Frigate's — an expiry judged on
    somebody else's clock expires early or never when the two drift."""


class FrigateObjects:
    """Frigate's delta stream, held as the live object set and answered as ticks."""

    def __init__(
        self,
        camera_id: CameraId,
        *,
        frigate_camera: str,
        width: int,
        height: int,
        ttl_s: float = DEFAULT_TTL_S,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._camera_id = camera_id
        self._frigate_camera = frigate_camera
        self._width = width
        self._height = height
        self._ttl_s = ttl_s
        self._clock = clock
        self._live: dict[str, _Sighting] = {}
        self._ids: dict[str, TrackId] = {}
        self._next_id = 1
        self._anchor: tuple[datetime, float] | None = None
        """Frigate's newest capture time, and the monotonic reading when we received it.
        Together they let an idle tick extrapolate forward — see `tick_ts`."""
        self._last_tick: datetime | None = None

    def apply(self, payload: bytes, *, now: float) -> bool:
        """Take one raw message. Returns whether it was ours and understood.

        Never raises. A broker publishing junk, or a Frigate release we do not understand,
        costs that message and nothing else — a camera that dies of a bad payload is a
        camera that stops counting because somebody else deployed something.
        """
        event = _read(payload)
        if event is None or not self._is_ours(event):
            return False
        object_id = str(event["id"])
        if event["_kind"] == "end":
            self._forget(object_id)
            return True
        return self._remember(object_id, event, now=now)

    def tick(self, *, now: float) -> list[Track]:
        """Every object currently live, oldest first, as `Track`s.

        Callable with no messages at all, and that is not a formality: Frigate says
        nothing when nothing moves, so without a tick of its own a quiet shop would go
        unmeasured rather than measured-as-empty — the difference between a gap and a
        zero, which the whole occupancy design turns on (ADR-0016).
        """
        self._expire(now)
        return [
            Track(
                camera_id=self._camera_id,
                track_id=sighting.track_id,
                ts=sighting.ts,
                foot_point=sighting.foot_point,
                score=sighting.score,
                # Never coasted: every object here is one Frigate actually saw, so
                # geometry may use these positions without algorithms.md §3.4's reserve.
                time_since_update=0,
            )
            for sighting in sorted(self._live.values(), key=lambda s: s.track_id)
        ]

    def tick_ts(self, *, now: float) -> FrameTs:
        """The capture time this tick stands for.

        Anchored to Frigate's newest `frame_time`, because bucketing is by capture time
        and our own clock would file an event under the minute we *processed* it — then
        **extrapolated forward by the monotonic time elapsed since that anchor**.

        The extrapolation is what makes an idle camera measurable rather than merely
        silent. Frigate publishes on change, so a quiet room produces no messages and the
        anchor stops moving; a tick that simply repeated it would present a zero-width
        interval, and every consumer of a sampled state drops those (algorithms.md §6.2).
        The camera would go from "measured, empty" to "not measured", which is precisely
        the distinction occupancy is built to preserve.

        Wall-clock is used only before Frigate has ever spoken, when there is no anchor to
        extrapolate from.

        It **never runs backwards**: the supervisor forgets closed buckets, so a
        regressing timestamp does not produce an out-of-order metric — it produces a
        silently discarded minute.
        """
        if self._anchor is None:
            candidate = self._clock()
        else:
            captured_at, anchored_at = self._anchor
            candidate = captured_at + timedelta(seconds=max(now - anchored_at, 0.0))
        self._last_tick = candidate if self._last_tick is None else max(self._last_tick, candidate)
        return FrameTs(self._last_tick)

    # --- Internals ----------------------------------------------------------

    def _is_ours(self, event: dict[str, Any]) -> bool:
        """`frigate/events` carries every camera on the box, so the payload's own name is
        what says whose a message is — never the topic it arrived on."""
        return bool(
            event["camera"] == self._frigate_camera
            and event["label"] == PERSON
            and not event["false_positive"]
        )

    def _remember(self, object_id: str, event: dict[str, Any], *, now: float) -> bool:
        ts = datetime.fromtimestamp(float(event["frame_time"]), tz=UTC)
        if self._anchor is None or ts > self._anchor[0]:
            # Only a *newer* capture time re-anchors. Frigate can deliver out of order,
            # and moving the anchor back would rewind the tick clock with it.
            self._anchor = (ts, now)
        self._live[object_id] = _Sighting(
            track_id=self._track_id(object_id),
            foot_point=self._foot_point(event["box"]),
            score=float(event["score"]),
            ts=FrameTs(ts),
            seen_at=now,
        )
        return True

    def _forget(self, object_id: str) -> None:
        self._live.pop(object_id, None)
        self._ids.pop(object_id, None)

    def _expire(self, now: float) -> None:
        for object_id in [
            key for key, seen in self._live.items() if now - seen.seen_at > self._ttl_s
        ]:
            self._forget(object_id)

    def _track_id(self, object_id: str) -> TrackId:
        """Frigate's ids are strings and ours are ints, so the mapping is ours to keep.

        Monotonic and never reused, because handing a live object's number to a new one
        merges two people into one track — and a `TrackId` is already documented as unique
        per camera per run and nothing more.
        """
        if object_id not in self._ids:
            self._ids[object_id] = TrackId(self._next_id)
            self._next_id += 1
        return self._ids[object_id]

    def _foot_point(self, box: list[int]) -> NormPoint:
        """Bottom-centre of the box, normalized — the one geometric convention (§0.2).

        Frigate's boxes are in its *detect* resolution and ours are in `[0, 1]`, so the
        divisor is the camera's `reference_resolution`. Those are two config values a user
        can set inconsistently, so the result is **clamped rather than rejected**: a
        foot-point outside the frame would fail every polygon test downstream, and a
        camera that counts slightly wrong beats one that counts nothing.
        """
        x1, _, x2, y2 = box
        return (
            _clamp((x1 + x2) / 2 / self._width),
            _clamp(y2 / self._height),
        )


def _clamp(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def _read(payload: bytes) -> dict[str, Any] | None:
    """Parse one message into the fields we use, or `None` if it is not one.

    **Unknown keys are ignored, not refused** — the opposite of the rule `mesopic.yaml`
    follows, and deliberately so. Our own config forbids extras because an unrecognised
    key there is a user's mistake; this is somebody else's wire format, and forbidding
    what we do not recognise means breaking on their next release.
    """
    if not payload or len(payload) > MAX_PAYLOAD_BYTES:
        return None
    try:
        document = json.loads(payload)
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(document, dict):
        return None
    kind = document.get("type")
    after = document.get("after")
    if kind not in _EVENT_KINDS or not isinstance(after, dict) or not _is_complete(after):
        return None
    box = after["box"]
    return {
        "_kind": kind,
        "id": after["id"],
        "camera": after["camera"],
        "frame_time": after["frame_time"],
        "label": after["label"],
        "box": box,
        "score": after.get("score") if _is_number(after.get("score")) else 0.0,
        "false_positive": bool(after.get("false_positive", False)),
    }


def _is_complete(after: dict[str, Any]) -> bool:
    """Does this object carry every field we read, in a shape we can use?"""
    box = after.get("box")
    return bool(
        isinstance(box, list)
        and len(box) == _BOX_VALUES
        and all(_is_number(value) for value in box)
        and all(after.get(key) is not None for key in _REQUIRED)
        and _is_number(after["frame_time"])
    )


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


class MqttSubscriber(Protocol):
    """The part of `paho`'s client this source uses, so a test can stand in for it."""

    on_message: Any

    def username_pw_set(self, username: str, password: str) -> object: ...
    def connect(self, host: str, port: int, keepalive: int) -> object: ...
    def subscribe(self, topic: str) -> object: ...
    def loop_start(self) -> object: ...
    def disconnect(self) -> object: ...
    def loop_stop(self) -> object: ...


class FrigateTrackSource:
    """The plumbing: subscribe, buffer, and yield the ticks `FrigateObjects` produces.

    Everything decided is in `FrigateObjects`; this holds a socket and a queue.

    `loop_start` hands the connection to paho's own thread, which is what gives this
    source its reconnect behaviour for free — a broker that comes back does not need the
    engine restarted, exactly as the exporter side works.
    """

    def __init__(
        self,
        camera_id: CameraId,
        *,
        broker: str,
        port: int,
        topic: str,
        objects: FrigateObjects,
        credentials: tuple[str, str] | None = None,
        idle_tick_s: float = IDLE_TICK_S,
        client: MqttSubscriber | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._camera_id = camera_id
        self._broker = broker
        self._port = port
        self._topic = topic
        self._objects = objects
        self._credentials = credentials
        self._idle_tick_s = idle_tick_s
        self._monotonic = monotonic
        self._inbox: queue.Queue[bytes] = queue.Queue(maxsize=INBOX_SIZE)
        self._closed = False
        default = Client(CallbackAPIVersion.VERSION2)
        self._client: MqttSubscriber = client if client is not None else default
        self._client.on_message = self._on_message

    def ticks(self) -> Iterator[tuple[FrameTs, list[Track]]]:
        """Yield a tick per message, and one every `idle_tick_s` when none arrive.

        The idle tick is not a nicety. Frigate says nothing when nothing moves, so without
        it a quiet shop produces no occupancy samples at all and reads as unmeasured
        rather than as empty — and nothing would ever expire an object whose `end` was
        missed, since expiry only runs on a tick.
        """
        self._connect()
        while not self._closed:
            self._drain()
            now = self._monotonic()
            yield self._objects.tick_ts(now=now), self._objects.tick(now=now)

    def close(self) -> None:
        self._closed = True
        with suppress(Exception):
            self._client.disconnect()
            self._client.loop_stop()

    def _connect(self) -> None:
        if self._credentials is not None:
            self._client.username_pw_set(*self._credentials)
        self._client.connect(self._broker, self._port, KEEPALIVE_S)
        self._client.subscribe(self._topic)
        self._client.loop_start()

    def _on_message(self, _client: object, _userdata: object, message: Any) -> None:
        """Paho's thread calls this. It must not block and must not raise.

        The queue is bounded and the oldest is dropped when it fills: a supervisor stall
        must cost the stalest objects rather than back up into paho's network loop, which
        is the same rule the worker's own outbox follows.
        """
        with suppress(Exception):
            if self._inbox.full():
                with suppress(queue.Empty):
                    self._inbox.get_nowait()
            self._inbox.put_nowait(bytes(message.payload))

    def _drain(self) -> None:
        """Take what has arrived, waiting up to one idle interval for the first."""
        deadline = self._monotonic() + self._idle_tick_s
        try:
            payload = self._inbox.get(timeout=self._idle_tick_s)
        except queue.Empty:
            return
        self._objects.apply(payload, now=self._monotonic())
        while self._monotonic() < deadline:
            try:
                payload = self._inbox.get_nowait()
            except queue.Empty:
                return
            self._objects.apply(payload, now=self._monotonic())
