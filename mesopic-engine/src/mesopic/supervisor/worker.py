"""One camera's pipeline, in its own process.

The loop is `ingest -> sample -> detect -> track -> analytics -> emit`. The frame is a
local variable inside a single iteration: it is never stored, never buffered across
ticks, and never sent anywhere. When the iteration ends, it is gone (ADR-0005).

`camera_loop` takes its parts rather than building them, so the loop can be tested with
no network and the wiring can be tested with no loop (`pipeline.build_pipeline`).

Implements P2.7 (engine-architecture.md §9).
"""

from __future__ import annotations

import queue
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from multiprocessing.queues import Queue

from mesopic.analytics.geometry import GeometryAnalytics
from mesopic.config.schema import MesopicConfig
from mesopic.detector.detector import Detector
from mesopic.errors import StreamDropped
from mesopic.ingest.source import FrameSource, TrackSource
from mesopic.sampler.sampler import FrameSampler
from mesopic.supervisor.backpressure import Backpressure, Outbox, Sink
from mesopic.supervisor.control import (
    ControlMessage,
    Heartbeat,
    Reconfigure,
    Retarget,
    Snapshot,
    SnapshotReply,
    Stop,
    WorkerChannels,
)
from mesopic.supervisor.pipeline import TrackPipeline, build_pipeline
from mesopic.supervisor.snapshot import encode_snapshot
from mesopic.tracker.tracker import Tracker
from mesopic.types import CameraId, DecodedFrame, FrameTs, RawEvent

OUTBOX_SIZE = 256
"""Events buffered locally before the oldest is dropped. Small on purpose: this is a
stall absorber, not a store, and everything in it is at risk if the process dies."""


HEARTBEAT_INTERVAL_S = 1.0
"""How often a worker reports itself. Ten of these make `handle.STALL_AFTER_S`.

Not per admitted frame: the heartbeat queue holds four, the supervisor drains once a
second, and a beat per frame at `fps_max` would overrun the queue on every tick to say
the same thing four times.
"""


class HeartbeatEmitter:
    """Reports the rate a worker *achieved*, at most once per interval.

    Achieved, not targeted: `Backpressure.target_fps` is what the loop asked the sampler
    for, and the two diverge exactly when something is wrong. A camera delivering half
    what it promises is the case this exists to make visible.

    **Silence is the signal.** Nothing here is driven by a clock — only by a frame being
    admitted — so a worker blocked inside `source.frames()` on a wedged connection emits
    nothing at all. Emitting a zero-fps beat on a timer instead would keep the camera
    reading `STREAMING` forever with a number beside it saying it was doing nothing.
    """

    def __init__(
        self,
        sink: Sink[Heartbeat] | None,
        *,
        interval_s: float = HEARTBEAT_INTERVAL_S,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._sink = sink
        self._interval_s = interval_s
        self._monotonic = monotonic
        self._window_start = monotonic()
        self._admitted = 0
        self._latest: FrameTs | None = None

    def note_admitted(self, ts: FrameTs) -> None:
        """Count one admitted frame, and report the window if it has closed."""
        self._admitted += 1
        self._latest = ts
        elapsed = self._monotonic() - self._window_start
        if elapsed < self._interval_s:
            return
        self._emit(self._admitted / elapsed)
        self._window_start = self._monotonic()
        self._admitted = 0

    def _emit(self, effective_fps: float) -> None:
        """Send, or drop it — never die of it.

        Same rule as a snapshot reply: a supervisor that stopped draining must not turn
        a status message into a dropped stream. An unreported camera is a far better
        outcome than one that stops counting because nobody was listening.
        """
        if self._sink is None or self._latest is None:
            return
        with suppress(queue.Full):
            self._sink.put_nowait(
                Heartbeat(last_frame_ts=self._latest, effective_fps=effective_fps)
            )


@dataclass(slots=True)
class LoopStats:
    """What the loop did, for the supervisor's log line and `/healthz` later."""

    frames_admitted: int = 0
    events_emitted: int = 0
    dropped_events: int = 0
    stream_dropped: bool = False


def camera_loop(
    camera_id: CameraId,
    *,
    source: FrameSource,
    sampler: FrameSampler,
    detector: Detector,
    tracker: Tracker,
    analytics: GeometryAnalytics,
    sink: Sink[RawEvent],
    fps_min: float,
    fps_max: float,
    outbox_size: int = OUTBOX_SIZE,
    control: Queue[ControlMessage] | None = None,
    snapshots: Queue[SnapshotReply] | None = None,
    heartbeats: Sink[Heartbeat] | None = None,
    monotonic: Callable[[], float] = time.monotonic,
) -> LoopStats:
    """Run one camera until its stream drops or it is told to stop.

    Returns rather than raises on a dropped stream: for a camera, "the stream ended" is
    a drop, and it is the way every real run ends. The supervisor decides whether that
    deserves a restart; the loop's job is to release the socket and say what happened.

    Exactly one control message is handled per decoded frame. A burst of calibration
    requests therefore costs one frame each rather than starving the stream, and the
    channel drains at the rate the camera actually runs.
    """
    stats = LoopStats()
    outbox: Outbox[RawEvent] = Outbox(maxlen=outbox_size)
    pressure = Backpressure(sampler, fps_min=fps_min, fps_max=fps_max, start_fps=fps_max)
    beats = HeartbeatEmitter(heartbeats, monotonic=monotonic)
    try:
        for frame in source.frames():
            message = _next_message(control)
            if isinstance(message, Stop):
                break
            if isinstance(message, Retarget):
                pressure.set_envelope(fps_min=message.fps_min, fps_max=message.fps_max)
            elif message is not None:
                # A snapshot answers from the frame in hand and a reconfigure closes
                # what is open, so both need this iteration's frame and neither may
                # skip it — the loop carries on to the sampler either way.
                for event in _handle(message, frame, camera_id, analytics, snapshots):
                    outbox.push(event)
                    stats.events_emitted += 1
            # The sampler consumes as well as answers, so it is asked exactly once per
            # decoded frame (P1.3).
            if not sampler.is_due(frame.ts):
                continue
            stats.frames_admitted += 1
            beats.note_admitted(frame.ts)

            detections = detector.detect(frame)
            tracks = tracker.update(frame, detections)
            for event in analytics.on_tracks(camera_id, tracks):
                outbox.push(event)
                stats.events_emitted += 1

            outbox.flush(sink)
            pressure.observe(under_pressure=outbox.under_pressure)
    except StreamDropped:
        stats.stream_dropped = True
    finally:
        outbox.flush(sink)
        stats.dropped_events = outbox.dropped
        source.close()
        detector.close()
    return stats


def track_loop(
    camera_id: CameraId,
    *,
    source: TrackSource,
    sampler: FrameSampler,
    analytics: GeometryAnalytics,
    sink: Sink[RawEvent],
    fps_min: float,
    fps_max: float,
    outbox_size: int = OUTBOX_SIZE,
    control: Queue[ControlMessage] | None = None,
    heartbeats: Sink[Heartbeat] | None = None,
    monotonic: Callable[[], float] = time.monotonic,
) -> LoopStats:
    """Run one camera whose upstream already detected and tracked (P4.3).

    The same loop as `camera_loop` with the middle removed: no decode, no detect, no
    track. What is left is the part that was never about frames — sample, run geometry,
    emit, shed under pressure — so the two share their outbox, their backpressure and
    their heartbeat rather than growing second versions of each.

    The sampler still applies. Frigate's publish rate is not ours to set, but how often we
    run geometry is, and that is the CPU the budget is about.

    **A snapshot request is ignored here, because there is no frame to answer it with.**
    The calibration route refuses a Frigate camera before it ever sends one, so this is
    the backstop rather than the message: a request that did arrive costs one unanswered
    reply and never a stalled camera.
    """
    stats = LoopStats()
    outbox: Outbox[RawEvent] = Outbox(maxlen=outbox_size)
    pressure = Backpressure(sampler, fps_min=fps_min, fps_max=fps_max, start_fps=fps_max)
    beats = HeartbeatEmitter(heartbeats, monotonic=monotonic)
    try:
        for ts, tracks in source.ticks():
            message = _next_message(control)
            if isinstance(message, Stop):
                break
            if isinstance(message, Retarget):
                pressure.set_envelope(fps_min=message.fps_min, fps_max=message.fps_max)
            if isinstance(message, Reconfigure):
                for event in analytics.reconfigure(message.geometry, camera_id=camera_id, ts=ts):
                    outbox.push(event)
                    stats.events_emitted += 1
            if not sampler.is_due(ts):
                continue
            stats.frames_admitted += 1
            beats.note_admitted(ts)

            for event in analytics.on_tracks(camera_id, tracks, ts=ts):
                outbox.push(event)
                stats.events_emitted += 1

            outbox.flush(sink)
            pressure.observe(under_pressure=outbox.under_pressure)
    except StreamDropped:
        stats.stream_dropped = True
    finally:
        outbox.flush(sink)
        stats.dropped_events = outbox.dropped
        source.close()
    return stats


def _next_message(control: Queue[ControlMessage] | None) -> ControlMessage | None:
    if control is None:
        return None
    try:
        return control.get_nowait()
    except queue.Empty:
        return None


def _handle(
    message: Snapshot | Reconfigure,
    frame: DecodedFrame,
    camera_id: CameraId,
    analytics: GeometryAnalytics,
    snapshots: Queue[SnapshotReply] | None,
) -> list[RawEvent]:
    """Act on one control message. Returns whatever events it produced."""
    if isinstance(message, Snapshot):
        _reply(snapshots, message, frame)
        return []
    return analytics.reconfigure(message.geometry, camera_id=camera_id, ts=frame.ts)


def _reply(snapshots: Queue[SnapshotReply] | None, message: Snapshot, frame: DecodedFrame) -> None:
    """Answer a snapshot request, or drop it — never die of it.

    A worker that raised because nobody was listening would turn a calibration request
    into a dropped stream, which is a far worse outcome than an unanswered click. The
    reply queue is bounded, so a caller that walked away cannot make this grow either.
    """
    if snapshots is None:
        return
    with suppress(queue.Full):
        snapshots.put_nowait(
            SnapshotReply(request_id=message.request_id, jpeg=encode_snapshot(frame))
        )


def run_camera_worker(camera_id: CameraId, config: MesopicConfig, channels: WorkerChannels) -> None:
    """Entry point for the per-camera process. Blocks until told to stop.

    Which loop runs is decided by what the source can hand over, not by a flag: a Frigate
    camera has tracks and no frames, so there is nothing for the detector to be given.
    """
    pipeline = build_pipeline(config, camera_id)
    if isinstance(pipeline, TrackPipeline):
        track_loop(
            camera_id,
            source=pipeline.source,
            sampler=pipeline.sampler,
            analytics=pipeline.analytics,
            sink=channels.events,
            fps_min=config.budget.fps_min,
            fps_max=config.budget.fps_max,
            control=channels.control,
            heartbeats=channels.heartbeats,
        )
        return
    camera_loop(
        camera_id,
        source=pipeline.source,
        sampler=pipeline.sampler,
        detector=pipeline.detector,
        tracker=pipeline.tracker,
        analytics=pipeline.analytics,
        sink=channels.events,
        fps_min=config.budget.fps_min,
        fps_max=config.budget.fps_max,
        control=channels.control,
        snapshots=channels.snapshots,
        heartbeats=channels.heartbeats,
    )
