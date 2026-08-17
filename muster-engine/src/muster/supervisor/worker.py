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
from contextlib import suppress
from dataclasses import dataclass
from multiprocessing.queues import Queue

from muster.analytics.geometry import GeometryAnalytics
from muster.config.schema import MusterConfig
from muster.detector.detector import Detector
from muster.errors import StreamDropped
from muster.ingest.source import FrameSource
from muster.sampler.sampler import FrameSampler
from muster.supervisor.backpressure import Backpressure, Outbox, Sink
from muster.supervisor.control import (
    ControlMessage,
    Reconfigure,
    Snapshot,
    SnapshotReply,
    Stop,
)
from muster.supervisor.pipeline import build_pipeline
from muster.supervisor.snapshot import encode_snapshot
from muster.tracker.tracker import Tracker
from muster.types import CameraId, DecodedFrame, RawEvent

OUTBOX_SIZE = 256
"""Events buffered locally before the oldest is dropped. Small on purpose: this is a
stall absorber, not a store, and everything in it is at risk if the process dies."""


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
    try:
        for frame in source.frames():
            message = _next_message(control)
            if isinstance(message, Stop):
                break
            if message is not None and not isinstance(message, Stop):
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


def run_camera_worker(
    camera_id: CameraId,
    config: MusterConfig,
    events_out: Queue[RawEvent],
    control_in: Queue[ControlMessage],
    snapshots_out: Queue[SnapshotReply] | None = None,
) -> None:
    """Entry point for the per-camera process. Blocks until told to stop."""
    pipeline = build_pipeline(config, camera_id)
    camera_loop(
        camera_id,
        source=pipeline.source,
        sampler=pipeline.sampler,
        detector=pipeline.detector,
        tracker=pipeline.tracker,
        analytics=pipeline.analytics,
        sink=events_out,
        fps_min=config.budget.fps_min,
        fps_max=config.budget.fps_max,
        control=control_in,
        snapshots=snapshots_out,
    )
