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
from dataclasses import dataclass
from multiprocessing.queues import Queue

from muster.analytics.geometry import GeometryAnalytics
from muster.config.schema import MusterConfig
from muster.detector.detector import Detector
from muster.errors import StreamDropped
from muster.ingest.source import FrameSource
from muster.sampler.sampler import FrameSampler
from muster.supervisor.backpressure import Backpressure, Outbox, Sink
from muster.supervisor.pipeline import build_pipeline
from muster.tracker.tracker import Tracker
from muster.types import CameraId, RawEvent

OUTBOX_SIZE = 256
"""Events buffered locally before the oldest is dropped. Small on purpose: this is a
stall absorber, not a store, and everything in it is at risk if the process dies."""

STOP = "stop"
"""The control message this loop returns on. `WorkerHandle.STOP` must match."""


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
    control: Queue[str] | None = None,
) -> LoopStats:
    """Run one camera until its stream drops or it is told to stop.

    Returns rather than raises on a dropped stream: for a camera, "the stream ended" is
    a drop, and it is the way every real run ends. The supervisor decides whether that
    deserves a restart; the loop's job is to release the socket and say what happened.
    """
    stats = LoopStats()
    outbox: Outbox[RawEvent] = Outbox(maxlen=outbox_size)
    pressure = Backpressure(sampler, fps_min=fps_min, fps_max=fps_max, start_fps=fps_max)
    try:
        for frame in source.frames():
            if _stop_requested(control):
                break
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


def _stop_requested(control: Queue[str] | None) -> bool:
    if control is None:
        return False
    try:
        return control.get_nowait() == STOP
    except queue.Empty:
        return False


def run_camera_worker(
    camera_id: CameraId,
    config: MusterConfig,
    events_out: Queue[RawEvent],
    control_in: Queue[str],
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
    )
