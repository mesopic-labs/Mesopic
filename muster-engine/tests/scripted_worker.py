"""Camera workers that emit scripted events instead of decoding video.

A module of its own, and every entry point here is module-level, because the supervisor
pins the `spawn` start method: the child re-imports the target by qualified name, so a
closure or a locally-defined function cannot be a worker entry. `functools.partial` over
one of these is picklable and is how a test scripts what a worker does.

Not a fake in the mock sense — these run in a real process, put on a real
`multiprocessing.Queue`, and die in the ways a real worker dies. What they leave out is
only the CV pipeline.
"""

from __future__ import annotations

import os
import signal
import time
from datetime import UTC, datetime
from multiprocessing.queues import Queue

from muster.config.schema import MusterConfig
from muster.supervisor.control import ControlMessage, Heartbeat, Retarget, Stop, WorkerChannels
from muster.types import CameraId, EventKind, FrameTs, LineId, RawEvent, TrackId, ZoneId

SCRIPTED_FRAME_TS = FrameTs(datetime(2026, 8, 18, 9, 30, tzinfo=UTC))
"""The frame time every scripted heartbeat claims. Fixed so a test can assert on it."""

SCRIPTED_FPS = 2.5
"""The rate every scripted heartbeat claims to have achieved."""


def _beat(fps: float = SCRIPTED_FPS) -> Heartbeat:
    return Heartbeat(last_frame_ts=SCRIPTED_FRAME_TS, effective_fps=fps)


def _idle_until_stopped(control: Queue[ControlMessage]) -> None:
    while not isinstance(control.get(), Stop):
        time.sleep(0.01)


def line_of(camera_id: CameraId) -> LineId:
    """Each camera counts on its own line, so a miscounted row names the wrong camera.

    The supervisor tests build a config to match. Sharing one line id across cameras
    would make an attribution bug invisible — every row would look right.
    """
    return LineId(f"{camera_id}-line")


def _crossing(camera_id: CameraId, index: int) -> RawEvent:
    return RawEvent(
        camera_id=camera_id,
        ts=FrameTs(datetime(2026, 8, 16, 9, 30, index % 60, tzinfo=UTC)),
        kind=EventKind.LINE_CROSS,
        track_id=TrackId(index),
        line_id=line_of(camera_id),
        direction=1,
    )


def _hit(camera_id: CameraId, zone_id: ZoneId, index: int) -> RawEvent:
    return RawEvent(
        camera_id=camera_id,
        ts=FrameTs(datetime(2026, 8, 16, 9, 30, index % 60, tzinfo=UTC)),
        kind=EventKind.HEATMAP_HIT,
        track_id=TrackId(index),
        zone_id=zone_id,
        cell=(3, 4),
        dt_s=1.0,
    )


def emit_then_exit(
    camera_id: CameraId,
    config: MusterConfig,  # the entry-point contract
    channels: WorkerChannels,
    *,
    crossings: int = 1,
    hits: int = 0,
    hit_zone: str = "",
) -> None:
    """Put `crossings` line-cross events and `hits` heatmap hits, then return normally."""
    for index in range(crossings):
        channels.events.put(_crossing(camera_id, index))
    for index in range(hits):
        channels.events.put(_hit(camera_id, ZoneId(hit_zone), index))
    channels.events.close()
    channels.events.join_thread()


def emit_then_die(
    camera_id: CameraId,
    config: MusterConfig,  # the entry-point contract
    channels: WorkerChannels,
    *,
    crossings: int = 1,
) -> None:
    """Put events, then `SIGKILL` itself — the crash the supervisor must survive."""
    for index in range(crossings):
        channels.events.put(_crossing(camera_id, index))
    channels.events.close()
    channels.events.join_thread()
    os.kill(os.getpid(), signal.SIGKILL)


def emit_occupancy_sample(
    camera_id: CameraId,
    config: MusterConfig,  # the entry-point contract
    channels: WorkerChannels,
) -> None:
    """One trackless occupancy sample — the kind the raw event log cannot hold."""
    channels.events.put(
        RawEvent(
            camera_id=camera_id,
            ts=FrameTs(datetime(2026, 8, 16, 9, 30, 5, tzinfo=UTC)),
            kind=EventKind.OCCUPANCY_SAMPLE,
            track_id=None,
            zone_id=ZoneId(f"{camera_id}-zone"),
            value=2.0,
            dt_s=0.4,
        )
    )
    channels.events.close()
    channels.events.join_thread()


def run_until_stopped(
    camera_id: CameraId,
    config: MusterConfig,  # the entry-point contract
    channels: WorkerChannels,
) -> None:
    """Idle until told to stop — the shape a real worker has, heartbeat included.

    A real worker reports itself (P3.7), so one that did not would read as `CONNECT`
    forever and quietly change what every supervisor test using this is asserting.
    """
    channels.heartbeats.put(_beat())
    _idle_until_stopped(channels.control)
    channels.events.close()
    channels.events.join_thread()


def beat_then_idle(
    camera_id: CameraId,
    config: MusterConfig,  # the entry-point contract
    channels: WorkerChannels,
) -> None:
    """Report once, then idle — a worker that is streaming and has nothing to say."""
    channels.heartbeats.put(_beat())
    _idle_until_stopped(channels.control)


def beat_ramp_then_idle(
    camera_id: CameraId,
    config: MusterConfig,  # the entry-point contract
    channels: WorkerChannels,
    *,
    beats: int = 4,
) -> None:
    """Report `beats` times with a rising fps, so "newest" is distinguishable from "first"."""
    for index in range(beats):
        channels.heartbeats.put(_beat(SCRIPTED_FPS + float(index)))
    _idle_until_stopped(channels.control)


def silent_until_stopped(
    camera_id: CameraId,
    config: MusterConfig,  # the entry-point contract
    channels: WorkerChannels,
) -> None:
    """Alive and saying nothing — the wedged stream `/healthz` could not see before P3.7."""
    _idle_until_stopped(channels.control)


def report_retarget_then_stop(
    camera_id: CameraId,
    config: MusterConfig,  # the entry-point contract
    channels: WorkerChannels,
) -> None:
    """Answer each `Retarget` with an event carrying the ceiling it was handed.

    A control message is only useful if it survives the `spawn` pickler and arrives with
    its numbers intact, and neither is observable from the parent — the queue accepts
    anything and `put_nowait` returns before the child has read it. So this worker says
    what it heard the only way a worker can, on the events queue.
    """
    channels.heartbeats.put(_beat())
    while True:
        message = channels.control.get()
        if isinstance(message, Stop):
            break
        if isinstance(message, Retarget):
            channels.events.put(
                RawEvent(
                    camera_id=camera_id,
                    ts=SCRIPTED_FRAME_TS,
                    kind=EventKind.OCCUPANCY_SAMPLE,
                    track_id=None,
                    zone_id=ZoneId(f"{camera_id}-zone"),
                    value=message.fps_max,
                    dt_s=1.0,
                )
            )
    channels.events.close()
    channels.events.join_thread()
