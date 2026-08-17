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
from muster.types import CameraId, EventKind, FrameTs, LineId, RawEvent, TrackId, ZoneId

CONTROL_STOP = "stop"


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


def emit_then_exit(
    camera_id: CameraId,
    config: MusterConfig,  # the entry-point contract
    events_out: Queue[RawEvent],
    control_in: Queue[str],
    *,
    crossings: int = 1,
) -> None:
    """Put `crossings` line-cross events, then return normally."""
    for index in range(crossings):
        events_out.put(_crossing(camera_id, index))
    events_out.close()
    events_out.join_thread()


def emit_then_die(
    camera_id: CameraId,
    config: MusterConfig,
    events_out: Queue[RawEvent],
    control_in: Queue[str],
    *,
    crossings: int = 1,
) -> None:
    """Put events, then `SIGKILL` itself — the crash the supervisor must survive."""
    for index in range(crossings):
        events_out.put(_crossing(camera_id, index))
    events_out.close()
    events_out.join_thread()
    os.kill(os.getpid(), signal.SIGKILL)


def emit_occupancy_sample(
    camera_id: CameraId,
    config: MusterConfig,
    events_out: Queue[RawEvent],
    control_in: Queue[str],
) -> None:
    """One trackless occupancy sample — the kind the raw event log cannot hold."""
    events_out.put(
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
    events_out.close()
    events_out.join_thread()


def run_until_stopped(
    camera_id: CameraId,
    config: MusterConfig,
    events_out: Queue[RawEvent],
    control_in: Queue[str],
) -> None:
    """Idle until told to stop — the shape a real worker has."""
    while control_in.get() != CONTROL_STOP:
        time.sleep(0.01)
    events_out.close()
    events_out.join_thread()
