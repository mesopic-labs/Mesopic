"""One camera's pipeline, in its own process.

The loop is `ingest -> sample -> detect -> track -> analytics -> emit`. The frame is a
local variable inside a single iteration: it is never stored, never buffered across
ticks, and never sent anywhere. When the iteration ends, it is gone (ADR-0005).
"""

from __future__ import annotations

from muster.types import CameraId


def run_camera_worker(camera_id: CameraId) -> None:
    """Entry point for the per-camera process. Blocks until told to stop."""
    raise NotImplementedError
