"""The `/healthz` payload: engine and per-camera health (engine-architecture.md §15).

This is the only view an operator gets of a box they cannot SSH into, so it reports
degradation honestly: a camera in BACKOFF is degraded, not fatal — the other cameras keep
counting and the dead one's metrics simply show a gap.
"""

from __future__ import annotations

from dataclasses import dataclass

from muster.types import CameraId, CameraState, FrameTs


@dataclass(frozen=True, slots=True)
class CameraHealth:
    """One camera's liveness, as reported to `/healthz` and the dashboard."""

    camera_id: CameraId
    state: CameraState
    last_frame_ts: FrameTs | None
    consecutive_failures: int
    effective_fps: float
