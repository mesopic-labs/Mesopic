"""The `/healthz` payload: engine and per-camera health (engine-architecture.md §15).

This is the only view an operator gets of a box they cannot SSH into, so it reports
degradation honestly: a camera in BACKOFF is degraded, not fatal — the other cameras keep
counting and the dead one's metrics simply show a gap.

Two deliberate silences, both of which would be easy to fill with something plausible:

* **`last_frame_ts` and `effective_fps` are `None`.** They are known inside the worker
  process and never reported back to the supervisor, so there is nothing here to report
  (see `WorkerHandle.state`). Back-filling them from the newest metric bucket would make
  a healthy camera nobody walks past look stalled.
* **`down` is only decided from the cameras.** §15 also reserves it for "store
  unwritable", which needs a write-failure signal the supervisor does not yet surface — a
  probe write on every poll would be its own small disease. Named here so the gap is
  visible rather than assumed closed.

Implements P3.1.
"""

from __future__ import annotations

import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from muster.config.schema import MusterConfig
from muster.supervisor.handle import WorkerReport
from muster.types import CameraId, CameraState, FrameTs

DISK_PRESSURE_PCT = 10
"""Below this much free space the engine says so. Handling it is M6 (§14); saying it is
free, and an operator who can see it coming can act before the store stops writing."""


class HealthStatus(StrEnum):
    """§15's three states. `degraded` is first-class: some metrics have gaps, the box works."""

    OK = "ok"
    DEGRADED = "degraded"
    DOWN = "down"


@dataclass(frozen=True, slots=True)
class CameraHealth:
    """One camera's liveness, as reported to `/healthz` and the dashboard."""

    camera_id: CameraId
    state: CameraState
    last_frame_ts: FrameTs | None
    consecutive_failures: int
    effective_fps: float | None


@dataclass(frozen=True, slots=True)
class SyncHealth:
    """How far behind the cloud this box is. Meaningful even when sync is off."""

    enabled: bool
    unsynced_rows: int


@dataclass(frozen=True, slots=True)
class DiskHealth:
    free_pct: int
    pressure: bool

    @classmethod
    def of(cls, path: Path) -> DiskHealth:
        """Free space on the filesystem holding `path` — the store's, not the root's.

        An appliance with the database on a separate data volume would otherwise report
        the health of a partition nothing is written to.
        """
        usage = shutil.disk_usage(path)
        free_pct = int(usage.free * 100 / usage.total) if usage.total else 0
        return cls(free_pct=free_pct, pressure=free_pct < DISK_PRESSURE_PCT)


@dataclass(frozen=True, slots=True)
class EngineHealth:
    """The whole `/healthz` document."""

    status: HealthStatus
    uptime_s: int
    schema_version: int
    sync: SyncHealth
    disk: DiskHealth
    cameras: tuple[CameraHealth, ...]


def camera_health(
    config: MusterConfig, reports: Mapping[CameraId, WorkerReport]
) -> list[CameraHealth]:
    """Every configured camera, whether or not the supervisor runs a worker for it.

    A camera absent from `/healthz` reads as a camera that does not exist, so config —
    not the worker set — decides who appears. One the operator disabled is `DISABLED`
    rather than missing or broken.
    """
    return [
        CameraHealth(
            camera_id=camera.camera_id,
            state=_state_of(reports.get(camera.camera_id)),
            last_frame_ts=None,
            consecutive_failures=_failures_of(reports.get(camera.camera_id)),
            effective_fps=None,
        )
        for camera in config.cameras
    ]


def _state_of(report: WorkerReport | None) -> CameraState:
    return CameraState.DISABLED if report is None else report.state


def _failures_of(report: WorkerReport | None) -> int:
    return 0 if report is None else report.consecutive_failures


def engine_health(
    cameras: Sequence[CameraHealth],
    *,
    uptime_s: float,
    schema_version: int,
    sync: SyncHealth,
    disk: DiskHealth,
) -> EngineHealth:
    """Fold the parts into one document, and decide the one field that is a judgement."""
    return EngineHealth(
        status=_status(cameras, disk),
        uptime_s=int(uptime_s),
        schema_version=schema_version,
        sync=sync,
        disk=disk,
        cameras=tuple(cameras),
    )


def _status(cameras: Sequence[CameraHealth], disk: DiskHealth) -> HealthStatus:
    """`down` if nothing is counting, `degraded` if something is wrong but counting continues.

    A camera the operator disabled is excluded from the judgement entirely: it is neither
    evidence of health nor of failure. An engine whose every camera is disabled is `down`
    — nothing is being counted, however deliberately.

    **A past failure degrades even a camera that is currently up.** A camera whose URL is
    wrong flaps rather than staying down — start, fail, die, back off, start — and process
    liveness reads as `STREAMING` for most of that cycle, so judging on the current state
    alone reports a box that has never counted anything as healthy. The count latches
    until a restart, which is the right direction to be wrong in: it is visible in the
    payload and it never says "fine" about a camera that is not.
    """
    streaming = [camera for camera in cameras if camera.state is CameraState.STREAMING]
    if not streaming:
        return HealthStatus.DOWN
    faulty = [
        camera
        for camera in cameras
        if camera.state not in (CameraState.STREAMING, CameraState.DISABLED)
        or camera.consecutive_failures > 0
    ]
    if faulty or disk.pressure:
        return HealthStatus.DEGRADED
    return HealthStatus.OK
