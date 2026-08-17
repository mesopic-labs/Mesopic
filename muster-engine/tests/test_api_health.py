"""What `/healthz` is allowed to claim (engine-architecture.md §15).

The endpoint is the only view an operator gets of a box they cannot SSH into, so these
tests are mostly about honesty rather than plumbing:

* `degraded` is a first-class state — one flapping camera must not read as a dead box,
  and must not read as a healthy one either;
* a camera switched off in config is not a broken camera, and saying so would page
  someone about a decision they made;
* the two fields the supervisor genuinely does not know — `last_frame_ts` and
  `effective_fps` — serialise as `null` rather than as a plausible-looking guess.

Red-first for P3.1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from muster.api.health import (
    DISK_PRESSURE_PCT,
    CameraHealth,
    DiskHealth,
    HealthStatus,
    SyncHealth,
    camera_health,
    engine_health,
)
from muster.config.schema import MusterConfig
from muster.supervisor.handle import WorkerReport
from muster.types import CameraId, CameraState

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_CONFIG = REPO_ROOT / "examples" / "muster.yaml"

FRONT_DOOR = CameraId("front-door")
TILL = CameraId("till")


@pytest.fixture
def config() -> MusterConfig:
    parsed: dict[str, Any] = yaml.safe_load(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    return MusterConfig.model_validate(parsed)


def _healthy(camera_id: CameraId = FRONT_DOOR) -> CameraHealth:
    return CameraHealth(
        camera_id=camera_id,
        state=CameraState.STREAMING,
        last_frame_ts=None,
        consecutive_failures=0,
        effective_fps=None,
    )


def _sick(camera_id: CameraId = TILL) -> CameraHealth:
    return CameraHealth(
        camera_id=camera_id,
        state=CameraState.BACKOFF,
        last_frame_ts=None,
        consecutive_failures=7,
        effective_fps=None,
    )


def _engine(*cameras: CameraHealth) -> Any:
    return engine_health(
        cameras,
        uptime_s=84213.0,
        schema_version=1,
        sync=SyncHealth(enabled=False, unsynced_rows=0),
        disk=DiskHealth(free_pct=50, pressure=False),
    )


# --- The status rule --------------------------------------------------------


def test_every_camera_streaming_is_ok() -> None:
    assert _engine(_healthy(FRONT_DOOR), _healthy(TILL)).status is HealthStatus.OK


def test_one_flapping_camera_degrades_the_engine_without_downing_it() -> None:
    """The other cameras keep counting; the dead one's metrics show a gap (§15)."""
    assert _engine(_healthy(FRONT_DOOR), _sick(TILL)).status is HealthStatus.DEGRADED


def test_a_camera_that_has_crashed_degrades_the_engine_even_while_it_is_up() -> None:
    """Caught live against a camera with an unreachable URL: `status: ok`, and nothing counted.

    A camera that cannot connect *flaps* — the worker starts, fails, dies, backs off and
    starts again — and process liveness is `STREAMING` for most of that cycle, so polling
    at any moment usually finds it "up". Judging on state alone therefore reports a box
    that has never counted anything as healthy.

    So a non-zero failure count is degrading in its own right. It latches: nothing resets
    it short of a restart, so one blip last Tuesday still shows. That is the deliberate
    direction to be wrong in — the count is right there in the payload saying how bad it
    was, and the alternative is a dead camera reported as fine. The heartbeat that carries
    real stream state is what retires this.
    """
    flapping = CameraHealth(
        camera_id=TILL,
        state=CameraState.STREAMING,
        last_frame_ts=None,
        consecutive_failures=3,
        effective_fps=None,
    )

    assert _engine(_healthy(FRONT_DOOR), flapping).status is HealthStatus.DEGRADED


def test_no_camera_producing_is_down() -> None:
    """`down` is reserved for "nothing is being counted", which this is."""
    assert _engine(_sick(FRONT_DOOR), _sick(TILL)).status is HealthStatus.DOWN


def test_a_disabled_camera_neither_degrades_nor_downs_the_engine() -> None:
    """Switching a camera off is a decision, not a fault."""
    disabled = CameraHealth(
        camera_id=TILL,
        state=CameraState.DISABLED,
        last_frame_ts=None,
        consecutive_failures=0,
        effective_fps=None,
    )

    assert _engine(_healthy(FRONT_DOOR), disabled).status is HealthStatus.OK


def test_an_engine_with_every_camera_disabled_is_down() -> None:
    """Nothing is being counted, and an operator looking at the box should see that."""
    disabled = CameraHealth(
        camera_id=FRONT_DOOR,
        state=CameraState.DISABLED,
        last_frame_ts=None,
        consecutive_failures=0,
        effective_fps=None,
    )

    assert _engine(disabled).status is HealthStatus.DOWN


# --- Merging config with what the supervisor runs ---------------------------


def test_a_camera_the_supervisor_runs_reports_its_worker_state(config: MusterConfig) -> None:
    states = {
        camera.camera_id: WorkerReport(state=CameraState.STREAMING, consecutive_failures=0)
        for camera in config.cameras
    }

    reported = {health.camera_id: health.state for health in camera_health(config, states)}

    assert reported == {camera_id: report.state for camera_id, report in states.items()}


def test_a_camera_disabled_in_config_is_reported_as_disabled(config: MusterConfig) -> None:
    """The supervisor has no worker for it, so this is the only place it can be named."""
    parsed = config.model_dump()
    parsed["cameras"][0]["enabled"] = False
    partly_disabled = MusterConfig.model_validate(parsed)
    running = {
        camera.camera_id: WorkerReport(state=CameraState.STREAMING, consecutive_failures=0)
        for camera in partly_disabled.cameras[1:]
    }

    reported = {
        health.camera_id: health.state for health in camera_health(partly_disabled, running)
    }

    assert reported[config.cameras[0].camera_id] is CameraState.DISABLED


def test_every_configured_camera_appears_exactly_once(config: MusterConfig) -> None:
    """A camera missing from `/healthz` reads as a camera that does not exist."""
    states = {
        config.cameras[0].camera_id: WorkerReport(
            state=CameraState.STREAMING, consecutive_failures=0
        )
    }

    reported = [health.camera_id for health in camera_health(config, states)]

    assert sorted(reported) == sorted(camera.camera_id for camera in config.cameras)


def test_the_failure_count_is_reported_not_assumed(config: MusterConfig) -> None:
    """§15 shows this field as the thing that explains *why* the engine is degraded.

    It is the one number in `CameraHealth` the supervisor really does know, so reporting
    a constant beside `state: backoff` would be exactly the plausible-looking guess the
    two `None` fields below exist to refuse.
    """
    flapping = {
        config.cameras[0].camera_id: WorkerReport(state=CameraState.BACKOFF, consecutive_failures=7)
    }

    (first, *_) = camera_health(config, flapping)

    assert first.consecutive_failures == 7


def test_the_fields_the_supervisor_cannot_know_are_none(config: MusterConfig) -> None:
    """A plausible-looking guess in a health field is worse than an absent one.

    `last_frame_ts` and `effective_fps` live inside the worker process and are never
    reported back (see `WorkerHandle.state`). Until a status heartbeat exists they are
    `None`, and this test is what stops one being back-filled from something adjacent —
    the newest metric bucket, say, which is silent for a healthy camera nobody walks past.
    """
    states = {
        config.cameras[0].camera_id: WorkerReport(
            state=CameraState.STREAMING, consecutive_failures=0
        )
    }

    (first, *_) = camera_health(config, states)

    assert first.last_frame_ts is None
    assert first.effective_fps is None


# --- Disk -------------------------------------------------------------------


def test_a_nearly_full_disk_is_reported_as_pressure() -> None:
    engine = engine_health(
        [_healthy()],
        uptime_s=1.0,
        schema_version=1,
        sync=SyncHealth(enabled=False, unsynced_rows=0),
        disk=DiskHealth(free_pct=DISK_PRESSURE_PCT - 1, pressure=True),
    )

    assert engine.disk.pressure is True
    assert engine.status is HealthStatus.DEGRADED


def test_disk_pressure_is_measured_from_the_stores_own_filesystem(tmp_path: Path) -> None:
    """Free space on `/` says nothing about the volume the database is mounted on."""
    disk = DiskHealth.of(tmp_path)

    assert 0 <= disk.free_pct <= 100
    assert disk.pressure is (disk.free_pct < DISK_PRESSURE_PCT)
