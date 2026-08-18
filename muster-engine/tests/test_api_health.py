"""What `/healthz` is allowed to claim (engine-architecture.md §15).

The endpoint is the only view an operator gets of a box they cannot SSH into, so these
tests are mostly about honesty rather than plumbing:

* `degraded` is a first-class state — one flapping camera must not read as a dead box,
  and must not read as a healthy one either;
* a camera switched off in config is not a broken camera, and saying so would page
  someone about a decision they made;
* `last_frame_ts` and `effective_fps` are passed through from the worker's own
  heartbeat (P3.7) and are `null` when it has not sent one — never back-filled from
  something adjacent that happens to be in reach.

Red-first for P3.1, extended for P3.7.
"""

from __future__ import annotations

from datetime import UTC, datetime
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
from muster.types import CameraId, CameraState, FrameTs

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_CONFIG = REPO_ROOT / "examples" / "muster.yaml"

FRONT_DOOR = CameraId("front-door")
TILL = CameraId("till")
LAST_FRAME_TS = FrameTs(datetime(2026, 8, 18, 9, 42, 58, tzinfo=UTC))


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


def test_a_camera_reports_the_frame_time_and_fps_its_worker_sent(config: MusterConfig) -> None:
    """P3.7 fills in the two fields P3.1 could only serialise as `null`.

    They are passed through from the worker's own heartbeat and nothing else. The value
    of that is entirely in where it comes from: back-filling either from the newest
    metric bucket — the obvious adjacent source — would report a healthy camera nobody
    walks past as stalled.
    """
    states = {
        config.cameras[0].camera_id: WorkerReport(
            state=CameraState.STREAMING,
            consecutive_failures=0,
            last_frame_ts=LAST_FRAME_TS,
            effective_fps=2.5,
        )
    }

    (first, *_) = camera_health(config, states)

    assert first.last_frame_ts == LAST_FRAME_TS
    assert first.effective_fps == pytest.approx(2.5)


def test_a_camera_whose_worker_has_not_reported_still_says_so(config: MusterConfig) -> None:
    """`null` remains the right answer before the first heartbeat, and after a restart.

    A guess here would be worse than an absence: this is the field an operator reads to
    decide whether a camera has stopped, so a plausible number is the one failure mode
    that matters.
    """
    states = {
        config.cameras[0].camera_id: WorkerReport(state=CameraState.CONNECT, consecutive_failures=0)
    }

    (first, *_) = camera_health(config, states)

    assert first.last_frame_ts is None
    assert first.effective_fps is None


def test_a_box_whose_cameras_are_still_connecting_is_degraded_not_down(
    config: MusterConfig,
) -> None:
    """A box two seconds into startup is not a box that has failed.

    `down` is what an orchestrator and the appliance telemetry page on, and every engine
    passes through "no camera has reported yet" on its way up. Reading that as `down`
    would fire an alert on every restart, which is the fastest way to teach an operator
    to ignore the field. Nothing is counting yet, so it is not `ok` either.
    """
    connecting = CameraHealth(
        camera_id=FRONT_DOOR,
        state=CameraState.CONNECT,
        last_frame_ts=None,
        consecutive_failures=0,
        effective_fps=None,
    )

    assert _engine(connecting).status is HealthStatus.DEGRADED


def test_a_stalled_camera_degrades_the_engine(config: MusterConfig) -> None:
    """A wedged camera is exactly as degraded as a dead one, and was invisible before.

    Its process is alive and it has never failed, so `STREAMING` plus a zero failure
    count — every signal `/healthz` had before P3.7 — would have read `ok`.
    """
    stalled = CameraHealth(
        camera_id=FRONT_DOOR,
        state=CameraState.STALLED,
        last_frame_ts=LAST_FRAME_TS,
        consecutive_failures=0,
        effective_fps=0.0,
    )

    assert _engine(_healthy(), stalled).status is HealthStatus.DEGRADED


def test_an_engine_whose_every_camera_has_stalled_is_down(config: MusterConfig) -> None:
    """Nothing is being counted, which is what `down` means (§15)."""
    stalled = CameraHealth(
        camera_id=FRONT_DOOR,
        state=CameraState.STALLED,
        last_frame_ts=LAST_FRAME_TS,
        consecutive_failures=0,
        effective_fps=0.0,
    )

    assert _engine(stalled).status is HealthStatus.DOWN


def test_down_is_still_decided_from_the_cameras_alone(config: MusterConfig) -> None:
    """The OTHER deliberate silence, which P3.7 does not close.

    §15 also reserves `down` for "store unwritable". That needs a write-failure signal
    the supervisor does not surface, and probing with a write on every poll is its own
    small disease. Pinned so filling in `last_frame_ts` is not read as having filled in
    everything §15 asks for.
    """
    unwritable_disk = DiskHealth(free_pct=0, pressure=True)

    engine = engine_health(
        [_healthy()],
        uptime_s=1.0,
        schema_version=1,
        sync=SyncHealth(enabled=False, unsynced_rows=0),
        disk=unwritable_disk,
    )

    assert engine.status is HealthStatus.DEGRADED


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
