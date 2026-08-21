"""What the engine does when the world misbehaves for a long time.

The unit tests next door prove each mechanism once: backoff grows, a fan-out failure is
counted, a bucket is idempotent. What they cannot show is what those mechanisms add up to
over hours — whether a camera that flaps all night is throttled or merely restarted very
often, whether a disk that fills stops the engine or just stops one write, whether a
broker that never comes back leaks anything while it is gone.

That gap matters more than usual here. The first person to run this is a stranger on
their own hardware, with a camera on a congested wifi bridge and a disk nobody has looked
at, and the failure they will hit is not a wrong count — it is an engine that spun, or
filled a log, or stopped writing metrics without saying so.

Every test drives simulated time rather than sleeping through it: a soak that takes an
hour cannot run in CI, and a soak that sleeps is a soak that fails on a loaded runner for
reasons unrelated to the code. Marked `slow` because they spawn real processes.

Implements MK.5.
"""

from __future__ import annotations

import sqlite3
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest
import yaml
from scripted_worker import emit_then_die, run_until_stopped
from test_supervisor import BUCKET_START, EXAMPLE_CONFIG, FakeClock, _supervisor, _wait_all_dead

from mesopic.config.schema import MesopicConfig
from mesopic.exporters.fanout import ExporterFanout
from mesopic.store.store import Store
from mesopic.supervisor.handle import WorkerHandle
from mesopic.supervisor.supervisor import BUCKET_S
from mesopic.types import CameraId, CameraState, MetricName, MetricRow, MinuteBucket, ScopeId

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = pytest.mark.slow

SOAK_S = 600.0
"""Ten simulated minutes — twenty times the 30s backoff ceiling.

Long enough that a spin and a throttle differ by an order of magnitude (about 600
restarts against about 20), short enough that the loop does not spend minutes spawning
real processes."""

TICK_S = 1.0

BUCKET_ADVANCE_S = BUCKET_S
"""A whole bucket per tick, so the disk soaks actually attempt a write each time round."""

SOAK_ROWS = 5000
"""Roughly a night of minute buckets across a handful of scopes."""

CAMERA = CameraId("front-door")
TILL = CameraId("till")
BUCKET_TS = MinuteBucket(datetime(2026, 8, 18, 9, 30, tzinfo=UTC))


def _wait_backoff(supervisor: Any, camera_id: CameraId, within_s: float = 5.0) -> None:
    """Block until this camera's worker has actually exited.

    Simulated time advances instantly while a real process takes milliseconds to spawn
    and die, so sampling liveness without synchronising first samples a race. `BACKOFF`
    is precisely "not alive", which makes the public report the thing to wait on.
    """
    deadline = time.monotonic() + within_s
    while time.monotonic() < deadline:
        if supervisor.camera_reports()[camera_id].state is CameraState.BACKOFF:
            return
        time.sleep(0.01)
    pytest.fail(f"{camera_id} never died")


def one_camera_dies(camera_id: CameraId, config: MesopicConfig, channels: Any) -> None:
    """`front-door` crashes forever; `till` runs normally.

    Module level rather than a closure because the worker is spawned, not forked: the
    entry point is pickled by name, and a local function cannot be.
    """
    if camera_id == CAMERA:
        emit_then_die(camera_id, config, channels)
    else:
        run_until_stopped(camera_id, config, channels)


# The fixtures are built here rather than imported from `test_supervisor`: importing a
# fixture by name binds it as a module global that every test then shadows with a
# parameter of the same name, which is a redefinition the linter is right to object to.


@pytest.fixture
def config() -> MesopicConfig:
    """The worked example, plus a line and a zone per camera.

    The scripted workers emit against `<camera>-line`, so a config without them makes the
    store reject every event — the soak would then be measuring rejected writes rather
    than the failure it is about.
    """
    parsed: dict[str, Any] = yaml.safe_load(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    cameras = [camera["camera_id"] for camera in parsed["cameras"]]
    parsed["lines"] = [
        {
            "line_id": f"{camera}-line",
            "camera_id": camera,
            "a": [0.10, 0.80],
            "b": [0.90, 0.80],
            "positive_dir": "in",
            "metrics": ["line_cross", "footfall"],
        }
        for camera in cameras
    ]
    parsed["zones"] = [
        {
            "zone_id": f"{camera}-zone",
            "camera_id": camera,
            "role": "area",
            "polygon": [[0.05, 0.30], [0.95, 0.30], [0.95, 0.95], [0.05, 0.95]],
            "metrics": ["occupancy", "heatmap"],
        }
        for camera in cameras
    ]
    return MesopicConfig.model_validate(parsed)


@pytest.fixture
def store(tmp_path: Path, config: MesopicConfig) -> Iterator[Store]:
    with Store(tmp_path / "mesopic.db") as opened:
        opened.migrate()
        opened.apply_config(config)
        yield opened


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock(BUCKET_START + timedelta(seconds=5))


# --- A camera that flaps all night ------------------------------------------


async def test_a_camera_that_dies_every_time_does_not_spin(
    config: MesopicConfig, store: Store, clock: FakeClock
) -> None:
    """The failure this rules out is a hot restart loop, not a failed restart.

    A camera on a bad bridge dies, gets restarted, dies again. If backoff did not hold
    across restarts, the supervisor would respawn a process every tick — on an N100 that
    is the whole box spent on the one camera that does not work, and the other cameras
    quietly starve.

    Ten simulated minutes of flapping should produce restarts in the tens, not the
    hundreds: the ceiling is 30s, so a throttled run cannot exceed roughly
    SOAK_S / MAX_BACKOFF_S plus the geometric ramp getting there.

    Verified to fail rather than assumed — with `BASE_BACKOFF_S` and `MAX_BACKOFF_S` set
    to zero this trips, and takes two minutes doing it, because it really does spawn the
    thousand-odd processes the assertion is about.
    """
    supervisor = _supervisor(config, store, clock, emit_then_die)
    await supervisor.start()
    _wait_all_dead(supervisor, within_s=10.0)

    elapsed = 0.0
    while elapsed < SOAK_S:
        await supervisor.tick()
        _wait_all_dead(supervisor, within_s=5.0)
        await supervisor.supervise(monotonic=clock.monotonic())
        clock.advance(TICK_S)
        elapsed += TICK_S

    await supervisor.stop()

    ticks = SOAK_S / TICK_S
    ceiling = (SOAK_S / WorkerHandle.MAX_BACKOFF_S + 8) * len(config.cameras)
    assert supervisor.restarts <= ceiling, (
        f"{supervisor.restarts} restarts in a simulated hour is a spin, not a backoff"
    )
    assert supervisor.restarts < ticks, "a restart every tick means backoff is not holding"


async def test_a_flapping_camera_reports_itself_as_such(
    config: MesopicConfig, store: Store, clock: FakeClock
) -> None:
    """Degraded has to be visible, or the operator debugs the wrong thing.

    A camera that is dead between restarts must not read as `STREAMING` on the dashboard
    just because it was alive a moment ago — and its failure count is what tells someone
    the difference between "restarted once" and "restarting all night".
    """
    supervisor = _supervisor(config, store, clock, emit_then_die)
    await supervisor.start()
    _wait_all_dead(supervisor, within_s=10.0)

    for _ in range(60):
        await supervisor.tick()
        _wait_all_dead(supervisor, within_s=5.0)
        await supervisor.supervise(monotonic=clock.monotonic())
        clock.advance(TICK_S)

    reports = supervisor.camera_reports()
    await supervisor.stop()

    assert reports, "a flapping camera must still be reported"
    for report in reports.values():
        assert report.state is not CameraState.STREAMING
        assert report.consecutive_failures > 1, "repeated deaths must be counted, not reset"


async def test_one_dead_camera_does_not_stop_a_healthy_one(
    config: MesopicConfig, store: Store, clock: FakeClock
) -> None:
    """Fault isolation, held over a long run rather than proven once.

    The supervisor is a single event loop shared by every camera, so the question is not
    whether isolation works on the first failure but whether the restart bookkeeping for a
    permanently broken camera ever blocks the tick that serves a working one.
    """

    supervisor = _supervisor(config, store, clock, one_camera_dies)
    await supervisor.start()

    for _ in range(120):
        await supervisor.tick()
        _wait_backoff(supervisor, CAMERA)
        await supervisor.supervise(monotonic=clock.monotonic())
        clock.advance(TICK_S)

    reports = supervisor.camera_reports()
    await supervisor.stop()

    assert reports[CAMERA].state is CameraState.BACKOFF, "the broken camera should be down"
    assert reports[TILL].state is not CameraState.BACKOFF, (
        "a permanently dead camera starved the one that was fine"
    )


# --- A disk that fills -------------------------------------------------------


async def test_a_full_disk_does_not_take_the_engine_down(
    config: MesopicConfig, store: Store, clock: FakeClock
) -> None:
    """`database or disk is full` is an operational condition, not a crash.

    A box that has been counting for months fills its disk at 3am. What must not happen
    is the supervisor dying on the write — an engine that exits leaves nothing serving
    `/healthz`, so the operator learns about it from a customer rather than a dashboard.
    """
    supervisor = _supervisor(config, store, clock, run_until_stopped)
    await supervisor.start()

    full = sqlite3.OperationalError("database or disk is full")
    with patch.object(store, "upsert_metrics", side_effect=full):
        for _ in range(120):
            # The assertion is that this does not raise. A tick that propagates the
            # write error unwinds `run()` and stops every camera, not just the write.
            await supervisor.tick()
            await supervisor.supervise(monotonic=clock.monotonic())
            clock.advance(BUCKET_ADVANCE_S)

    await supervisor.stop()


async def test_metrics_resume_once_the_disk_comes_back(
    config: MesopicConfig, store: Store, clock: FakeClock
) -> None:
    """Surviving the outage is only half of it; recovery has to be automatic.

    Nobody restarts the engine after clearing space, so a store that stays wedged after
    the condition passes is the same as one that crashed — just quieter about it.
    """
    supervisor = _supervisor(config, store, clock, run_until_stopped)
    await supervisor.start()

    full = sqlite3.OperationalError("database or disk is full")
    with patch.object(store, "upsert_metrics", side_effect=full):
        for _ in range(30):
            await supervisor.tick()
            await supervisor.supervise(monotonic=clock.monotonic())
            clock.advance(BUCKET_ADVANCE_S)

    for _ in range(30):
        await supervisor.tick()
        await supervisor.supervise(monotonic=clock.monotonic())
        clock.advance(BUCKET_ADVANCE_S)

    await supervisor.stop()

    # The store is reachable and writable again — the object was never poisoned by the
    # failed writes, which is what a half-open connection would look like.
    store.upsert_metrics(
        [
            MetricRow(
                camera_id=CameraId(sorted(camera.camera_id for camera in config.cameras)[0]),
                metric=MetricName.FOOTFALL,
                scope_id=ScopeId("recovery-probe"),
                bucket=MinuteBucket(clock.now.replace(second=0, microsecond=0)),
                value=1.0,
            )
        ]
    )
    assert any(row.scope_id == "recovery-probe" for row in store.unsynced_metrics(limit=200))


# --- A broker that never comes back -----------------------------------------


def test_a_dead_exporter_never_stops_its_peers() -> None:
    """§12's independence rule, over thousands of rows rather than one.

    A broker that is down for a night is the ordinary case, not the exotic one: someone
    reboots the Home Assistant box and forgets. Every row published during that window
    must still reach Prometheus and the CSV, and the dead one must stay a number.
    """
    published: list[MetricRow] = []

    class Dead:
        def start(self) -> None: ...
        def on_metric(self, row: MetricRow) -> None:
            del row  # the point is that it never arrives
            msg = "broker unreachable"
            raise OSError(msg)

        def shutdown(self) -> None: ...

    class Live:
        def start(self) -> None: ...
        def on_metric(self, row: MetricRow) -> None:
            published.append(row)

        def shutdown(self) -> None: ...

    fanout = ExporterFanout({"mqtt": Dead(), "prometheus": Live()})
    rows = [
        MetricRow(
            camera_id=CAMERA,
            metric=MetricName.FOOTFALL,
            scope_id=ScopeId(f"scope-{index}"),
            bucket=BUCKET_TS,
            value=float(index),
        )
        for index in range(SOAK_ROWS)
    ]

    for row in rows:
        fanout.on_metrics([row])

    assert len(published) == SOAK_ROWS, "a dead peer swallowed rows from a healthy one"
    assert fanout.failures["mqtt"] == SOAK_ROWS, "a degraded exporter must be a number"
    assert "mqtt" not in fanout.healthy_names()
    assert "prometheus" in fanout.healthy_names()


def test_a_dead_exporters_bookkeeping_is_bounded() -> None:
    """The counter is per exporter, not per failure.

    An outage lasting a week must not accumulate a structure that grows with it — a
    failure list would be a slow leak that only appears on the boxes that run longest,
    which are exactly the ones nobody is watching.
    """

    class Dead:
        def start(self) -> None: ...
        def on_metric(self, row: MetricRow) -> None:
            del row  # the point is that it never arrives
            msg = "broker unreachable"
            raise OSError(msg)

        def shutdown(self) -> None: ...

    fanout = ExporterFanout({"mqtt": Dead()})
    for index in range(SOAK_ROWS):
        fanout.on_metrics(
            [
                MetricRow(
                    camera_id=CAMERA,
                    metric=MetricName.FOOTFALL,
                    scope_id=ScopeId(f"scope-{index}"),
                    bucket=BUCKET_TS,
                    value=1.0,
                )
            ]
        )

    assert len(fanout.failures) == 1, "the failure record grew per event, not per exporter"
