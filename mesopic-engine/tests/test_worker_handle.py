"""One camera's process, its bounded queue, and its restart policy.

These tests spawn real OS processes and put on real `multiprocessing.Queue`s. That is
deliberate: what P2.7 is *for* is the behaviour of a worker that dies, a queue that
fills, and a payload that has to survive pickling — none of which an in-process fake
queue would exercise. What is left out is only the CV pipeline (see `scripted_worker`).

Red-first for P2.7.
"""

from __future__ import annotations

from collections.abc import Iterator
from functools import partial
from pathlib import Path

import pytest
import yaml
from scripted_worker import emit_then_die, emit_then_exit, run_until_stopped

from mesopic.config.schema import MesopicConfig
from mesopic.supervisor.handle import WorkerHandle
from mesopic.types import CameraId, EventKind

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_CONFIG = REPO_ROOT / "examples" / "mesopic.yaml"

FRONT_DOOR = CameraId("front-door")


@pytest.fixture
def config() -> MesopicConfig:
    parsed = yaml.safe_load(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    return MesopicConfig.model_validate(parsed)


@pytest.fixture
def handles() -> Iterator[list[WorkerHandle]]:
    """Stop whatever a test started, even when it fails — a leaked process hangs CI."""
    started: list[WorkerHandle] = []
    yield started
    for handle in started:
        handle.stop(timeout=2.0)


def test_a_started_worker_puts_events_the_handle_can_drain(
    config: MesopicConfig, handles: list[WorkerHandle]
) -> None:
    """The whole point of the process boundary: small events cross it intact."""
    handle = WorkerHandle(FRONT_DOOR, config=config, entry=partial(emit_then_exit, crossings=3))
    handles.append(handle)

    handle.start()
    events = handle.drain_blocking(expected=3, timeout=5.0)

    assert len(events) == 3
    assert {event.camera_id for event in events} == {FRONT_DOOR}
    assert {event.kind for event in events} == {EventKind.LINE_CROSS}


def test_a_worker_that_exits_is_no_longer_alive(
    config: MesopicConfig, handles: list[WorkerHandle]
) -> None:
    handle = WorkerHandle(FRONT_DOOR, config=config, entry=partial(emit_then_exit, crossings=1))
    handles.append(handle)

    handle.start()
    handle.drain_blocking(expected=1, timeout=5.0)
    handle.wait_exit(timeout=5.0)

    assert not handle.is_alive()


def test_events_put_before_a_crash_still_arrive(
    config: MesopicConfig, handles: list[WorkerHandle]
) -> None:
    """A worker killed mid-run must not take its already-queued events with it.

    This is the fault isolation §9 claims: one camera dying costs that camera's next
    frames, never the counts it already produced.
    """
    handle = WorkerHandle(FRONT_DOOR, config=config, entry=partial(emit_then_die, crossings=2))
    handles.append(handle)

    handle.start()
    events = handle.drain_blocking(expected=2, timeout=5.0)

    assert len(events) == 2


def test_a_stopped_worker_is_asked_before_it_is_killed(
    config: MesopicConfig, handles: list[WorkerHandle]
) -> None:
    """`stop` sends the control message first; `terminate` is the fallback, not the plan."""
    handle = WorkerHandle(FRONT_DOOR, config=config, entry=run_until_stopped)
    handles.append(handle)

    handle.start()
    handle.stop(timeout=5.0)

    assert not handle.is_alive()
    assert handle.stopped_cleanly


def test_the_control_message_a_stop_sends_is_the_documented_one(
    config: MesopicConfig, handles: list[WorkerHandle]
) -> None:
    """A worker that loops on a different sentinel would hang until the kill timeout.

    Asserted by stopping a real worker rather than by comparing constants: since P3.8 the
    message is a `Stop()` instance, and two sides agreeing on a *type* is exactly what an
    equality check between two imports of the same name cannot tell you.
    """
    handle = WorkerHandle(FRONT_DOOR, config=config, entry=run_until_stopped)
    handles.append(handle)

    handle.start()
    handle.request_stop()
    handle.wait_exit(timeout=5.0)

    assert not handle.is_alive()


# --- Restart backoff --------------------------------------------------------


def test_a_fresh_handle_is_due_for_restart_immediately(config: MesopicConfig) -> None:
    handle = WorkerHandle(FRONT_DOOR, config=config, entry=partial(emit_then_exit))

    assert handle.restart_due(now=0.0)


def test_backoff_grows_with_consecutive_failures(config: MesopicConfig) -> None:
    """A camera whose RTSP URL is wrong must not be respawned in a tight loop."""
    handle = WorkerHandle(FRONT_DOOR, config=config, entry=partial(emit_then_exit))

    handle.note_failure(now=100.0)
    first = handle.restart_due(now=100.5)
    handle.note_failure(now=101.0)
    second = handle.restart_due(now=102.5)

    assert not first, "a worker must not be respawned in the same instant it died"
    assert not second, "the second failure must wait longer than the first"
    assert handle.restart_due(now=110.0)


def test_backoff_is_capped(config: MesopicConfig) -> None:
    """Unbounded doubling means a camera that recovers is never retried again."""
    handle = WorkerHandle(FRONT_DOOR, config=config, entry=partial(emit_then_exit))

    for tick in range(20):
        handle.note_failure(now=float(tick))

    assert handle.restart_due(now=19.0 + WorkerHandle.MAX_BACKOFF_S)


def test_a_successful_run_clears_the_backoff(config: MesopicConfig) -> None:
    """Backoff is for a camera that keeps failing, not one that failed hours ago."""
    handle = WorkerHandle(FRONT_DOOR, config=config, entry=partial(emit_then_exit))

    handle.note_failure(now=100.0)
    handle.note_started(now=200.0)
    handle.note_failure(now=300.0)

    assert not handle.restart_due(now=300.5)
    assert handle.restart_due(now=302.0)
