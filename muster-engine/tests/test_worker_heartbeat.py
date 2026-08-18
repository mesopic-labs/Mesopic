"""What a worker tells the supervisor about itself, and what `/healthz` may say about it.

Before P3.7 the supervisor saw process liveness and nothing else, so a camera that
connected and then silently stopped delivering frames was indistinguishable from a
healthy one: the process was up, `STREAMING` was reported, and `last_frame_ts` was
`null`. A wedged stream is exactly the failure a health endpoint exists to catch and was
the one it could not see.

Two things these tests are careful about, both easy to get backwards:

* **Staleness is measured on the monotonic clock, at receipt.** Judging it by comparing
  the frame's own UTC timestamp against wall-now would read a camera whose clock is
  skewed as a camera that has stopped.
* **A stale heartbeat is not an absent one.** A worker that died holds its last value
  forever; only a clock tells the two apart.

`camera_state` is a free function rather than a method so the state machine can be
driven directly — every branch below is a fact about the rule, not about a process that
happened to be in the right condition when the test ran. The wiring that feeds it real
values is exercised further down against real spawned workers.

Red-first for P3.7 (engine-architecture.md §9, §15).
"""

from __future__ import annotations

import queue
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from functools import partial
from pathlib import Path
from typing import Any

import pytest
import yaml
from scripted_worker import (
    SCRIPTED_FPS,
    SCRIPTED_FRAME_TS,
    beat_ramp_then_idle,
    beat_then_idle,
    silent_until_stopped,
)

from muster.config.schema import MusterConfig
from muster.supervisor.control import Heartbeat
from muster.supervisor.handle import (
    STALL_AFTER_S,
    WorkerEntry,
    WorkerHandle,
    WorkerReport,
    camera_state,
)
from muster.supervisor.worker import HEARTBEAT_INTERVAL_S, HeartbeatEmitter
from muster.types import CameraId, CameraState, FrameTs

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_CONFIG = REPO_ROOT / "examples" / "muster.yaml"

CAMERA = CameraId("front-door")
FRAME_TS = FrameTs(datetime(2026, 8, 18, 9, 30, tzinfo=UTC))


# --- The state rule ---------------------------------------------------------


def test_a_dead_process_is_in_backoff() -> None:
    state = camera_state(alive=False, since_last_beat_s=0.0, since_start_s=0.0)

    assert state is CameraState.BACKOFF


def test_a_dead_process_is_in_backoff_however_fresh_its_last_heartbeat() -> None:
    """Liveness outranks the heartbeat: a crashed worker's last word is not a pulse."""
    state = camera_state(alive=False, since_last_beat_s=0.0, since_start_s=1000.0)

    assert state is CameraState.BACKOFF


def test_a_live_worker_that_has_not_yet_reported_is_connecting() -> None:
    """Opening an RTSP session takes seconds, and that is not a stall.

    `CONNECT` has been unreachable since P2.7 — a live process read `STREAMING` from the
    instant it spawned, including while it was still dialling. Reporting the truth here
    costs nothing and stops a slow camera looking like a broken one.
    """
    state = camera_state(alive=True, since_last_beat_s=None, since_start_s=1.0)

    assert state is CameraState.CONNECT


def test_a_live_worker_that_never_reports_stops_being_given_the_benefit_of_the_doubt() -> None:
    """A process that came up and never delivered a frame is not connecting any more."""
    state = camera_state(alive=True, since_last_beat_s=None, since_start_s=STALL_AFTER_S + 1.0)

    assert state is CameraState.STALLED


def test_a_worker_that_has_just_reported_is_streaming() -> None:
    state = camera_state(alive=True, since_last_beat_s=0.5, since_start_s=60.0)

    assert state is CameraState.STREAMING


def test_a_live_worker_that_stopped_reporting_is_stalled() -> None:
    """THE failure this whole card exists for.

    The process is up, so `is_alive()` is true and `consecutive_failures` is zero — every
    signal the supervisor had before P3.7 says this camera is fine. Only the absence of a
    heartbeat says otherwise.
    """
    state = camera_state(alive=True, since_last_beat_s=STALL_AFTER_S + 1.0, since_start_s=600.0)

    assert state is CameraState.STALLED


def test_the_stall_threshold_is_not_hit_exactly_at_the_boundary() -> None:
    """A heartbeat that is exactly `STALL_AFTER_S` old is late, not yet missing.

    Pinned because the comparison is the whole rule and both directions are defensible;
    what is not defensible is it changing silently.
    """
    assert camera_state(alive=True, since_last_beat_s=STALL_AFTER_S, since_start_s=600.0) is (
        CameraState.STREAMING
    )


# --- What a Heartbeat carries -----------------------------------------------


def test_a_heartbeat_is_frozen() -> None:
    """It crosses a process boundary, so it is picklable by construction and immutable.

    The `spawn` start method makes the first mandatory; the second is what stops the
    supervisor's copy being edited into disagreeing with what the worker sent.
    """
    beat = Heartbeat(last_frame_ts=FRAME_TS, effective_fps=2.5)

    with pytest.raises(AttributeError):
        beat.effective_fps = 3.0  # type: ignore[misc]


# --- The handle's grip on it ------------------------------------------------
#
# Real spawned workers, real queues. A heartbeat is a thing that crosses a process
# boundary, and a fake queue in this process would prove nothing about whether it can.


@pytest.fixture
def config() -> MusterConfig:
    parsed: dict[str, Any] = yaml.safe_load(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    return MusterConfig.model_validate(parsed)


@contextmanager
def running(handle: WorkerHandle) -> Iterator[WorkerHandle]:
    handle.start()
    try:
        yield handle
    finally:
        handle.stop(timeout=2.0)


def _handle(config: MusterConfig, entry: WorkerEntry) -> WorkerHandle:
    return WorkerHandle(CAMERA, config=config, entry=entry)


def _await_heartbeat(handle: WorkerHandle, timeout: float = 5.0) -> WorkerReport:
    """Drain until a heartbeat arrives, or fail saying it never did."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        handle.drain_heartbeats()
        report = handle.report()
        if report.last_frame_ts is not None:
            return report
        time.sleep(0.02)
    pytest.fail("the worker sent no heartbeat")


def test_a_handle_reports_what_the_worker_actually_sent(config: MusterConfig) -> None:
    with running(_handle(config, beat_then_idle)) as handle:
        report = _await_heartbeat(handle)

    assert report.last_frame_ts == SCRIPTED_FRAME_TS
    assert report.effective_fps == pytest.approx(SCRIPTED_FPS)


def test_a_worker_that_has_sent_nothing_reports_no_frame_time(config: MusterConfig) -> None:
    """The `null` P3.1 shipped is still the right answer before the first heartbeat."""
    with running(_handle(config, silent_until_stopped)) as handle:
        handle.drain_heartbeats()
        report = handle.report()

    assert report.last_frame_ts is None
    assert report.effective_fps is None
    assert report.state is CameraState.CONNECT


def test_draining_keeps_the_newest_of_several_queued_heartbeats(config: MusterConfig) -> None:
    """The supervisor ticks once a second and a worker beats faster than that.

    Keeping the first of a batch would report a reading that is already superseded, and
    would do it under exactly the load that makes several pile up.
    """
    with running(_handle(config, partial(beat_ramp_then_idle, beats=4))) as handle:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            handle.drain_heartbeats()
            if handle.report().effective_fps == pytest.approx(SCRIPTED_FPS + 3.0):
                break
            time.sleep(0.02)

        assert handle.report().effective_fps == pytest.approx(SCRIPTED_FPS + 3.0)


def test_a_restart_clears_the_dead_workers_heartbeat(config: MusterConfig) -> None:
    """Otherwise a crash-looping camera reads `STREAMING` the instant it respawns.

    The heartbeat belongs to a process, not to a camera. A new worker inheriting its
    predecessor's last value would report a pulse it never produced — and would do it
    fastest exactly when the camera is flapping, which is when the truth matters most.
    """
    handle = _handle(config, beat_then_idle)
    with running(handle):
        _await_heartbeat(handle)

    handle.start()
    try:
        report = handle.report()
    finally:
        handle.stop(timeout=2.0)

    assert report.last_frame_ts is None
    assert report.effective_fps is None


# --- What the worker actually sends -----------------------------------------


class _Beats:
    """A heartbeat sink that never rejects, so emission is what is under test."""

    def __init__(self) -> None:
        self.sent: list[Heartbeat] = []

    def put_nowait(self, item: Heartbeat) -> None:
        self.sent.append(item)


class _FullBeats:
    """A heartbeat sink that always rejects — the supervisor that stopped draining."""

    def put_nowait(self, _item: Heartbeat) -> None:
        raise queue.Full


def _admit(emitter: HeartbeatEmitter, clock: FakeMonotonic, *, frames: int, over_s: float) -> None:
    """Admit `frames` frames spread evenly across `over_s` seconds of the fake clock."""
    for index in range(frames):
        clock.advance(over_s / frames)
        emitter.note_admitted(FrameTs(FRAME_TS + timedelta(seconds=index)))


class FakeMonotonic:
    def __init__(self) -> None:
        self.now = 0.0

    def advance(self, seconds: float) -> None:
        self.now += seconds

    def __call__(self) -> float:
        return self.now


def test_a_worker_reports_the_fps_it_achieved_not_the_one_it_was_asked_for() -> None:
    """`Backpressure.target_fps` is the rate requested; this is the rate delivered.

    The two diverge exactly when something is wrong — a box that cannot keep up, a camera
    delivering fewer frames than it claims — which is the only time anybody reads this.
    Reporting the target would be the plausible-looking guess `/healthz` refuses on
    principle, and it would read as healthy precisely when it should not.
    """
    clock = FakeMonotonic()
    beats = _Beats()
    emitter = HeartbeatEmitter(beats, monotonic=clock)

    _admit(emitter, clock, frames=3, over_s=HEARTBEAT_INTERVAL_S)

    assert [beat.effective_fps for beat in beats.sent] == [
        pytest.approx(3.0 / HEARTBEAT_INTERVAL_S)
    ]


def test_a_worker_reports_the_capture_time_of_its_newest_admitted_frame() -> None:
    clock = FakeMonotonic()
    beats = _Beats()
    emitter = HeartbeatEmitter(beats, monotonic=clock)

    _admit(emitter, clock, frames=3, over_s=HEARTBEAT_INTERVAL_S)

    assert beats.sent[-1].last_frame_ts == FrameTs(FRAME_TS + timedelta(seconds=2))


def test_a_worker_that_admits_no_frames_reports_nothing() -> None:
    """THE signal, stated as an absence.

    A wedged RTSP connection blocks inside `source.frames()`, so no frame is admitted and
    no heartbeat is emitted. Emitting a zero-fps beat instead would keep the camera
    reading `STREAMING` forever with a number beside it saying it was doing nothing —
    which is a stall dressed as a measurement.
    """
    clock = FakeMonotonic()
    beats = _Beats()
    HeartbeatEmitter(beats, monotonic=clock)

    clock.advance(HEARTBEAT_INTERVAL_S * 100)

    assert beats.sent == []


def test_heartbeats_are_rate_limited_rather_than_sent_per_frame() -> None:
    """The queue holds four. A beat per admitted frame would overrun it every tick."""
    clock = FakeMonotonic()
    beats = _Beats()
    emitter = HeartbeatEmitter(beats, monotonic=clock)

    _admit(emitter, clock, frames=50, over_s=HEARTBEAT_INTERVAL_S * 2)

    assert len(beats.sent) == 2


def test_a_full_heartbeat_queue_does_not_kill_the_worker() -> None:
    """A supervisor that stopped draining must not turn into a dropped stream.

    Same rule as the snapshot reply: an unanswered status is a far better outcome than a
    camera that stops counting because nobody was listening to it.
    """
    clock = FakeMonotonic()
    emitter = HeartbeatEmitter(_FullBeats(), monotonic=clock)

    _admit(emitter, clock, frames=3, over_s=HEARTBEAT_INTERVAL_S)
