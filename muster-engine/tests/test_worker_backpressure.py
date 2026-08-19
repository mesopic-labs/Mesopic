"""What a worker does when the supervisor falls behind.

§9's rule is that load-shedding happens upstream, at the sampler: the answer to "the
supervisor is behind" is *sample fewer frames*, which means decode and infer less, which
is where the CPU is actually going. Dropping events instead would shed the cheap tail
work and keep paying for the expensive part.

Two units, both pure and both driven here without a process or a queue, because the
policy is arithmetic and only the plumbing needs a real queue (see `test_worker_handle`).

Red-first for P2.7.
"""

from __future__ import annotations

import queue

import pytest

from muster.supervisor.backpressure import Backpressure, Outbox


class FullSink:
    """A sink that refuses everything — the supervisor stalled on a slow disk."""

    def put_nowait(self, _item: object) -> None:
        raise queue.Full


class CountingSink:
    """A sink that accepts up to `capacity`, then refuses."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.items: list[object] = []

    def put_nowait(self, item: object) -> None:
        if len(self.items) >= self.capacity:
            raise queue.Full
        self.items.append(item)


class RecordingSampler:
    def __init__(self) -> None:
        self.rates: list[float] = []

    def set_target_fps(self, target_fps: float) -> None:
        self.rates.append(target_fps)


# --- Outbox -----------------------------------------------------------------


def test_pushed_events_reach_the_sink_in_order() -> None:
    sink = CountingSink(capacity=10)
    outbox: Outbox[int] = Outbox(maxlen=10)

    for index in range(3):
        outbox.push(index)
    outbox.flush(sink)

    assert sink.items == [0, 1, 2]
    assert outbox.pending == 0


def test_a_full_sink_leaves_events_pending_rather_than_losing_them() -> None:
    """A momentary stall must not cost counts; that is what the local buffer is for."""
    sink = CountingSink(capacity=2)
    outbox: Outbox[int] = Outbox(maxlen=10)

    for index in range(5):
        outbox.push(index)
    outbox.flush(sink)

    assert sink.items == [0, 1]
    assert outbox.pending == 3
    assert outbox.dropped == 0


def test_the_oldest_event_is_dropped_when_the_buffer_overflows() -> None:
    """§9 says drop the oldest. The newest events describe the scene as it is now."""
    outbox: Outbox[int] = Outbox(maxlen=3)

    for index in range(5):
        outbox.push(index)
    sink = CountingSink(capacity=10)
    outbox.flush(sink)

    assert sink.items == [2, 3, 4]
    assert outbox.dropped == 2


def test_a_drained_outbox_reports_no_pressure() -> None:
    sink = CountingSink(capacity=10)
    outbox: Outbox[int] = Outbox(maxlen=10)
    outbox.push(1)

    outbox.flush(sink)

    assert not outbox.under_pressure


def test_an_outbox_that_could_not_flush_reports_pressure() -> None:
    outbox: Outbox[int] = Outbox(maxlen=10)
    outbox.push(1)

    outbox.flush(FullSink())

    assert outbox.under_pressure


# --- Backpressure policy ----------------------------------------------------


def test_pressure_sheds_fps_immediately() -> None:
    """Shedding takes effect on the next frame, not a period later (P1.3)."""
    sampler = RecordingSampler()
    policy = Backpressure(sampler, fps_min=1.0, fps_max=5.0, start_fps=5.0)

    policy.observe(under_pressure=True)

    assert sampler.rates == [4.0]


def test_shedding_never_goes_below_the_configured_floor() -> None:
    """Below `fps_min` the engine is no longer measuring anything useful."""
    sampler = RecordingSampler()
    policy = Backpressure(sampler, fps_min=2.0, fps_max=5.0, start_fps=5.0)

    for _ in range(20):
        policy.observe(under_pressure=True)

    assert policy.target_fps == pytest.approx(2.0)
    assert min(sampler.rates) == pytest.approx(2.0)


def test_relief_must_be_sustained_before_fps_climbs_back() -> None:
    """Recovering on the first clear tick oscillates: shed, recover, shed, recover."""
    sampler = RecordingSampler()
    policy = Backpressure(sampler, fps_min=1.0, fps_max=5.0, start_fps=5.0)
    policy.observe(under_pressure=True)
    shed = policy.target_fps

    policy.observe(under_pressure=False)

    assert policy.target_fps == pytest.approx(shed)


def test_sustained_relief_climbs_back_toward_the_ceiling() -> None:
    sampler = RecordingSampler()
    policy = Backpressure(sampler, fps_min=1.0, fps_max=5.0, start_fps=5.0)
    policy.observe(under_pressure=True)

    for _ in range(Backpressure.RELIEF_TICKS * 10):
        policy.observe(under_pressure=False)

    assert policy.target_fps == pytest.approx(5.0)


def test_recovery_never_exceeds_the_configured_ceiling() -> None:
    sampler = RecordingSampler()
    policy = Backpressure(sampler, fps_min=1.0, fps_max=3.0, start_fps=3.0)

    for _ in range(50):
        policy.observe(under_pressure=False)

    assert max(sampler.rates, default=3.0) <= 3.0


def test_a_steady_state_does_not_touch_the_sampler() -> None:
    """A no-op call per frame that reconfigures the sampler is a per-frame allocation."""
    sampler = RecordingSampler()
    policy = Backpressure(sampler, fps_min=1.0, fps_max=5.0, start_fps=5.0)

    for _ in range(3):
        policy.observe(under_pressure=False)

    assert sampler.rates == []


# --- A new envelope arriving mid-run (P3.4) ---------------------------------


def test_a_new_envelope_lowers_a_target_that_is_now_above_it() -> None:
    """The budget an operator just saved has to bite now, not after the next shed.

    Without the clamp, `set_envelope` would only bound where recovery may climb *to*: a
    camera already running at 5 fps when the ceiling drops to 2 keeps running at 5 until
    something puts it under pressure, which on an idle box is never. The operator sees a
    saved config the engine is visibly ignoring.
    """
    sampler = RecordingSampler()
    policy = Backpressure(sampler, fps_min=1.0, fps_max=5.0, start_fps=5.0)

    policy.set_envelope(fps_min=1.0, fps_max=2.0)

    assert policy.target_fps == pytest.approx(2.0)
    assert sampler.rates == [2.0]


def test_a_new_envelope_raises_a_target_that_is_now_below_it() -> None:
    """The floor moving up is the same promise in the other direction."""
    sampler = RecordingSampler()
    policy = Backpressure(sampler, fps_min=1.0, fps_max=5.0, start_fps=5.0)
    for _ in range(20):
        policy.observe(under_pressure=True)

    policy.set_envelope(fps_min=4.0, fps_max=5.0)

    assert policy.target_fps == pytest.approx(4.0)


def test_a_new_envelope_leaves_a_target_that_still_fits_alone() -> None:
    """A budget save that did not move this camera must not disturb its shed state."""
    sampler = RecordingSampler()
    policy = Backpressure(sampler, fps_min=1.0, fps_max=5.0, start_fps=5.0)
    policy.observe(under_pressure=True)
    shed = policy.target_fps
    sampler.rates.clear()

    policy.set_envelope(fps_min=1.0, fps_max=5.0)

    assert policy.target_fps == pytest.approx(shed)
    assert sampler.rates == []


def test_recovery_after_a_new_envelope_stops_at_the_new_ceiling() -> None:
    """The clamp is not enough on its own: the bounds themselves have to move."""
    sampler = RecordingSampler()
    policy = Backpressure(sampler, fps_min=1.0, fps_max=5.0, start_fps=5.0)
    policy.set_envelope(fps_min=1.0, fps_max=2.0)

    for _ in range(Backpressure.RELIEF_TICKS * 10):
        policy.observe(under_pressure=False)

    assert policy.target_fps == pytest.approx(2.0)
