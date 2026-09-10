"""Admission control, tested with a fake clock and no camera.

The sampler decides how much work the whole engine does, so its failure modes are
throughput failures rather than wrong answers: admit too few and metrics degrade, admit
too many and the N100 misses the M0 gate. The case worth the most tests is the one that
looks like neither — a reconnect delivering a burst of buffered frames, which a naive
catch-up gate turns into a detection storm precisely when the box is already behind.

Covers P1.3.
"""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime, timedelta

import pytest

from mesopic.sampler.sampler import FrameSampler
from mesopic.types import FrameTs

_T0 = FrameTs(datetime(2026, 8, 11, 12, 0, 0, tzinfo=UTC))


def _stream(start: FrameTs, fps: float, count: int) -> list[FrameTs]:
    """`count` capture timestamps at a steady `fps`, starting at `start`."""
    step = timedelta(seconds=1.0 / fps)
    return [start + step * i for i in range(count)]


def _admitted(sampler: FrameSampler, stream: list[FrameTs]) -> list[FrameTs]:
    return [ts for ts in stream if sampler.is_due(ts)]


def _per_second_counts(start: FrameTs, admitted: list[FrameTs]) -> list[int]:
    """Admissions bucketed into one-second windows of *capture* time."""
    counts: Counter[int] = Counter(int((ts - start).total_seconds()) for ts in admitted)
    return [counts[second] for second in range(max(counts, default=-1) + 1)]


def test_the_first_frame_is_always_admitted() -> None:
    """A fresh sampler has no baseline to measure against; refusing would stall startup."""
    assert FrameSampler(target_fps=3.0).is_due(_T0) is True


def test_a_25_fps_stream_is_admitted_at_the_target_rate() -> None:
    """The P1.3 acceptance criterion: 25 fps in, ~3 fps out, every second."""
    sampler = FrameSampler(target_fps=3.0)

    admitted = _admitted(sampler, _stream(_T0, fps=25.0, count=250))

    assert all(2 <= count <= 4 for count in _per_second_counts(_T0, admitted))


def test_frames_inside_the_period_are_rejected() -> None:
    """The gate is the point: everything between two due instants is dropped."""
    sampler = FrameSampler(target_fps=2.0)
    sampler.is_due(_T0)

    assert sampler.is_due(_T0 + timedelta(milliseconds=200)) is False
    assert sampler.is_due(_T0 + timedelta(milliseconds=499)) is False


def test_a_frame_exactly_on_the_due_instant_is_admitted() -> None:
    """A synthetic stream lands exactly on the boundary; `>` instead of `>=` halves it."""
    sampler = FrameSampler(target_fps=2.0)
    sampler.is_due(_T0)

    assert sampler.is_due(_T0 + timedelta(milliseconds=500)) is True


def test_a_late_frame_does_not_push_the_next_one_back() -> None:
    """Real cameras do not stamp frames on the due instant. Re-basing on each admitted
    frame charges its lateness to the next period, so every late frame costs rate."""
    sampler = FrameSampler(target_fps=2.0)
    sampler.is_due(_T0)
    sampler.is_due(_T0 + timedelta(milliseconds=600))

    assert sampler.is_due(_T0 + timedelta(milliseconds=1000)) is True


def test_a_16_fps_stream_averages_the_target_rather_than_a_divisor_of_it() -> None:
    """The M0 bench's case, seen on a real 16 fps camera. Waiting for the first frame
    past the period turns 400 ms into 437.5 ms on a 62.5 ms grid, so 2.5 fps becomes
    16/7 — the gate's floor then passes or fails on the camera's frame rate, not the box."""
    sampler = FrameSampler(target_fps=2.5)

    admitted = _admitted(sampler, _stream(_T0, fps=16.0, count=16 * 60))

    assert len(admitted) == pytest.approx(2.5 * 60, abs=1)


def test_a_late_frame_does_not_let_its_duplicate_through() -> None:
    """Cameras that stamp on a coarse clock send several frames with one timestamp. A
    due instant snapped *to* a late frame would admit its duplicate straight after it —
    the detector run twice on one instant."""
    sampler = FrameSampler(target_fps=2.5)
    sampler.is_due(_T0)
    late = _T0 + timedelta(milliseconds=900)

    assert sampler.is_due(late) is True
    assert sampler.is_due(late) is False


def test_the_first_frame_after_a_reconnect_gap_is_admitted() -> None:
    """Recovery: a camera that went away and came back must resume detecting at once."""
    sampler = FrameSampler(target_fps=3.0)
    sampler.is_due(_T0)

    assert sampler.is_due(_T0 + timedelta(seconds=60)) is True


def test_a_reconnect_burst_is_admitted_at_the_target_rate_not_all_at_once() -> None:
    """The storm case. A gate that advances by whole periods banks 60s of credit during
    the outage and then spends it on the buffered frames, at exactly the moment the box
    is least able to absorb it. Admission must follow capture time, not arrears."""
    sampler = FrameSampler(target_fps=3.0)
    _admitted(sampler, _stream(_T0, fps=25.0, count=50))

    resumed = _T0 + timedelta(seconds=60)
    burst = _admitted(sampler, _stream(resumed, fps=25.0, count=50))

    assert all(count <= 4 for count in _per_second_counts(resumed, burst))


def test_a_clock_rewind_re_arms_the_gate_instead_of_stalling_it() -> None:
    """A camera whose clock steps backwards would otherwise silence the sampler until
    real time caught up — minutes of a blind pipeline, reported as healthy."""
    sampler = FrameSampler(target_fps=3.0)
    sampler.is_due(_T0)

    assert sampler.is_due(_T0 - timedelta(seconds=30)) is True


def test_a_frame_slightly_out_of_order_is_still_rejected() -> None:
    """Re-arming on any backwards step would make jittered arrival a bypass."""
    sampler = FrameSampler(target_fps=1.0)
    sampler.is_due(_T0)

    assert sampler.is_due(_T0 - timedelta(milliseconds=50)) is False


def test_lowering_the_target_sheds_frames_from_the_next_one() -> None:
    """Backpressure is only useful if it bites immediately, not a period later."""
    sampler = FrameSampler(target_fps=10.0)
    sampler.is_due(_T0)

    sampler.set_target_fps(1.0)

    assert sampler.is_due(_T0 + timedelta(milliseconds=200)) is False
    assert sampler.is_due(_T0 + timedelta(seconds=1)) is True


def test_raising_the_target_admits_sooner() -> None:
    """The supervisor must be able to give budget back once the queue drains."""
    sampler = FrameSampler(target_fps=1.0)
    sampler.is_due(_T0)

    sampler.set_target_fps(10.0)

    assert sampler.is_due(_T0 + timedelta(milliseconds=100)) is True


@pytest.mark.parametrize("target_fps", [0.0, -1.0, float("nan"), float("inf")])
def test_construction_rejects_an_unusable_target(target_fps: float) -> None:
    """Zero would divide, negative would invert the gate, and neither is a rate."""
    with pytest.raises(ValueError, match="target_fps"):
        FrameSampler(target_fps=target_fps)


@pytest.mark.parametrize("target_fps", [0.0, -1.0, float("nan"), float("inf")])
def test_shedding_cannot_drive_the_target_to_an_unusable_value(target_fps: float) -> None:
    """Backpressure has a floor: shedding to zero is a stopped camera, not a slow one."""
    sampler = FrameSampler(target_fps=3.0)

    with pytest.raises(ValueError, match="target_fps"):
        sampler.set_target_fps(target_fps)


def test_a_naive_timestamp_is_rejected() -> None:
    """UTC everywhere. A naive datetime compares fine against another naive one, so this
    would otherwise surface as a timezone-shaped fps bug much further downstream."""
    sampler = FrameSampler(target_fps=3.0)

    with pytest.raises(ValueError, match="timezone-aware"):
        sampler.is_due(FrameTs(datetime(2026, 8, 11, 12, 0, 0)))  # noqa: DTZ001
