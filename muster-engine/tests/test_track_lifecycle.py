"""Birth, confirmation, coasting and death.

The rule that earns its own test file is death-by-wall-clock. algorithms.md §3.4 wants
track memory to mean "two seconds of tolerance" regardless of fps; expressing it as a
frame count makes it mean 2 s at 5 fps and 10 s at 1 fps, which is wrong -- the person
has not been gone longer just because we sampled less often.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np

from muster.tracker.track import TrackRecord, TrackState
from muster.types import FrameTs, TrackId

BOX = np.array([100.0, 200.0, 140.0, 300.0])
T0 = FrameTs(datetime(2026, 8, 10, 12, 0, tzinfo=UTC))


def _at(seconds: float) -> FrameTs:
    return FrameTs(T0 + timedelta(seconds=seconds))


def _new(n_init: int = 2) -> TrackRecord:
    return TrackRecord(track_id=TrackId(1), box=BOX, score=0.9, ts=T0, n_init=n_init)


def test_a_new_track_starts_tentative() -> None:
    """A single confident box is not yet a person; it might be a flickering artifact."""
    assert _new().state is TrackState.TENTATIVE


def test_a_tentative_track_confirms_after_n_init_matches() -> None:
    track = _new(n_init=2)
    track.mark_matched(BOX, 0.9, _at(0.5))
    assert track.state is TrackState.CONFIRMED


def test_n_init_of_one_confirms_on_birth() -> None:
    """The low-fps trade ADR-0014 #4 puts on the table: frames are too precious to spend."""
    assert _new(n_init=1).state is TrackState.CONFIRMED


def test_a_partly_proven_track_is_still_not_a_person() -> None:
    """One match short of n_init must not confirm -- the off-by-one boundary.

    Both states are captured before either is asserted: narrowing `track.state` in an
    assertion would hide the mutation `mark_matched` performs from the type checker.
    """
    track = _new(n_init=3)

    track.mark_matched(BOX, 0.9, _at(0.5))
    after_one_match = track.state
    track.mark_matched(BOX, 0.9, _at(1.0))
    after_two_matches = track.state

    assert after_one_match is TrackState.TENTATIVE
    assert after_two_matches is TrackState.CONFIRMED


def test_an_unmatched_tentative_track_is_not_worth_keeping() -> None:
    track = _new(n_init=3)
    track.mark_missed()
    assert track.is_expired(_at(0.5), track_memory_s=2.0)


def test_a_confirmed_track_goes_lost_before_it_dies() -> None:
    track = _new(n_init=1)
    track.mark_missed()
    assert track.state is TrackState.LOST
    assert not track.is_expired(_at(0.5), track_memory_s=2.0)


def test_a_lost_track_revives_on_a_match() -> None:
    """The whole point of coasting: a brief occlusion must not cost the identity."""
    track = _new(n_init=1)
    track.mark_missed()
    track.mark_matched(BOX, 0.8, _at(0.5))
    assert track.state is TrackState.CONFIRMED
    assert track.time_since_update == 0


def test_track_memory_is_wall_clock_not_frames() -> None:
    """Identical frame counts, different fps -- and only the slow one has aged out."""
    fast, slow = _new(n_init=1), _new(n_init=1)
    for _ in range(3):
        fast.mark_missed()
        slow.mark_missed()
    assert not fast.is_expired(_at(0.6), track_memory_s=2.0)  # 3 ticks at 5 fps
    assert slow.is_expired(_at(3.0), track_memory_s=2.0)  # 3 ticks at 1 fps


def test_coasting_depth_is_counted_for_geometry() -> None:
    """engine §8 refuses to emit an event where both endpoints are unobserved."""
    track = _new(n_init=1)
    track.mark_missed()
    track.mark_missed()
    assert track.time_since_update == 2
