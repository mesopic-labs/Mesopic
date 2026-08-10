"""Validate the scorer against known-bad trackers, not just the honest one.

`make check` ran nothing against `tests/association` before this file existed (fix
round 1, Important 5) -- a scorer with no tests of its own is not evidence, it is an
unverified assumption wearing the shape of evidence. Each cheat below is a tracker that
is obviously wrong in one specific way; the scorer must catch that way and no other.
"""

from __future__ import annotations

from itertools import count

from muster.tracker.bytetrack import ByteTrackTracker, foot_point
from muster.types import DecodedFrame, Detection, PixelBox, Track, TrackId

from .score import score_run
from .walkers import HEIGHT, WIDTH, walk_scenario

Run = list[tuple[DecodedFrame, list[Detection], dict[int, PixelBox]]]
"""One `walk_scenario` run -- shared alias so the cheat trackers below stay readable."""


def _one_giant_track(run: Run) -> list[list[Track]]:
    """One identity, glued to whichever detection sorts first in the list each tick.

    The obviously-wrong behaviour a real tracker must never exhibit: every present
    person collapsed into a single published identity.
    """
    published: list[list[Track]] = []
    for frame, dets, _ in run:
        if not dets:
            published.append([])
            continue
        box, score = dets[0].box, dets[0].score
        published.append(
            [
                Track(
                    camera_id=frame.camera_id,
                    track_id=TrackId(1),
                    ts=frame.ts,
                    foot_point=foot_point(box, frame.width, frame.height),
                    score=score,
                    time_since_update=0,
                )
            ]
        )
    return published


def _new_track_each_tick(run: Run) -> list[list[Track]]:
    """A fresh, never-reused identity for every detection at every tick.

    The other obviously-wrong behaviour: correct instantaneous localisation, zero
    identity continuity at all.
    """
    ids = count(1)
    published: list[list[Track]] = []
    for frame, dets, _ in run:
        published.append(
            [
                Track(
                    camera_id=frame.camera_id,
                    track_id=TrackId(next(ids)),
                    ts=frame.ts,
                    foot_point=foot_point(d.box, frame.width, frame.height),
                    score=d.score,
                    time_since_update=0,
                )
                for d in dets
            ]
        )
    return published


def test_one_giant_track_scores_badly_on_merges_and_mostly_tracked() -> None:
    """A single identity absorbing several people must show up as merges, not IDSW.

    Clean input isn't enough to provoke this: with no dropped detections, "first in
    list order" is always the same walker (list order mirrors cast order), so the
    glued track never actually changes owner and merges stay zero -- it only shows
    badly on `mostly_tracked`. Recall corruption reorders who is first when an earlier
    walker's detection is dropped, which is what actually exercises the merge count.
    """
    run = walk_scenario("group", dt_s=0.5, seed=4, recall=0.7)
    truth = [gt for _, _, gt in run]
    score = score_run(_one_giant_track(run), truth, width=WIDTH, height=HEIGHT)
    assert score.merges > 0
    assert score.mostly_tracked < 0.5


def test_new_track_each_tick_scores_badly_on_id_switches() -> None:
    """Never reusing an identity must be caught as pervasive ID switching."""
    run = walk_scenario("group", dt_s=0.5, seed=1)
    truth = [gt for _, _, gt in run]
    score = score_run(_new_track_each_tick(run), truth, width=WIDTH, height=HEIGHT)
    # 3 walkers x 11 ticks after their first each = 33 switches; a loose bound keeps
    # this from being brittle against unrelated changes to the scenario's tick count.
    assert score.id_switches > 20
    assert score.merges == 0  # never-reused ids can never be reassigned


def test_honest_tracker_scores_perfectly_on_perfect_input() -> None:
    """The scorer's own easy case: no corruption, no ambiguity, one walker.

    If this does not read exactly IDSW=0 FRAG=0 MERGE=0 NEVER=0/1 MT=1.00, the bug is
    in the harness, not the tracker (task-8-brief.md Step 5).
    """
    run = walk_scenario("single", dt_s=0.2, seed=1)
    tracker = ByteTrackTracker(n_init=2)
    published = [tracker.update(frame, dets) for frame, dets, _ in run]
    truth = [gt for _, _, gt in run]
    score = score_run(published, truth, width=WIDTH, height=HEIGHT)
    assert score.id_switches == 0
    assert score.fragmentations == 0
    assert score.merges == 0
    assert score.never_confirmed == 0
    assert score.walkers == 1
    assert score.mostly_tracked == 1.0
