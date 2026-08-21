"""Validate the scorer against known-bad trackers, not just the honest one.

A scorer with no tests of its own is not evidence, it is an unverified assumption
wearing the shape of evidence. Each cheat below is a tracker that is obviously wrong in
one specific way; the scorer must catch that way and no other.
"""

from __future__ import annotations

from datetime import timedelta
from itertools import count

import numpy as np

from mesopic.tracker.bytetrack import ByteTrackTracker, foot_point
from mesopic.types import DecodedFrame, Detection, FrameTs, PixelBox, Track, TrackId

from .score import score_run
from .walkers import _NO_PIXELS, CAM, HEIGHT, T0, WIDTH, _box_for, _jitter, _Walker, walk_scenario

Run = list[tuple[DecodedFrame, list[Detection], dict[int, PixelBox]]]
"""One `walk_scenario` run -- shared alias so the cheat trackers below stay readable."""


def _single_walker_run(z_m: float, *, dt_s: float, seed: int, box_sigma: float) -> Run:
    """A single straight-line walker at an arbitrary depth, corrupted the way
    `walk_scenario` corrupts `single`.

    `single` itself is fixed at z=6 m; this reaches the other depths (3 m, 12 m) needed
    to check the match radius's two floors across the sweep's depth range, reusing
    `walkers.py`'s own geometry and jitter so the numbers stay consistent with the rest
    of the harness.
    """
    rng = np.random.default_rng(seed)
    walker = _Walker(0, (-4.0, z_m), (1.0, 0.0), 0.0)
    run: Run = []
    for tick in range(30):
        t_s = tick * dt_s
        frame = DecodedFrame(
            camera_id=CAM,
            ts=FrameTs(T0 + timedelta(seconds=t_s)),
            image=_NO_PIXELS,
            width=WIDTH,
            height=HEIGHT,
        )
        box = _box_for(walker, t_s)
        if box is None:
            run.append((frame, [], {}))
            continue
        det = Detection(box=_jitter(box, box_sigma, rng), score=0.9)
        run.append((frame, [det], {0: box}))
    return run


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


def test_match_radius_tolerates_sweep_jitter_at_every_depth() -> None:
    """A perfectly-tracked walker must not read as lost purely from detection noise.

    The crowding-only 0.04-of-height radius gives ~7.8 px at 12 m, tighter than the
    sweep's own box_sigma=5-8 px corruption -- without a noise floor on top, a
    perfectly tracked far walker would score badly on fragmentation and mostly-tracked
    for no reason but jitter. The radius's noise floor (`_JITTER_MULTIPLE`) must clear
    that at every depth the sweep actually uses.
    """
    for z_m in (3.0, 6.0, 12.0):
        for sigma in (5.0, 8.0):
            run = _single_walker_run(z_m, dt_s=0.5, seed=1, box_sigma=sigma)
            tracker = ByteTrackTracker(n_init=2)
            published = [tracker.update(frame, dets) for frame, dets, _ in run]
            truth = [gt for _, _, gt in run]
            score = score_run(published, truth, width=WIDTH, height=HEIGHT, box_sigma=sigma)
            assert score.fragmentations == 0, f"z={z_m} sigma={sigma}"
            assert score.mostly_tracked == 1.0, f"z={z_m} sigma={sigma}"


def test_group_still_resolves_its_three_walkers_without_merging() -> None:
    """The noise floor must not have widened the radius past `group`'s own spacing.

    `group`'s three walkers are the scenario the crowding floor was derived to resolve;
    adding a noise floor on top must not undo that.
    """
    run = walk_scenario("group", dt_s=0.5, seed=1)
    tracker = ByteTrackTracker(n_init=2)
    published = [tracker.update(frame, dets) for frame, dets, _ in run]
    truth = [gt for _, _, gt in run]
    score = score_run(published, truth, width=WIDTH, height=HEIGHT)
    assert score.merges == 0
    assert score.id_switches == 0
    assert score.mostly_tracked == 1.0


def test_honest_tracker_scores_perfectly_on_perfect_input() -> None:
    """The scorer's own easy case: no corruption, no ambiguity, one walker.

    If this does not read exactly IDSW=0 FRAG=0 MERGE=0 NEVER=0/1 MT=1.00, the bug is
    in the harness, not the tracker.
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
