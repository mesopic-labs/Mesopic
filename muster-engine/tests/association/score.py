"""Association quality, scored against exact ground truth.

`never_confirmed` is the number that matters most and the one nothing else in the
project measures: ADR-0014's finding is that at low fps a large fraction of real people
never reach CONFIRMED, so they never reach geometry at all. That is a silent undercount
that looks exactly like ordinary detector recall loss, which is why it needs its own
metric rather than being folded into IDF1.

Matching is a single global one-to-one assignment per tick -- Hungarian, gated by a
radius, exactly the way `ByteTrackTracker` itself associates tracks to detections (fix
round 1). Matching each ground-truth walker to its independently-nearest track instead
lets two walkers claim the same track when they are close together, which is precisely
the `crossing` scenario's whole premise -- that independent-nearest bug manufactured
phantom ID switches out of a tie-break, not a real tracker mistake, and it would have
let an identity-collapsing tracker hide behind whichever walker its lone track happened
to be nearest that tick, gaming `never_confirmed` invisibly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from muster.types import PixelBox, Track

_MATCH_FRACTION_OF_HEIGHT = 0.04
"""How close a published foot-point must be to a true one, as a fraction of the true
walker's own box height, to count as that person -- the crowding floor.

Derived from `group`, the scenario built to be the hard case: at its original (now
superseded) spacing, three walkers half a metre apart in depth produced a true minimum
foot-point separation of ~43 px between the two nearest walkers at their closest
approach, reproducible directly from `walkers.py`'s own geometry. A match radius must
sit comfortably below half that separation (~21.5 px) or it cannot tell two crowded
walkers apart even when the tracker itself can -- a flat radius in pixel space, or one
scaled off a fixed reference depth, is 5-6x too generous to ever discriminate that
case. 0.04 gives ~14 px at 6 m: comfortably inside the ~21.5 px budget.

This floor alone is not sufficient across the depth range, though: at 12 m it gives
only ~7.8 px, tighter than the sweep's own detection jitter (sigma=5-8 px,
`walk_scenario`'s `box_sigma`), so a perfectly tracked far walker reads as lost purely
because the *detection* moved, not the track. See `_JITTER_MULTIPLE` for the other
floor the radius must also clear, and `_assert_radius_resolves_crowding` for the check
that the two floors haven't been asked to do something impossible in a given run.

`group`'s own spacing widened from that original `0.5m` (`walkers.py`'s
`_GROUP_DEPTH_SPACING_M`) so the scenario resolves at the sweep's upper jitter bound
(`box_sigma=8`); the ~43 px / ~0.5 m figures above describe the geometry this constant
was originally derived against, not the geometry `group` runs at now. `0.04` itself is
unchanged -- widening the spacing only gives it more headroom, never less.
"""

_JITTER_MULTIPLE = 3.0
"""How many detection-jitter standard deviations the match radius must clear -- the
noise floor.

A perfectly tracked walker at 12 m (`IDSW=0` the whole run) scores badly on
fragmentation and mostly-tracked under the sweep's own sigma=8 px corruption at the
crowding-only 0.04-of-height radius (~7.8 px there), because ordinary jitter routinely
exceeds a radius that tight (`tests/association/test_score.py`'s
`test_match_radius_tolerates_sweep_jitter_at_every_depth` pins this directly). Three
standard deviations covers ~99.7% of a Gaussian's mass, so genuine detection noise
almost never exceeds this floor by chance, while a radius this wide still cannot blur
two walkers together unless `group`'s own crowding floor also fails -- see
`_assert_radius_resolves_crowding`.
"""


@dataclass(frozen=True, slots=True)
class RunScore:
    """One (scenario, cost, fps, n_init) cell of the sweep."""

    id_switches: int
    fragmentations: int
    never_confirmed: int
    mostly_tracked: float
    merges: int = field(default=0)
    walkers: int = field(default=0)

    def as_row(self) -> str:
        """Render as a fixed-width line for a sweep report."""
        return (
            f"IDSW={self.id_switches:<3} FRAG={self.fragmentations:<3} "
            f"MERGE={self.merges:<3} NEVER={self.never_confirmed}/{self.walkers} "
            f"MT={self.mostly_tracked:.2f}"
        )


def _true_foot_point(box: PixelBox, width: int, height: int) -> tuple[float, float]:
    """Bottom-centre of `box`, normalized -- the same convention `foot_point` uses."""
    x1, _, x2, y2 = box
    return ((x1 + x2) / 2.0 / width, y2 / height)


def _match_radius(box: PixelBox, box_sigma: float) -> float:
    """The acceptance radius for `box`'s true owner: the looser of two floors.

    Must be wide enough to clear the run's own detection jitter (`_JITTER_MULTIPLE`) or
    a perfectly tracked walker reads as lost, and must stay narrower than half the
    minimum true separation between walkers currently present (checked separately, at
    scoring time, by `_assert_radius_resolves_crowding`) or it cannot tell two crowded
    people apart. Both floors are real and they pull in opposite directions across the
    depth range -- neither alone is sufficient.
    """
    _, y1, _, y2 = box
    height_floor = _MATCH_FRACTION_OF_HEIGHT * (y2 - y1)
    noise_floor = _JITTER_MULTIPLE * box_sigma
    return max(height_floor, noise_floor)


def _pixel_distance(
    a: tuple[float, float], b: tuple[float, float], width: int, height: int
) -> float:
    dx = (b[0] - a[0]) * width
    dy = (b[1] - a[1]) * height
    return float(np.hypot(dx, dy))


def _assert_radius_resolves_crowding(
    present: dict[int, PixelBox], radii: dict[int, float], width: int, height: int
) -> None:
    """Refuse to score a tick where the match radius cannot resolve two real walkers.

    The noise floor (`_JITTER_MULTIPLE`) and the crowding floor (`_MATCH_FRACTION_OF_
    HEIGHT`) are independent constraints pulling in opposite directions; nothing
    guarantees a run satisfies both just because each constant was individually
    reasonable. A radius wide enough to tolerate jitter but too wide to separate two
    close walkers would silently return a `never_confirmed`/`mostly_tracked` number
    that means nothing -- a merge the scorer cannot see is worse than a scorer that
    refuses to run.
    """
    ids = list(present)
    points = {w: _true_foot_point(present[w], width, height) for w in ids}
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            separation = _pixel_distance(points[a], points[b], width, height)
            radius = max(radii[a], radii[b])
            if radius >= separation / 2.0:
                msg = (
                    f"match radius {radius:.1f}px cannot resolve walkers {a} and {b}, "
                    f"separated by only {separation:.1f}px (need radius < "
                    f"{separation / 2.0:.1f}px) -- this scenario is too crowded for "
                    "the current noise floor; redesign the scenario's spacing or "
                    "lower box_sigma rather than trusting this run's numbers"
                )
                raise ValueError(msg)


def _match_tick(
    present: dict[int, PixelBox], observed: list[Track], width: int, height: int, box_sigma: float
) -> dict[int, int]:
    """One tick's walker -> track assignment: a single global one-to-one solve.

    Built the same way the tracker itself associates -- a distance matrix and
    `linear_sum_assignment` -- then reject any solved pair whose distance exceeds
    that walker's own radius. Rejecting after solving, rather than gating the matrix
    before solving, keeps a rejected row genuinely unmatched instead of letting scipy
    silently hand it a worse column.

    The crowding check runs even when there are no tracks to match against -- it is a
    property of the scenario's ground truth and this run's radius, not of the
    tracker's current output, so a tick with zero published tracks must not silently
    skip it.
    """
    if not present:
        return {}
    radii = {w: _match_radius(box, box_sigma) for w, box in present.items()}
    _assert_radius_resolves_crowding(present, radii, width, height)
    if not observed:
        return {}
    walker_ids = list(present)
    points = {w: _true_foot_point(present[w], width, height) for w in walker_ids}
    distances: NDArray[np.float64] = np.array(
        [
            [_pixel_distance(points[w], t.foot_point, width, height) for t in observed]
            for w in walker_ids
        ]
    )
    rows, cols = linear_sum_assignment(distances)
    matches: dict[int, int] = {}
    for row, col in zip(rows, cols, strict=True):
        if distances[row, col] < radii[walker_ids[row]]:
            matches[walker_ids[row]] = int(observed[col].track_id)
    return matches


def score_run(
    published: list[list[Track]],
    truth: list[dict[int, PixelBox]],
    *,
    width: int,
    height: int,
    box_sigma: float = 0.0,
) -> RunScore:
    """Compare what the tracker published against what was actually there.

    Args:
        published: One tick's live `Track` list per element, as returned by
            `ByteTrackTracker.update`.
        truth: One tick's `walker_id -> PixelBox` ground truth per element, aligned
            index-for-index with `published`.
        width: Frame width in pixels, for de-normalizing published foot-points.
        height: Frame height in pixels, for de-normalizing published foot-points.
        box_sigma: The same `box_sigma` the run was generated with
            (`walk_scenario`'s detection-jitter standard deviation, in pixels). Not
            inferred from the data -- passed through explicitly so the match radius's
            noise floor reflects the corruption this run actually used, not a guess.

    Returns:
        The scenario's `RunScore`.

    Raises:
        ValueError: If the match radius this `box_sigma` implies cannot resolve two
            walkers who are genuinely close together in this run -- see
            `_assert_radius_resolves_crowding`.
    """
    walker_track: dict[int, int] = {}  # walker_id -> the track_id it currently owns
    track_walker: dict[int, int] = {}  # track_id -> the walker_id it currently owns
    tracked_last: dict[int, bool] = {}  # walker_id -> was it matched last time present
    seen_ticks: dict[int, int] = {}  # walker_id -> ticks it was tracked at all
    total_ticks: dict[int, int] = {}  # walker_id -> ticks it was actually present
    switches = 0
    fragmentations = 0
    merges = 0

    for tracks, present in zip(published, truth, strict=True):
        observed = [t for t in tracks if t.time_since_update == 0]
        matches = _match_tick(present, observed, width, height, box_sigma)
        for walker_id in present:
            total_ticks[walker_id] = total_ticks.get(walker_id, 0) + 1
            track_id = matches.get(walker_id)
            if track_id is None:
                if tracked_last.get(walker_id, False):
                    fragmentations += 1
                tracked_last[walker_id] = False
                continue
            seen_ticks[walker_id] = seen_ticks.get(walker_id, 0) + 1
            tracked_last[walker_id] = True

            previous_track = walker_track.get(walker_id)
            if previous_track is not None and previous_track != track_id:
                switches += 1
            walker_track[walker_id] = track_id

            # A merge is the same failure as a switch, seen from the track's side: one
            # published identity absorbing a second real person is a distinct, silent
            # failure -- invisible to id_switches, which only ever looks from the
            # walker's side.
            previous_walker = track_walker.get(track_id)
            if previous_walker is not None and previous_walker != walker_id:
                merges += 1
            track_walker[track_id] = walker_id

    walkers = len(total_ticks)
    never = walkers - len(seen_ticks)
    mostly = sum(1 for w, ticks in total_ticks.items() if seen_ticks.get(w, 0) >= 0.8 * ticks)
    return RunScore(
        id_switches=switches,
        fragmentations=fragmentations,
        never_confirmed=never,
        mostly_tracked=mostly / walkers if walkers else 0.0,
        merges=merges,
        walkers=walkers,
    )
