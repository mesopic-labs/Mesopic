"""Association quality, scored against exact ground truth.

`never_confirmed` is the number that matters most and the one nothing else in the
project measures: ADR-0014's finding is that at low fps a large fraction of real people
never reach CONFIRMED, so they never reach geometry at all. That is a silent undercount
that looks exactly like ordinary detector recall loss, which is why it needs its own
metric rather than being folded into IDF1.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from muster.types import PixelBox, Track

_MATCH_FRACTION_OF_HEIGHT = 0.35
"""How close a published foot-point must be to a true one, as a fraction of the true
walker's own box height, to count as that person.

A flat pixel budget is wrong on purpose here: box height falls off with depth by
design (task-8-report.md measures ~568 px at 3 m down to ~196 px at 12 m for this
rig), so a fixed threshold is generous for a near person and tight for a far one --
exactly backwards, since a near person's foot-point also moves more pixels per frame
for the same real-world speed. Scaling by the true box's own height keeps the match
tolerance proportional to the walker's apparent size instead. 0.35 is anchored at the
scenarios' reference depth (6 m, ~348 px tall): 0.35 * 348 ~= 122 px, matching the
flat 120 px this replaces at the distance the scenarios are actually built around, so
existing intuition about "how close counts" still holds there while now scaling
honestly for everyone else.
"""


@dataclass(frozen=True, slots=True)
class RunScore:
    """One (scenario, cost, fps, n_init) cell of the sweep."""

    id_switches: int
    fragmentations: int
    never_confirmed: int
    mostly_tracked: float
    walkers: int = field(default=0)

    def as_row(self) -> str:
        """Render as a fixed-width line for a sweep report."""
        return (
            f"IDSW={self.id_switches:<3} FRAG={self.fragmentations:<3} "
            f"NEVER={self.never_confirmed}/{self.walkers} MT={self.mostly_tracked:.2f}"
        )


def _true_foot_point(box: PixelBox, width: int, height: int) -> tuple[float, float]:
    """Bottom-centre of `box`, normalized -- the same convention `foot_point` uses."""
    x1, _, x2, y2 = box
    return ((x1 + x2) / 2.0 / width, y2 / height)


def score_run(
    published: list[list[Track]],
    truth: list[dict[int, PixelBox]],
    *,
    width: int,
    height: int,
) -> RunScore:
    """Compare what the tracker published against what was actually there.

    Args:
        published: One tick's live `Track` list per element, as returned by
            `ByteTrackTracker.update`.
        truth: One tick's `walker_id -> PixelBox` ground truth per element, aligned
            index-for-index with `published`.
        width: Frame width in pixels, for de-normalizing published foot-points.
        height: Frame height in pixels, for de-normalizing published foot-points.

    Returns:
        The scenario's `RunScore`.
    """
    assignments: dict[int, int] = {}  # walker_id -> the track_id it currently owns
    seen_ticks: dict[int, int] = {}  # walker_id -> ticks it was tracked at all
    total_ticks: dict[int, int] = {}  # walker_id -> ticks it was actually present
    switches = 0
    fragmentations = 0

    for tracks, present in zip(published, truth, strict=True):
        observed = [t for t in tracks if t.time_since_update == 0]
        for walker_id, box in present.items():
            total_ticks[walker_id] = total_ticks.get(walker_id, 0) + 1
            point = _true_foot_point(box, width, height)
            match = _nearest(observed, point, box, width, height)
            if match is None:
                if walker_id in assignments:
                    fragmentations += 1
                continue
            seen_ticks[walker_id] = seen_ticks.get(walker_id, 0) + 1
            previous = assignments.get(walker_id)
            if previous is not None and previous != match:
                switches += 1
            assignments[walker_id] = match

    walkers = len(total_ticks)
    never = walkers - len(seen_ticks)
    mostly = sum(1 for w, ticks in total_ticks.items() if seen_ticks.get(w, 0) >= 0.8 * ticks)
    return RunScore(
        id_switches=switches,
        fragmentations=fragmentations,
        never_confirmed=never,
        mostly_tracked=mostly / walkers if walkers else 0.0,
        walkers=walkers,
    )


def _nearest(
    tracks: list[Track],
    point: tuple[float, float],
    true_box: PixelBox,
    width: int,
    height: int,
) -> int | None:
    """The track whose foot-point is closest to `point`, if any is close enough.

    The acceptance radius scales with `true_box`'s own height rather than a flat
    pixel count -- see `_MATCH_FRACTION_OF_HEIGHT`.
    """
    _, y1, _, y2 = true_box
    max_distance = _MATCH_FRACTION_OF_HEIGHT * (y2 - y1)
    best_id: int | None = None
    best_distance = max_distance
    for track in tracks:
        dx = (track.foot_point[0] - point[0]) * width
        dy = (track.foot_point[1] - point[1]) * height
        distance = float(np.hypot(dx, dy))
        if distance < best_distance:
            best_distance, best_id = distance, int(track.track_id)
    return best_id
