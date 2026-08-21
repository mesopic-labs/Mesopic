"""Association cost functions — the pluggable seam ADR-0014 exists to create.

Plain IoU is flat at zero once two boxes stop overlapping, which is the regime Mesopic
runs in: at 2 fps a 1.5 m/s walker moves ~1.5 box-widths between sampled frames, so the
Hungarian solver is handed a matrix of identical 1.0s and matches arbitrarily. Every
function here takes `(x1, y1, x2, y2)` box arrays and returns a cost matrix where lower
is a better match; the candidates differ only in what they do once overlap hits zero.

Which one is the default is a measured choice, not an argued one — see ADR-0014 and
`tests/association/`.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

Boxes = NDArray[np.float64]
"""`(N, 4)` array of `(x1, y1, x2, y2)`."""

CostMatrix = NDArray[np.float64]
"""`(M, N)` array; lower is a better match."""

# `expansion_iou_cost` inflates each box by `_EXPANSION_PER_SECOND * dt_s / 2.0` of its
# OWN width/height, on every side (see `_expand`) -- a per-second fraction of the box's
# own size, not a physical speed in m/s despite numerically matching algorithms.md
# §3.3.1's 1.5 m/s reference walker. Unlike every constant in `kalman.py`, this one is
# not bridged through the metres->pixels scale (`height_px / _PERSON_HEIGHT_M`), so its
# actual inflation rate depends on the box's own aspect ratio -- for a person-shaped box
# it comes out to several box-widths per second, not 1.5 m/s of ground speed. See
# ADR-0014's benchmark result for `expansion_iou_cost`'s standing (it placed last); this
# scale question is an open caveat on that result, not one this comment resolves.
_EXPANSION_PER_SECOND = 1.5

# Weight on the width/height agreement term. Centre distance alone will happily match a
# far-away person to a near one; requiring similar apparent size breaks those ties.
_SCALE_PENALTY_WEIGHT = 0.5


class CostFunction(Protocol):
    """Scores every (track, detection) pair for the Hungarian assignment."""

    def __call__(self, tracks: Boxes, dets: Boxes, dt_s: float) -> CostMatrix:
        """Return the `(len(tracks), len(dets))` cost matrix.

        Args:
            tracks: Predicted track boxes for this tick.
            dets: Candidate detection boxes for this tick.
            dt_s: Seconds since the previous sampled frame. Costs that widen with the
                sampling gap use it; the others ignore it.
        """
        ...


def _pairwise_areas(tracks: Boxes, dets: Boxes) -> tuple[Boxes, Boxes, Boxes]:
    """Intersection, union, and smallest-enclosing-box areas for every pair."""
    lt = np.maximum(tracks[:, None, :2], dets[None, :, :2])
    rb = np.minimum(tracks[:, None, 2:], dets[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0.0, None), axis=2)

    area_t = np.prod(tracks[:, 2:] - tracks[:, :2], axis=1)
    area_d = np.prod(dets[:, 2:] - dets[:, :2], axis=1)
    union = area_t[:, None] + area_d[None, :] - inter

    enc_lt = np.minimum(tracks[:, None, :2], dets[None, :, :2])
    enc_rb = np.maximum(tracks[:, None, 2:], dets[None, :, 2:])
    enclosing = np.prod(np.clip(enc_rb - enc_lt, 0.0, None), axis=2)
    return inter, union, enclosing


def _iou(tracks: Boxes, dets: Boxes) -> CostMatrix:
    inter, union, _ = _pairwise_areas(tracks, dets)
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def iou_cost(tracks: Boxes, dets: Boxes, dt_s: float) -> CostMatrix:
    """`1 - IoU`, in `[0, 1]`. The control: what stock ByteTrack does, and what
    ADR-0014 found to be uninformative below ~3 fps."""
    del dt_s  # IoU has no notion of how long the gap was; that is the problem.
    return 1.0 - _iou(tracks, dets)


def giou_cost(tracks: Boxes, dets: Boxes, dt_s: float) -> CostMatrix:
    """`1 - GIoU`, in `[0, 2]`. Keeps decreasing as boxes separate, because the
    enclosing-box penalty grows without bound where IoU has already saturated."""
    del dt_s
    inter, union, enclosing = _pairwise_areas(tracks, dets)
    iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    penalty = np.divide(
        enclosing - union, enclosing, out=np.zeros_like(enclosing), where=enclosing > 0
    )
    return 1.0 - (iou - penalty)


def centre_distance_cost(tracks: Boxes, dets: Boxes, dt_s: float) -> CostMatrix:
    """Normalized centre separation plus a size-agreement penalty, in `[0, 2)`.

    The cheapest and most interpretable candidate, and the one the OC-SORT paper itself
    suggests (ADR-0014). Distance is scaled by the enclosing box's diagonal so it stays
    comparable across a frame where near people are large and far people are small --
    and, because both box centres always lie inside their own enclosing box, that
    normalized term is itself bounded by 1 (the enclosing box's diagonal is the longest
    distance any two points inside it can be apart). The scale-agreement term is a sum
    of two `|a-b|/(a+b)` ratios, each bounded by 1, weighted by `_SCALE_PENALTY_WEIGHT`,
    so it too contributes at most `2 * _SCALE_PENALTY_WEIGHT`. Total ceiling:
    `1.0 + 2.0 * _SCALE_PENALTY_WEIGHT` -- `COST_CEILING` derives it the exact same way,
    so the two can never silently drift apart -- a strict bound approached in the
    degenerate limit, not generally attained by any real box pair.
    """
    del dt_s
    centres_t = (tracks[:, :2] + tracks[:, 2:]) / 2.0
    centres_d = (dets[:, :2] + dets[:, 2:]) / 2.0
    separation = np.linalg.norm(centres_t[:, None, :] - centres_d[None, :, :], axis=2)

    enc_lt = np.minimum(tracks[:, None, :2], dets[None, :, :2])
    enc_rb = np.maximum(tracks[:, None, 2:], dets[None, :, 2:])
    diagonal = np.linalg.norm(enc_rb - enc_lt, axis=2)
    normalized = np.divide(separation, diagonal, out=np.zeros_like(separation), where=diagonal > 0)

    size_t = tracks[:, 2:] - tracks[:, :2]
    size_d = dets[:, 2:] - dets[:, :2]
    ratio = np.abs(size_t[:, None, :] - size_d[None, :, :]) / np.maximum(
        size_t[:, None, :] + size_d[None, :, :], 1e-9
    )
    # np.sum's return type isn't precise enough for mypy to see through the expression;
    # annotate the result explicitly rather than reach for a blanket ignore.
    result: CostMatrix = normalized + _SCALE_PENALTY_WEIGHT * np.sum(ratio, axis=2)
    return result


def expansion_iou_cost(tracks: Boxes, dets: Boxes, dt_s: float) -> CostMatrix:
    """IoU after inflating both boxes by a Δt-scaled margin.

    The honest form of "widen the gate off Δt": it widens the *boxes*, which is a
    quantity that is not already zero. Its virtue is that it becomes plain IoU as the
    gap shrinks, so it costs nothing where IoU already works.
    """
    margin = _EXPANSION_PER_SECOND * max(dt_s, 0.0) / 2.0
    return 1.0 - _iou(_expand(tracks, margin), _expand(dets, margin))


def _expand(boxes: Boxes, margin: float) -> Boxes:
    """Inflate each box by `margin` times its own width/height, on every side."""
    if margin <= 0.0:
        return boxes
    size = boxes[:, 2:] - boxes[:, :2]
    pad = size * margin
    return np.concatenate([boxes[:, :2] - pad, boxes[:, 2:] + pad], axis=1)


COSTS: dict[str, CostFunction] = {
    "iou": iou_cost,
    "giou": giou_cost,
    "centre_distance": centre_distance_cost,
    "expansion_iou": expansion_iou_cost,
}
"""Every candidate ADR-0014 shortlisted, keyed by the name the sweep reports."""

COST_CEILING: dict[str, float | None] = {
    "iou": 1.0,
    "giou": 2.0,
    "centre_distance": 1.0 + 2.0 * _SCALE_PENALTY_WEIGHT,
    "expansion_iou": 1.0,
}
"""Each cost's supremum, or `None` if it has none (ADR-0014's gate-form decision).

The additive gate `max_cost + kappa * dt` grows without bound while every bounded cost
saturates, so past some dt the gate exceeds the cost's own ceiling and refuses nothing
at all (`test_bytetrack.py`'s
`test_the_additive_gate_has_a_crossover_fps_below_which_it_refuses_nothing`). A
saturating gate needs to know what it must stay under.

`centre_distance_cost`'s entry is computed from `_SCALE_PENALTY_WEIGHT` rather than a
bare numeral, so a future change to that weight cannot silently invalidate this table --
see `centre_distance_cost`'s own docstring for why `1.0 + 2.0 * weight` is the true
bound: both terms it sums are self-normalizing, so the cost never exceeds it (a strict
bound, not generally attained by any real box pair), the same character as `giou_cost`'s
own ceiling.
"""


def ceiling_for(cost: CostFunction) -> float | None:
    """The registered ceiling for `cost`, by identity lookup against `COSTS`.

    Args:
        cost: One of `COSTS`'s values.

    Returns:
        The cost's supremum, or `None` if it has none.

    Raises:
        ValueError: If `cost` is not one of `COSTS` -- a saturating gate can only be
            calibrated against a cost this module knows the range of.
    """
    for name, fn in COSTS.items():
        if fn is cost:
            return COST_CEILING[name]
    msg = "cost function is not registered in COSTS; its ceiling is unknown"
    raise ValueError(msg)
