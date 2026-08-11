"""Association-cost properties — ADR-0014's central claim, asserted rather than cited.

The ADR rests on one fact: IoU is *flat at zero* once boxes separate, so the Hungarian
solver has no gradient to follow, and no amount of threshold tuning recovers a
similarity that is identically zero. Below ~3 fps a normally-walking adult is in that
regime every single frame. These tests pin that fact and pin the property each
candidate replacement must have to fix it.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from numpy.typing import NDArray

from muster.tracker.cost import (
    COST_CEILING,
    COSTS,
    CostFunction,
    centre_distance_cost,
    expansion_iou_cost,
    giou_cost,
    iou_cost,
)

DT = 0.5  # 2 fps, the middle of the operating range


def _box(x: float) -> NDArray[np.float64]:
    """A 40x100 box (a person-ish aspect) with its left edge at `x`."""
    return np.array([[x, 0.0, x + 40.0, 100.0]], dtype=np.float64)


SEPARATIONS = [60.0, 100.0, 200.0, 400.0, 800.0]


def test_iou_is_flat_at_zero_once_boxes_separate() -> None:
    """The ADR-0014 finding. If this ever fails, the ADR's premise changed."""
    costs = [float(iou_cost(_box(0.0), _box(d), DT)[0, 0]) for d in SEPARATIONS]
    assert costs == [1.0] * len(SEPARATIONS)


@pytest.mark.parametrize(
    "cost_fn", [giou_cost, centre_distance_cost], ids=["giou", "centre_distance"]
)
def test_candidate_costs_keep_increasing_past_zero_overlap(cost_fn: CostFunction) -> None:
    """Strictly monotone where IoU is flat -- the property that makes matching possible."""
    costs = [float(cost_fn(_box(0.0), _box(d), DT)[0, 0]) for d in SEPARATIONS]
    assert all(a < b for a, b in itertools.pairwise(costs)), costs


def test_scale_penalty_weight_is_pinned() -> None:
    """`_SCALE_PENALTY_WEIGHT` (cost.py) sets `centre_distance_cost`'s scale-agreement
    term and, through it, the cost's own ceiling (`COST_CEILING`). Same-centre boxes
    isolate the term: separation is exactly zero, so the whole cost is the scale
    penalty alone -- which is what lets this test catch a change to the weight that no
    other test in the suite would notice.
    """
    track = np.array([[0.0, 0.0, 40.0, 100.0]], dtype=np.float64)  # w=40, h=100
    det = np.array([[-20.0, -50.0, 60.0, 150.0]], dtype=np.float64)  # same centre, w=80, h=200
    cost = float(centre_distance_cost(track, det, DT)[0, 0])
    # ratio = |a-b|/(a+b) for width and height: |40-80|/120 = 1/3, |100-200|/300 = 1/3.
    # The literal 0.5 below is the shipped weight, not a re-import of the constant --
    # re-importing it would make this test tautological under the exact mutation
    # (0.5 -> 1.0) it exists to catch.
    assert cost == pytest.approx(0.5 * (1.0 / 3.0 + 1.0 / 3.0), abs=1e-9)


def test_centre_distance_ceiling_matches_its_derivation() -> None:
    """`COST_CEILING["centre_distance"]` is computed as `1.0 + 2.0 * _SCALE_PENALTY_
    WEIGHT` (cost.py) rather than a bare `2.0`, so the two can never silently drift
    apart. Pinning the current shipped value here too (not just re-deriving it from the
    same expression `cost.py` uses) means a change to that expression's shape -- not
    only its inputs -- is still visible as a test failure.
    """
    assert COST_CEILING["centre_distance"] == pytest.approx(2.0)


def test_overlapping_boxes_cost_less_than_separated_ones() -> None:
    """Sanity: the cost must still rank the obvious case the obvious way."""
    for cost_fn in COSTS.values():
        near = float(cost_fn(_box(0.0), _box(10.0), DT)[0, 0])
        far = float(cost_fn(_box(0.0), _box(400.0), DT)[0, 0])
        assert near < far, cost_fn


def test_expansion_iou_degrades_to_plain_iou_at_high_fps() -> None:
    """ADR-0014's stated reason to prefer it: it is a no-op where IoU already works."""
    tracks, dets = _box(0.0), _box(20.0)
    assert expansion_iou_cost(tracks, dets, 0.0) == pytest.approx(iou_cost(tracks, dets, 0.0))


def test_expansion_iou_recovers_a_gap_that_plain_iou_cannot() -> None:
    """At 1 fps the margin must actually bridge a walker's displacement."""
    tracks, dets = _box(0.0), _box(60.0)
    assert float(iou_cost(tracks, dets, 1.0)[0, 0]) == 1.0
    assert float(expansion_iou_cost(tracks, dets, 1.0)[0, 0]) < 1.0


def test_identical_boxes_cost_zero() -> None:
    for name, cost_fn in COSTS.items():
        assert float(cost_fn(_box(0.0), _box(0.0), DT)[0, 0]) == pytest.approx(0.0, abs=1e-9), name


def test_cost_matrix_has_one_entry_per_pair() -> None:
    """Shape contract the Hungarian solver depends on."""
    tracks = np.vstack([_box(0.0), _box(50.0), _box(100.0)])
    dets = np.vstack([_box(10.0), _box(60.0)])
    for name, cost_fn in COSTS.items():
        assert cost_fn(tracks, dets, DT).shape == (3, 2), name


def test_empty_inputs_produce_an_empty_matrix() -> None:
    """A frame with no detections must not raise; it must yield nothing to match."""
    empty = np.empty((0, 4), dtype=np.float64)
    for name, cost_fn in COSTS.items():
        assert cost_fn(_box(0.0), empty, DT).shape == (1, 0), name
        assert cost_fn(empty, _box(0.0), DT).shape == (0, 1), name
