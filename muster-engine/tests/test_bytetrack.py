"""The association loop: confidence banding, the cost gate, and identity stability.

The gate test is a regression guard for a documented foot-gun. `max_cost` bounds the
association *cost* (`1 - similarity`), not the similarity. algorithms.md §3.3 records
that earlier drafts had it inverted, which would have shipped a tracker that
re-identifies almost every frame.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from muster.tracker.bytetrack import ByteTrackTracker
from muster.tracker.cost import CostFunction, iou_cost
from muster.types import CameraId, DecodedFrame, Detection, FrameTs, TrackId

WIDTH, HEIGHT = 1920, 1080
CAM = CameraId("cam-1")
T0 = datetime(2026, 8, 10, 12, 0, tzinfo=UTC)


def _frame(seconds: float) -> DecodedFrame:
    """A frame carrying no pixels: the tracker only ever reads ts, width and height."""
    return DecodedFrame(
        camera_id=CAM,
        ts=FrameTs(T0 + timedelta(seconds=seconds)),
        image=np.zeros((1, 1, 3), dtype=np.uint8),
        width=WIDTH,
        height=HEIGHT,
    )


def _walk(
    steps: int, *, dt: float, dx: int = 60, score: float = 0.9
) -> Iterator[tuple[DecodedFrame, list[Detection]]]:
    """A person crossing left to right, one detection per tick."""
    for i in range(steps):
        box = (100 + i * dx, 500, 140 + i * dx, 700)
        yield _frame(i * dt), [Detection(box=box, score=score)]


def test_a_person_walking_through_keeps_one_track_id() -> None:
    """P1.5's acceptance criterion, on synthetic ground truth."""
    tracker = ByteTrackTracker(n_init=2)
    seen: set[TrackId] = set()
    for frame, dets in _walk(10, dt=0.5):
        seen.update(t.track_id for t in tracker.update(frame, dets))
    assert len(seen) == 1


def test_plain_iou_fragments_where_the_default_cost_does_not() -> None:
    """ADR-0014's premise, end to end: same input, two costs, different identities."""

    def count_ids(cost: CostFunction) -> int:
        tracker = ByteTrackTracker(n_init=2, cost=cost)
        seen: set[TrackId] = set()
        for frame, dets in _walk(10, dt=0.5, dx=80):
            seen.update(t.track_id for t in tracker.update(frame, dets))
        return len(seen)

    assert count_ids(iou_cost) > count_ids(ByteTrackTracker().cost)


def test_a_tentative_track_is_not_published() -> None:
    """Unconfirmed tracks must never reach geometry; they might be artifacts."""
    tracker = ByteTrackTracker(n_init=2)
    frame, dets = next(_walk(1, dt=0.5))
    assert tracker.update(frame, dets) == []


def test_a_marginal_detection_sustains_but_cannot_birth() -> None:
    """§3.2's third band: 0.55 is above track_thresh but below det_thresh."""
    tracker = ByteTrackTracker(n_init=1)
    frame, _ = next(_walk(1, dt=0.5))
    assert tracker.update(frame, [Detection(box=(100, 500, 140, 700), score=0.55)]) == []


def test_a_confident_detection_births() -> None:
    tracker = ByteTrackTracker(n_init=1)
    frame, _ = next(_walk(1, dt=0.5))
    assert len(tracker.update(frame, [Detection(box=(100, 500, 140, 700), score=0.7)])) == 1


def test_max_cost_gates_the_cost_not_the_similarity() -> None:
    """The documented foot-gun. A tiny max_cost must be STRICT, not permissive."""
    strict = ByteTrackTracker(
        n_init=1,
        max_cost=0.01,
        max_cost_low=0.01,
        cost=iou_cost,
        gate_widening_per_second=0.0,
    )
    seen: set[TrackId] = set()
    for frame, dets in _walk(4, dt=0.5, dx=200):
        seen.update(t.track_id for t in strict.update(frame, dets))
    assert len(seen) == 4, "a near-zero cost gate must refuse distant matches"


def test_a_coasted_track_is_published_and_flagged() -> None:
    """algorithms §3.4: geometry needs path continuity AND the status to use it safely."""
    tracker = ByteTrackTracker(n_init=1)
    tracker.update(*next(_walk(1, dt=0.5)))
    tracks = tracker.update(_frame(0.5), [])
    assert len(tracks) == 1
    assert tracks[0].time_since_update == 1


def test_a_track_dies_after_track_memory_seconds() -> None:
    tracker = ByteTrackTracker(n_init=1, track_memory_s=1.0)
    tracker.update(*next(_walk(1, dt=0.5)))
    assert tracker.update(_frame(3.0), []) == []


def test_tracks_are_published_in_normalized_coordinates() -> None:
    """The pixel->normalized boundary, asserted at the only place it exists."""
    tracker = ByteTrackTracker(n_init=1)
    tracks = tracker.update(_frame(0.0), [Detection(box=(940, 340, 980, 540), score=0.9)])
    x, y = tracks[0].foot_point
    assert (x, y) == pytest.approx((960 / WIDTH, 540 / HEIGHT))


def test_two_people_keep_two_track_ids() -> None:
    tracker = ByteTrackTracker(n_init=1)
    seen: set[TrackId] = set()
    for i in range(6):
        dets = [
            Detection(box=(100 + i * 40, 500, 140 + i * 40, 700), score=0.9),
            Detection(box=(900 - i * 40, 500, 940 - i * 40, 700), score=0.9),
        ]
        seen.update(t.track_id for t in tracker.update(_frame(i * 0.5), dets))
    assert len(seen) == 2


def test_an_empty_frame_is_not_an_error() -> None:
    assert ByteTrackTracker().update(_frame(0.0), []) == []


def test_gate_widens_with_the_sampling_gap() -> None:
    """ADR-0014 Decision #1: accept iff cost < max_cost + kappa * dt_s.

    A match whose cost sits between max_cost and max_cost + kappa * dt_s must be
    accepted at a large dt and refused at a zero widening term -- otherwise kappa is
    decorative. The observable is track identity: a refused match births a new id, an
    accepted one keeps the old one.
    """
    tight = ByteTrackTracker(
        n_init=1, max_cost=0.01, max_cost_low=0.01, cost=iou_cost, gate_widening_per_second=0.0
    )
    widened = ByteTrackTracker(
        n_init=1, max_cost=0.01, max_cost_low=0.01, cost=iou_cost, gate_widening_per_second=5.0
    )
    frame0, dets0 = next(_walk(1, dt=0.5, dx=30))
    born_tight = {t.track_id for t in tight.update(frame0, dets0)}
    born_widened = {t.track_id for t in widened.update(frame0, dets0)}

    next_det = [Detection(box=(130, 500, 170, 700), score=0.9)]
    tight_ids = {t.track_id for t in tight.update(_frame(1.0), next_det)}
    widened_ids = {t.track_id for t in widened.update(_frame(1.0), next_det)}

    assert tight_ids != born_tight, "an ungated dt term must refuse this match"
    assert widened_ids == born_widened, "a widened gate must keep the same identity"
