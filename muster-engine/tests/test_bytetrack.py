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


def test_plain_iou_swaps_identities_where_the_default_cost_does_not() -> None:
    """ADR-0014's premise, end to end: same input, two costs, different identities.

    Originally this walked a single person and counted unique ids -- but that
    demonstration turns out to be structurally impossible once the gate widens enough
    to rescue a cold-start match at Muster's fps range (fix round 1, Important 1):
    `giou_cost` for a stationary (zero-velocity) prediction is `1 + (d-w)/(d+w)`, which
    exceeds 1.0 -- `iou_cost`'s own ceiling -- exactly when `d > w`, i.e. exactly
    ADR-0014's stated failure regime. Any `kappa` wide enough to admit a GIoU cold
    start there necessarily widens the gate past 1.0 too, so `iou_cost` stops ever
    being refused -- both costs converge to "always accepted" and the walk can no
    longer tell them apart by id count.

    So this tests the actual mechanism ADR-0014 names instead: "the Hungarian solver
    is handed a matrix of identical 1.0s and matches arbitrarily" (cost.py). Two people
    far enough apart that every track-detection pair has zero overlap -- a genuinely
    flat IoU matrix -- but at different true distances, so GIoU (which keeps
    decreasing past zero overlap) can still tell the correct pairing from the swap and
    IoU cannot. This isolates the assignment step from the gate entirely, so it holds
    for any kappa.
    """

    def run(cost: CostFunction) -> bool:
        tracker = ByteTrackTracker(n_init=1, cost=cost)
        born = tracker.update(
            _frame(0.0),
            [
                Detection(box=(100, 500, 140, 700), score=0.9),
                Detection(box=(350, 500, 390, 700), score=0.9),
            ],
        )
        id_near_100 = next(t.track_id for t in born if t.foot_point[0] * WIDTH < 300)
        id_near_350 = next(t.track_id for t in born if t.foot_point[0] * WIDTH >= 300)

        # Detections deliberately listed out of "natural" order: a detector's output
        # order carries no identity information, so the test must not accidentally
        # rely on it.
        tracks = tracker.update(
            _frame(0.5),
            [
                Detection(box=(400, 500, 440, 700), score=0.9),  # true continuation of 350
                Detection(box=(200, 500, 240, 700), score=0.9),  # true continuation of 100
            ],
        )
        id_at_200 = next(t.track_id for t in tracks if t.foot_point[0] * WIDTH < 300)
        id_at_400 = next(t.track_id for t in tracks if t.foot_point[0] * WIDTH >= 300)
        return id_at_200 == id_near_100 and id_at_400 == id_near_350

    assert run(iou_cost) is False, "a flat IoU matrix must not reliably avoid the swap"
    assert run(ByteTrackTracker().cost) is True, "GIoU must keep the correct identities"


def test_a_tentative_track_is_not_published() -> None:
    """Unconfirmed tracks must never reach geometry; they might be artifacts."""
    tracker = ByteTrackTracker(n_init=2)
    frame, dets = next(_walk(1, dt=0.5))
    assert tracker.update(frame, dets) == []


def test_a_marginal_detection_sustains_but_cannot_birth() -> None:
    """§3.2's third band: 0.55 is above track_thresh but below det_thresh.

    It sustains an existing track (stage 1's loose gate only cares about
    track_thresh) but can never start one on its own (birth requires det_thresh).
    An implementation that discarded every sub-det_thresh detection outright would
    pass the first half of this test but not the second.
    """
    tracker = ByteTrackTracker(n_init=1)
    box = (100, 500, 140, 700)

    assert tracker.update(_frame(0.0), [Detection(box=box, score=0.55)]) == []

    born = {t.track_id for t in tracker.update(_frame(0.5), [Detection(box=box, score=0.9)])}
    tracks = tracker.update(_frame(1.0), [Detection(box=box, score=0.55)])

    assert {t.track_id for t in tracks} == born
    assert tracks[0].time_since_update == 0


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


# --- Stage 2 (low-confidence recovery) ---------------------------------------------
#
# None of the tests above ever construct a detection below track_thresh, so max_cost_low
# and the whole recovery pass go completely unexercised: a stage 2 that did nothing at
# all would still pass every test above. These pin it down directly.


def test_a_low_confidence_detection_sustains_a_confirmed_track() -> None:
    """Stage 2 exists to keep a track alive through a momentary confidence dip."""
    tracker = ByteTrackTracker(n_init=1)
    box = (100, 500, 140, 700)
    born = {t.track_id for t in tracker.update(_frame(0.0), [Detection(box=box, score=0.9)])}

    tracks = tracker.update(_frame(0.5), [Detection(box=box, score=0.3)])

    assert {t.track_id for t in tracks} == born
    assert tracks[0].time_since_update == 0


def test_an_unmatched_low_confidence_detection_never_creates_a_track() -> None:
    """A sub-track_thresh detection may sustain an identity; it may never start one."""
    tracker = ByteTrackTracker(n_init=1)
    box = (100, 500, 140, 700)
    assert tracker.update(_frame(0.0), [Detection(box=box, score=0.3)]) == []
    assert tracker.update(_frame(0.5), [Detection(box=box, score=0.3)]) == []


def test_a_tentative_track_is_not_confirmed_by_a_low_confidence_match() -> None:
    """The stricter-birth-than-sustain rule must hold during recovery too.

    Stage 2 offering a low-confidence detection to an unproven track would let a
    single 0.9-score flicker plus a 0.3-score dip confirm a permanent identity --
    exactly the false-birth risk the module's own docstring says birth must avoid.
    """
    tracker = ByteTrackTracker(n_init=2)
    box = (100, 500, 140, 700)
    tracker.update(_frame(0.0), [Detection(box=box, score=0.9)])  # still TENTATIVE

    tracks = tracker.update(_frame(0.5), [Detection(box=box, score=0.3)])

    assert tracks == []


def test_a_far_low_confidence_detection_does_not_sustain() -> None:
    """max_cost_low must actually gate stage 2, not accept anything offered to it."""
    tracker = ByteTrackTracker(n_init=1)
    born = {
        t.track_id
        for t in tracker.update(_frame(0.0), [Detection(box=(100, 500, 140, 700), score=0.9)])
    }

    tracks = tracker.update(_frame(0.5), [Detection(box=(1200, 500, 1240, 700), score=0.3)])

    assert {t.track_id for t in tracks} == born, "the track must coast, not vanish"
    assert tracks[0].time_since_update == 1, "a distant low-confidence hit must not match"


# --- Out-of-order and identical timestamps -----------------------------------------


def test_an_out_of_order_frame_does_not_crash_or_corrupt_state() -> None:
    """A late frame (RTSP reconnect glitch) must degrade the tracker, not kill it."""
    tracker = ByteTrackTracker(n_init=1)
    tracker.update(_frame(1.0), [Detection(box=(100, 500, 140, 700), score=0.9)])

    tracker.update(_frame(0.5), [])  # arrives late; must not raise or rewind _last_ts

    tracks = tracker.update(_frame(1.5), [])
    assert tracks[0].time_since_update == 2, "the late frame counts as a miss, not a crash"


def test_identical_consecutive_timestamps_are_safe() -> None:
    """dt=0 must never be mistaken for the out-of-order case."""
    tracker = ByteTrackTracker(n_init=1)
    box = (100, 500, 140, 700)
    tracker.update(_frame(1.0), [Detection(box=box, score=0.9)])

    tracks = tracker.update(_frame(1.0), [Detection(box=box, score=0.9)])

    assert len(tracks) == 1
    assert tracks[0].time_since_update == 0
