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
from muster.tracker.cost import CostFunction, ceiling_for, centre_distance_cost, giou_cost, iou_cost
from muster.tracker.kalman import _MAX_WALK_SPEED_MS, _PERSON_HEIGHT_M
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

    Walking a single person and counting unique ids cannot demonstrate this: once the
    gate widens enough to rescue a cold-start match at Muster's fps range, the same
    widening lets `iou_cost` and the default converge to "always accepted", and the
    walk can no longer tell them apart by id count.

    So this tests the actual mechanism ADR-0014 names instead: "the Hungarian solver
    is handed a matrix of identical 1.0s and matches arbitrarily" (cost.py). Two people
    far enough apart that every track-detection pair has zero overlap -- a genuinely
    flat IoU matrix -- but at different true distances, so a cost that keeps
    decreasing past zero overlap can still tell the correct pairing from the swap and
    plain IoU cannot.

    The gate is deliberately neutralized (`max_cost=1_000.0, gate_widening_per_second=
    0.0`) rather than left at either cost's shipped or tuned settings: ADR-0014's
    benchmark found that no single `(max_cost, kappa)` pair serves both `iou_cost` and
    whatever the default happens to be, since their cost ranges do not overlap --
    gating this test at either cost's own tuned point would sometimes refuse the very
    match being tested, for reasons that have nothing to do with the assignment step
    this test exists to isolate. A gate wide enough to never refuse anything removes
    the gate as a variable entirely, leaving only the assignment step under test.
    `len(tracks) == 2` below guards against a coasted leg or a spurious birth letting
    the id selectors silently pick the wrong track and pass for the wrong reason.
    """

    def run(cost: CostFunction) -> bool:
        tracker = ByteTrackTracker(
            n_init=1, cost=cost, max_cost=1_000.0, gate_widening_per_second=0.0
        )
        born = tracker.update(
            _frame(0.0),
            [
                Detection(box=(100, 500, 140, 700), score=0.9),
                Detection(box=(350, 500, 390, 700), score=0.9),
            ],
        )
        assert len(born) == 2
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
        # A track that coasted (gate refused) or a spurious birth would otherwise let
        # the selectors below silently pick the wrong track and pass for the wrong
        # reason -- this guard is what catches that.
        assert len(tracks) == 2, "every leg must actually match, not coast or spawn a third track"
        id_at_200 = next(t.track_id for t in tracks if t.foot_point[0] * WIDTH < 300)
        id_at_400 = next(t.track_id for t in tracks if t.foot_point[0] * WIDTH >= 300)
        return id_at_200 == id_near_100 and id_at_400 == id_near_350

    assert run(iou_cost) is False, "a flat IoU matrix must not reliably avoid the swap"
    assert run(ByteTrackTracker().cost) is True, "the shipped default must keep correct identities"


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


# --- Shipped class defaults ---------------------------------------------------------
#
# Every test above passes n_init/track_memory_s explicitly, so none of them would
# notice a silent change to the class defaults themselves. ADR-0014 Decision #4
# specifically measured n_init=1 and did not take it; a revert to n_init=1 -- or a
# drift in track_memory_s -- must not ship undetected.


def test_default_n_init_requires_two_hits_to_confirm() -> None:
    """The shipped default is `n_init=2` (ADR-0014 Decision #4: `n_init=1` was
    measured and rejected). A single detection must not confirm a track under a bare
    `ByteTrackTracker()`.
    """
    tracker = ByteTrackTracker()
    box = (100, 500, 140, 700)
    assert tracker.update(_frame(0.0), [Detection(box=box, score=0.9)]) == [], (
        "one hit must not confirm under the shipped n_init"
    )
    tracks = tracker.update(_frame(0.5), [Detection(box=box, score=0.9)])
    assert len(tracks) == 1, "the second hit must confirm under the shipped n_init"


def test_default_track_memory_matches_the_shipped_value() -> None:
    """`track_memory_s=2.0` is the shipped default; every other test in this file
    passes it explicitly, so a silent change to the class default would go unnoticed
    without a test constructing a bare `ByteTrackTracker()`.
    """
    tracker = ByteTrackTracker(n_init=1)
    tracker.update(_frame(0.0), [Detection(box=(100, 500, 140, 700), score=0.9)])

    assert tracker.update(_frame(1.9), []) != [], "must still be alive just under 2.0s"
    assert tracker.update(_frame(2.1), []) == [], "must have expired just over 2.0s"


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


def test_stage_2s_gate_is_tighter_than_stage_1s() -> None:
    """The two-stage design's defining property: stage 2 must be STRICTER, not just
    gated at all.

    Swapping `max_cost_low` for `max_cost` at the stage-2 call site leaves every other
    test in this file green, because none of them place a cost strictly between the two
    thresholds. This one does: at `gate_widening_per_second=1.5`, dt=0.5s, a detection
    100px from the track predicts a GIoU cost of ~1.4286 -- inside stage 1's widened
    gate (0.8 + 1.5*0.5 = 1.55, which "would accept") but outside stage 2's
    (0.5 + 1.5*0.5 = 1.25, which must refuse).

    Pinned against an EXPLICIT `cost=giou_cost` with its own `(max_cost, max_cost_low,
    kappa)` rather than the class defaults (the shipped default is `centre_distance_
    cost`'s own tuned point, a different scale entirely) -- this test is about the
    two-stage gate's relative strictness, a property of any cost/gate combination, not
    about whichever cost currently ships.
    """
    tracker = ByteTrackTracker(
        n_init=1, cost=giou_cost, max_cost=0.8, max_cost_low=0.5, gate_widening_per_second=1.5
    )
    born = {
        t.track_id
        for t in tracker.update(_frame(0.0), [Detection(box=(100, 500, 140, 700), score=0.9)])
    }

    tracks = tracker.update(_frame(0.5), [Detection(box=(200, 500, 240, 700), score=0.3)])

    assert {t.track_id for t in tracks} == born, "the track must coast, not vanish"
    assert tracks[0].time_since_update == 1, "stage 2's tighter gate must refuse this cost"


def test_stage_2_uses_the_shipped_max_cost_low() -> None:
    """The two-stage design's tighter stage-2 gate, pinned against the SHIPPED default
    specifically -- `test_stage_2s_gate_is_tighter_than_stage_1s` covers the same
    property against an explicit `giou_cost`, but nothing before this touched whatever
    `max_cost_low` actually ships.

    A detection 300px away at `dt=0.5s` costs `centre_distance_cost` ~1.120 (found by
    search over box sizes and offsets -- `centre_distance_cost` needs a size mismatch as
    well as a position offset to clear 1.0, since two same-size boxes at any distance
    apart give a strictly-below-1.0 cost, only the scale-agreement term can push it past
    that): inside stage 1's gate (`0.4 + 1.5*0.5 = 1.15`, "would accept") but outside
    stage 2's (`0.25 + 1.5*0.5 = 1.0`, must refuse).
    """
    tracker = ByteTrackTracker(n_init=1)
    born = {
        t.track_id
        for t in tracker.update(_frame(0.0), [Detection(box=(100, 500, 140, 700), score=0.9)])
    }

    tracks = tracker.update(_frame(0.5), [Detection(box=(400, 500, 440, 540), score=0.3)])

    assert {t.track_id for t in tracks} == born, "the track must coast, not vanish"
    assert tracks[0].time_since_update == 1, "the shipped stage-2 gate must refuse this cost"


def test_a_far_low_confidence_detection_does_not_sustain() -> None:
    """max_cost_low must actually gate stage 2, not accept anything offered to it.

    A same-size box offered at any distance can never trigger this under the shipped
    `centre_distance_cost` -- two equal-size boxes cap the cost strictly below 1.0
    however far apart (`cost.py`'s `COST_CEILING` docstring: the scale-agreement term
    needs an actual size mismatch, not just distance), so the box here is also a
    different size, not just far away, landing at cost ~1.66 -- comfortably past the
    shipped `max_cost_low` gate at this dt (`0.25 + 1.5*0.5 = 1.0`).
    """
    tracker = ByteTrackTracker(n_init=1)
    born = {
        t.track_id
        for t in tracker.update(_frame(0.0), [Detection(box=(100, 500, 140, 700), score=0.9)])
    }

    tracks = tracker.update(_frame(0.5), [Detection(box=(900, 500, 1700, 505), score=0.3)])

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


def test_a_stray_frame_never_rewinds_the_elapsed_time_clock() -> None:
    """`time_since_update` cannot stand in for this: it counts missed TICKS, a quantity
    invariant to how large dt was computed to be, so it would not notice a rewound
    `_last_ts` either way. This checks the tracker's own notion of elapsed time
    directly: after a late frame, `_last_ts` must still be the last WELL-ORDERED
    timestamp, and the next legitimate frame's gap must be the real one -- not
    inflated by however early the stray frame arrived.
    """
    tracker = ByteTrackTracker(n_init=1)
    tracker.update(_frame(1.0), [Detection(box=(100, 500, 140, 700), score=0.9)])

    tracker.update(_frame(0.5), [])  # arrives late; must not become the new "previous"

    last_ts_after_late_frame = tracker._last_ts
    assert last_ts_after_late_frame == _frame(1.0).ts, "a late frame must not rewind _last_ts"

    next_gap = tracker._elapsed(_frame(1.5).ts)
    assert next_gap == pytest.approx(0.5), "the next well-ordered gap must be the real one"


def test_a_stray_frame_never_rewinds_a_tracks_own_last_observed_ts() -> None:
    """`bytetrack.py`'s `_elapsed` guards the TRACKER's clock against a reordered frame
    (see the test above), but `TrackRecord.last_observed_ts` is separate per-track
    state, set by `mark_matched` -- a second place the same rewind can happen,
    independently of whether `_elapsed` is guarded.

    A track observed at t=0.0/0.5/1.0 (stationary, so every prediction is exact and
    every match trivially clears the shipped gate), then an RTSP reconnect glitch
    delivers a detection at t=0.2 -- a real match, not an empty frame, because
    `mark_matched` (the code path this guards) only runs on a match. Without the
    `max(ts, last_observed_ts)` guard, `last_observed_ts` would rewind 1.0 -> 0.2, so
    the NEXT tick's gate would widen off a bogus 1.3s gap (t=1.5 - 0.2) instead of the
    real 0.5s one (t=1.5 - 1.0): at the shipped default (`max_cost=0.4, kappa=1.5`)
    that is 2.35 versus 1.15 -- 2.35 exceeds `centre_distance_cost`'s own 2.0 ceiling
    (`cost.py`'s `COST_CEILING`), so the gate would refuse nothing at all, the maximum
    possible ID-swap exposure. The same rewind makes `is_expired` treat the track as
    0.8s staler than it actually is. Both are checked directly, not inferred from a
    knock-on symptom.
    """
    tracker = ByteTrackTracker(n_init=1)
    box = (100, 500, 140, 700)
    tracker.update(_frame(0.0), [Detection(box=box, score=0.9)])
    tracker.update(_frame(0.5), [Detection(box=box, score=0.9)])
    tracker.update(_frame(1.0), [Detection(box=box, score=0.9)])

    tracker.update(_frame(0.2), [Detection(box=box, score=0.9)])  # the reordered frame

    track = tracker._tracks[0]
    assert track.last_observed_ts == _frame(1.0).ts, (
        "a stray frame must not rewind last_observed_ts"
    )

    gate = tracker._gate([track], tracker._max_cost, _frame(1.5).ts)
    assert gate[0] == pytest.approx(1.15), "the gate must widen off the real 0.5s gap, not 1.3s"

    assert not track.is_expired(_frame(1.5).ts, track_memory_s=2.0), (
        "a track truly last observed 0.5s ago must not expire under a 2.0s memory"
    )


@pytest.mark.parametrize("fps", [1, 2, 3, 5])
def test_a_brisk_walker_is_admitted_at_every_supported_frame_rate(fps: int) -> None:
    """kappa's default must actually admit kalman.py's own walking-speed envelope,
    at every fps Muster supports -- not just whichever one someone currently believes
    is tightest.

    Which fps is tightest depends on both kappa and the shipped cost function's own
    shape, and either can change independently. Parametrizing over all four supported
    rates means no single point can be mislabelled "the worst case", and a future
    change that shifts which rate is tightest is still covered without anyone needing
    to notice or relabel it.

    Recomputed honestly using kalman.py's own formula
    (`scale = height_px / _PERSON_HEIGHT_M`) with this suite's box aspect ratio
    (width = 0.2 * height, `_walk`'s own convention): the resulting cold-start cost is
    invariant to the box's absolute size for a fixed aspect ratio (both the walker's
    pixel displacement and the box width scale linearly with height, so their ratio
    does not), so 200px is a concrete but arbitrary choice, not a tuned one. Importing
    `_MAX_WALK_SPEED_MS` and `_PERSON_HEIGHT_M` directly, rather than hardcoding
    numbers, means this test cannot silently drift from kalman.py's own definition of
    "brisk" again.

    The admission check alone is a weak guard: at the shipped defaults the true margin
    runs from roughly 0.48 (5 fps, the tightest) to roughly 1.21 (1 fps), so a walker
    several times faster than "brisk" would still be admitted, and a real regression
    that thinned the margin substantially could still pass. Pinning the actual margin
    against `centre_distance_cost` -- the shipped default -- against literal expected
    values is what makes this a regression guard rather than a smoke test: if kappa,
    max_cost, or the cost function's shape drifts, the pinned numbers catch it even
    when the walker is still, technically, admitted.
    """
    height, width = 200, 40
    scale = height / _PERSON_HEIGHT_M  # px/m, kalman.py's own metres-to-pixels anchor
    dt = 1.0 / fps
    displacement = round(_MAX_WALK_SPEED_MS * dt * scale)
    track_box = (100, 500, 100 + width, 500 + height)
    det_box = (100 + displacement, 500, 100 + width + displacement, 500 + height)

    tracker = ByteTrackTracker(n_init=1)
    tracker.update(_frame(0.0), [Detection(box=track_box, score=0.9)])
    tracks = tracker.update(_frame(dt), [Detection(box=det_box, score=0.9)])

    assert len(tracks) == 1
    assert tracks[0].time_since_update == 0, (
        f"the brisk walker's first step must be admitted at {fps} fps"
    )

    cost = float(
        centre_distance_cost(
            np.array([track_box], dtype=np.float64), np.array([det_box], dtype=np.float64), dt
        )[0, 0]
    )
    gate = tracker._max_cost + tracker._gate_widening_per_second * dt
    expected_margin = {1: 1.2089, 2: 0.6870, 3: 0.5641, 5: 0.4845}[fps]
    assert (gate - cost) == pytest.approx(expected_margin, abs=1e-3), (
        f"the shipped gate's margin at {fps} fps has drifted from its measured value"
    )


def test_the_additive_gate_has_a_crossover_fps_below_which_it_refuses_nothing() -> None:
    """A known limitation of ADR-0014 Decision #1's gate form, pinned rather than fixed.

    `gate = max_cost + kappa * dt` grows without bound as dt grows, but every cost this
    module ships is bounded (`cost.py`'s `COST_CEILING`). Once the gate exceeds a cost's
    ceiling, it refuses NOTHING -- at any separation, however absurd. The ADR did not
    consider that an additive gate can outgrow a bounded cost.

    Reads `max_cost`/`kappa`/the cost's ceiling live off a bare `ByteTrackTracker()`
    rather than a frozen snapshot of some other cost's numbers: the stated purpose of
    this test -- that a future kappa change cannot move the dead zone without a test
    noticing -- requires reading the actual shipped defaults, since `centre_distance_
    cost` (ADR-0014's chosen default) is bounded by a different ceiling than `giou_
    cost` would be.

    The crossover this pins is TIGHT: `max_cost=0.4, kappa=1.5` gives `dt=(2.0-0.4)/1.5
    ~= 1.067s`, i.e. **~0.9375 fps** -- just BELOW Muster's 1 fps product floor. At
    exactly 1 fps the gate (1.9) still sits under the ceiling (2.0), so it retains real
    discriminating power there, but with only ~5% headroom, not a wide margin.
    """
    reference = ByteTrackTracker()
    cost, max_cost, kappa = reference.cost, reference._max_cost, reference._gate_widening_per_second
    ceiling = ceiling_for(cost)
    assert ceiling is not None, "the shipped cost must have a finite ceiling"
    crossover_dt = (ceiling - max_cost) / kappa

    assert crossover_dt == pytest.approx(1.0667, abs=1e-3)
    assert 1.0 / crossover_dt == pytest.approx(0.9375, abs=1e-3)

    # A near-ceiling pair for the shipped cost (centre_distance_cost): two same-size
    # boxes can never exceed cost 1.0 however far apart (the scale-agreement term needs
    # a genuine size mismatch too, not just distance -- found by search, cost.py's
    # COST_CEILING docstring), so "absurd" here means both far apart AND wildly
    # mismatched in size, landing at ~1.945 -- close enough to the 2.0 ceiling to sit
    # above the 1fps gate (1.9) while nowhere near the 0.4-0.7-ish range a real walker's
    # cold-start cost actually occupies (see test_sweep.py's Phase A grid comment).
    origin = (1979, 2876, 2117, 2878)  # w=138 h=2
    far_mismatched = (-2954, -1708, -2951, -1273)  # w=3 h=435

    # n_init=1 explicitly, same as the other gate-mechanism tests in this file: a
    # refused SECOND association on a still-TENTATIVE (n_init=2) track dies outright
    # rather than coasting, which would confound birth-confirmation policy with the
    # gate discrimination this test isolates. cost/max_cost/kappa still come from the
    # live shipped tracker above, not re-guessed here.
    def _tracker() -> ByteTrackTracker:
        return ByteTrackTracker(
            n_init=1, cost=cost, max_cost=max_cost, gate_widening_per_second=kappa
        )

    # Below the crossover (a gap even the adaptive controller's lowest supported fps
    # does not produce): the gate exceeds the shipped cost's own ceiling, so this
    # near-ceiling pair is still admitted -- verified end to end.
    below = _tracker()
    below.update(_frame(0.0), [Detection(box=origin, score=0.9)])
    below_dt = crossover_dt + 1.0
    tracks = below.update(_frame(below_dt), [Detection(box=far_mismatched, score=0.9)])
    assert tracks[0].time_since_update == 0, "beyond the crossover the gate refuses nothing at all"

    # At the product's documented 1 fps floor: the crossover sits just below it, so the
    # gate must still discriminate there -- barely, but for real.
    above = _tracker()
    above.update(_frame(0.0), [Detection(box=origin, score=0.9)])
    tracks = above.update(_frame(1.0), [Detection(box=far_mismatched, score=0.9)])
    assert tracks[0].time_since_update == 1, "at the 1fps product floor the gate must still refuse"


def test_a_multi_tick_occlusion_reacquires_under_gap_based_widening_only() -> None:
    """Gap-based widening: the gate must widen by the gap since a track was last
    OBSERVED, not by the current tick's interval alone.

    A track misses three ticks (0.5s each), then a detection appears 300px away --
    reachable only because 1.5s has actually elapsed since the last real observation.
    Reverting to tick-based widening (using only the last tick's 0.5s) leaves this
    unreachable and the track never reacquires -- measured directly below, not
    hypothesised: mutating `_gate` to use `dt_s` instead of the per-track gap turns
    this from a reacquisition into a fresh birth.
    """
    tracker = ByteTrackTracker(n_init=1, gate_widening_per_second=1.5, track_memory_s=5.0)
    born = {
        t.track_id
        for t in tracker.update(_frame(0.0), [Detection(box=(100, 500, 140, 700), score=0.9)])
    }
    for missed_tick in (0.5, 1.0, 1.5):
        tracker.update(_frame(missed_tick), [])

    tracks = tracker.update(_frame(2.0), [Detection(box=(400, 500, 440, 700), score=0.9)])

    assert {t.track_id for t in tracks} == born, "gap-based widening must reacquire the track"
    assert tracks[0].time_since_update == 0
