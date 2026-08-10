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
from muster.tracker.cost import CostFunction, giou_cost, iou_cost
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
    flat IoU matrix -- but at different true distances, so a cost that keeps
    decreasing past zero overlap can still tell the correct pairing from the swap and
    plain IoU cannot.

    The gate is deliberately neutralized (`max_cost=1_000.0, gate_widening_per_second=
    0.0`) rather than left at either cost's shipped or tuned settings: Task 9's sweep
    (ADR-0014 Constraint 1) found that no single `(max_cost, kappa)` pair serves both
    `iou_cost` and whatever the default happens to be, since their cost ranges do not
    overlap -- gating this test at either cost's own tuned point would sometimes refuse
    the very match being tested, for reasons that have nothing to do with the
    assignment step this test exists to isolate (fix round 2's re-review hit exactly
    this failure mode once already, when a too-tight gate coasted a leg and a silent
    `next(...)` selector picked the wrong track and returned `True` by accident --
    `len(tracks) == 2` below is the guard against that). A gate wide enough to never
    refuse anything removes the gate as a variable entirely, leaving only the
    assignment step under test.
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
        # reason -- exactly the accident the re-review found at kappa=1.0.
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
    test in this file green (measured -- see fix round 2 in task-7-report.md), because
    none of them place a cost strictly between the two thresholds. This one does: at
    `gate_widening_per_second=1.5`, dt=0.5s, a detection 100px from the track predicts
    a GIoU cost of ~1.4286 -- inside stage 1's widened gate (0.8 + 1.5*0.5 = 1.55, which
    "would accept") but outside stage 2's (0.5 + 1.5*0.5 = 1.25, which must refuse).

    Pinned against an EXPLICIT `cost=giou_cost` with its own historical `(max_cost,
    max_cost_low, kappa)` rather than the class defaults (Task 9 changed those to
    `centre_distance_cost`'s own tuned point, a different scale entirely) -- this test
    is about the two-stage gate's relative strictness, a property of any cost/gate
    combination, not about whichever cost currently ships.
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


def test_a_stray_frame_never_rewinds_the_elapsed_time_clock() -> None:
    """The actual corruption fix round 1's Important 4 named.

    `time_since_update` cannot catch a rewound `_last_ts`: it counts missed TICKS, a
    quantity invariant to how large dt was computed to be, so the round 1 report's
    claim that it would detect a rewind was wrong. This checks the tracker's own
    notion of elapsed time directly: after a late frame, `_last_ts` must still be the
    last WELL-ORDERED timestamp, and the next legitimate frame's gap must be the real
    one -- not inflated by however early the stray frame arrived.
    """
    tracker = ByteTrackTracker(n_init=1)
    tracker.update(_frame(1.0), [Detection(box=(100, 500, 140, 700), score=0.9)])

    tracker.update(_frame(0.5), [])  # arrives late; must not become the new "previous"

    last_ts_after_late_frame = tracker._last_ts
    assert last_ts_after_late_frame == _frame(1.0).ts, "a late frame must not rewind _last_ts"

    next_gap = tracker._elapsed(_frame(1.5).ts)
    assert next_gap == pytest.approx(0.5), "the next well-ordered gap must be the real one"


@pytest.mark.parametrize("fps", [1, 2, 3, 5])
def test_a_brisk_walker_is_admitted_at_every_supported_frame_rate(fps: int) -> None:
    """kappa's default must actually admit kalman.py's own walking-speed envelope,
    at every fps Muster supports -- not just whichever one someone currently believes
    is tightest.

    Fix round 1 claimed 3 fps was the worst case, at a +0.157 margin, computed against
    a pixel scale that was never tied to kalman.py's own metres-to-pixels derivation --
    that number was wrong twice over: the scale was wrong, AND 3 fps was not even the
    tightest point once corrected (fix round 2 corrected the scale and found 3 fps
    refused at kappa=1.5; fix round 3's re-review found 5 fps is actually tighter than
    3 fps at the corrected kappa=2.0: margins are +0.1422 at 3 fps vs +0.1189 at 5 fps).
    Naming a single fps "the worst case" invites exactly this kind of silent drift when
    kappa or the cost function changes. Parametrizing over all four rates means no
    single point can be mislabelled, and a future kappa change that shifts which rate
    is tightest is still covered without anyone needing to notice or relabel it.

    Recomputed honestly using kalman.py's own formula
    (`scale = height_px / _PERSON_HEIGHT_M`) with this suite's box aspect ratio
    (width = 0.2 * height, `_walk`'s own convention): the resulting cold-start GIoU
    cost is invariant to the box's absolute size for a fixed aspect ratio (both the
    walker's pixel displacement and the box width scale linearly with height, so their
    ratio does not), so 200px is a concrete but arbitrary choice, not a tuned one.
    Importing `_MAX_WALK_SPEED_MS` and `_PERSON_HEIGHT_M` directly, rather than
    hardcoding numbers, means this test cannot silently drift from kalman.py's own
    definition of "brisk" again.
    """
    height, width = 200, 40
    scale = height / _PERSON_HEIGHT_M  # px/m, kalman.py's own metres-to-pixels anchor
    dt = 1.0 / fps
    displacement = round(_MAX_WALK_SPEED_MS * dt * scale)

    tracker = ByteTrackTracker(n_init=1)
    tracker.update(_frame(0.0), [Detection(box=(100, 500, 100 + width, 500 + height), score=0.9)])
    tracks = tracker.update(
        _frame(dt),
        [
            Detection(
                box=(
                    100 + displacement,
                    500,
                    100 + width + displacement,
                    500 + height,
                ),
                score=0.9,
            )
        ],
    )

    assert len(tracks) == 1
    assert tracks[0].time_since_update == 0, (
        f"the brisk walker's first step must be admitted at {fps} fps"
    )


def test_the_additive_gate_stops_refusing_below_about_1_7_fps() -> None:
    """A known limitation of ADR-0014 Decision #1's gate form, pinned rather than fixed.

    `gate = max_cost + kappa * dt` grows without bound as dt grows, but a BOUNDED cost
    (e.g. `giou_cost`, bounded above by 2.0: cost.py, `1 - GIoU`, GIoU in `[-1, 1]`) is
    not. Once the gate exceeds that ceiling, it refuses NOTHING -- at any separation,
    however absurd. The ADR did not consider that an additive gate can outgrow a bounded
    cost.

    Task 9's sweep settled this as a real but non-blocking property: `gate_form` is now
    a swept axis (a `saturating` alternative exists precisely because of this dead
    zone -- `cost.py`'s `COST_CEILING`, `bytetrack.py`'s `_gate`), but the winning
    candidate, `centre_distance_cost`, has NO ceiling (unbounded by construction), so it
    cannot have this dead zone at all under the additive form -- there is nothing for
    the gate to outgrow. This test therefore pins the dead zone against an EXPLICIT
    bounded cost (`giou_cost`, its own historical `max_cost=0.8, kappa=2.0`) rather than
    the shipped default, since the property being tested is "an additive gate paired
    with a bounded cost", not "whatever ships".

    The crossover (gate == giou_cost's ceiling) is `dt = (2.0 - max_cost) / kappa`. At
    `max_cost=0.8, kappa=2.0` that is dt=0.6s, ~1.667 fps. Muster's product-documented
    1 fps floor -- and the rate the adaptive controller lands on under load, i.e. exactly
    when a busy scene makes swap risk highest -- sits inside this dead zone, for any
    bounded cost run under the additive form at these settings.
    """
    max_cost, kappa = 0.8, 2.0
    giou_ceiling = 2.0  # cost.py: `1 - GIoU`, GIoU in [-1, 1] -> cost in [0, 2]
    crossover_dt = (giou_ceiling - max_cost) / kappa

    assert crossover_dt == pytest.approx(0.6)
    assert 1.0 / crossover_dt == pytest.approx(1.6667, abs=1e-3)

    # Below the crossover (1 fps: the product floor, and where the controller lands
    # under load): the gate exceeds giou_cost's own ceiling, so an absurd jump that no
    # real cost value could ever clear is still admitted -- verified end to end.
    below = ByteTrackTracker(
        n_init=1, cost=giou_cost, max_cost=max_cost, gate_widening_per_second=kappa
    )
    below.update(_frame(0.0), [Detection(box=(100, 500, 140, 700), score=0.9)])
    tracks = below.update(_frame(1.0), [Detection(box=(100_000, 500, 100_040, 700), score=0.9)])
    assert tracks[0].time_since_update == 0, "below ~1.67 fps the gate refuses nothing at all"

    # Above the crossover (2 fps): the gate is still bounded below the ceiling, so the
    # same absurd jump is correctly refused -- the gate has real discriminating power.
    above = ByteTrackTracker(
        n_init=1, cost=giou_cost, max_cost=max_cost, gate_widening_per_second=kappa
    )
    above.update(_frame(0.0), [Detection(box=(100, 500, 140, 700), score=0.9)])
    tracks = above.update(_frame(0.5), [Detection(box=(100_000, 500, 100_040, 700), score=0.9)])
    assert tracks[0].time_since_update == 1, "above ~1.67 fps the same jump must still be refused"


def test_a_multi_tick_occlusion_reacquires_under_gap_based_widening_only() -> None:
    """Gap-based widening (fix round 1, folded-in finding): the gate must widen by the
    gap since a track was last OBSERVED, not by the current tick's interval alone.

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
