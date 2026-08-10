"""The ADR-0014 sweep, and the regression guard that keeps its answer honest.

The sweep is marked `slow` -- it is a study, run when the decision is being made or
revisited, not on every commit. `test_the_chosen_cost_still_beats_a_tuned_iou_baseline`
is the part that runs always: it is what stops the measured decision from silently
rotting when someone tunes a threshold two years from now.

Structured in two phases (task-9-report.md has both tables in full):

* **Phase A** tunes each (cost function, gate form) pair's own `(max_cost, kappa)`.
  Constraint 1 (task-9-brief.md's six carried-forward constraints): the candidates'
  cost ranges do not overlap (`iou_cost`/`expansion_iou_cost` in `[0, 1]`, `giou_cost`/
  `centre_distance_cost` in `[0, 2)`), so one fixed gate cannot serve all of them --
  sweeping candidates against a shared gate would measure gate calibration, not cost
  quality. Each candidate is tuned at its own best operating point instead. Tuned
  jointly across `n_init in {1, 2, 3}` (Fix round 1, Important 6) rather than at a
  single fixed `n_init`, which the first pass got wrong: at `n_init=2` alone every
  `centre_distance_cost` cell tied at `never-confirmed=0`, so selection silently fell
  through to the simplicity tiebreak (smallest kappa) on a metric that was not actually
  discriminating -- aggregating over all three `n_init` breaks that degenerate tie
  honestly.
* **Phase B** compares the tuned candidates across fps x n_init x scenario x seed.

Both phases run under detection corruption only (Constraint 2): on clean input all four
costs tie, because exact constant-velocity motion makes the Kalman prediction exact and
leaves nothing for any cost to disambiguate -- an all-tie result is evidence the harness
does not fabricate discrimination, not a ranking result. `test_clean_input_is_a_tie`
keeps that sanity check in the suite without letting it leak into the decision.

Runtime: a single `_run` call is ~1.3ms (measured). At the seed counts below (Fix round
1 raised these for significance -- Important 3 -- and widened Phase A's tuning to the
full `n_init` grid -- Important 6), the full file (Phase A + Phase B +
`test_significance_of_the_ranking` + the clean-input check) measures **180s (3 minutes),
measured directly** (`pytest -m slow -s`, wall clock). Comfortably under the 15 minute
budget; nothing was cut to get there.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from functools import cache
from typing import Literal

import pytest
from scipy.stats import ttest_rel

from muster.tracker.bytetrack import ByteTrackTracker
from muster.tracker.cost import COSTS, iou_cost

from .score import RunScore, score_run
from .walkers import HEIGHT, SCENARIOS, WIDTH, walk_scenario

GateForm = Literal["additive", "saturating"]

FPS_GRID = (5.0, 3.0, 2.0, 1.0)
N_INIT_GRID = (1, 2, 3)
SEEDS = tuple(range(1, 41))
"""40, not the original 5 (Fix round 1, Important 3): at 5 seeds the full-grid ranking
between the top two candidates was not statistically significant (paired t, p~0.20) --
a real property of the noise floor, not a mistake, but one the report must not paper
over with a ranking stated as fact. `test_significance_of_the_ranking` computes and
prints the actual statistic at this seed count; see task-9-report.md's Fix round 1 for
the count-vs-significance table that justified stopping at 40 rather than going higher
or lower."""

GUARD_SEEDS = (1, 2)
"""The regression guard runs on the fast path, so it takes a subset. If the margin it
asserts is not visible over two seeds, the margin is too thin to defend anyway."""

TUNE_SEEDS = tuple(range(1, 11))
"""Phase A tunes on a subset of `SEEDS` -- separates the candidates cleanly at 10 while
staying well short of Phase B's full 40, so Phase B's own numbers are not just
re-reporting what Phase A already saw with more seeds."""

# The corrupted operating point every decision number in this file comes from
# (Constraint 2). `box_sigma` is deliberately the sweep's own UPPER jitter bound, not a
# gentler value -- Constraint 3 requires `group`, the one scenario built to stress
# crowding, to run at that bound rather than being spared it via a lower sigma.
RECALL = 0.85
BOX_SIGMA = 8.0
FALSE_POSITIVE_RATE = 0.08
CORRUPTED = {"recall": RECALL, "box_sigma": BOX_SIGMA, "false_positive_rate": FALSE_POSITIVE_RATE}
CLEAN = {"recall": 1.0, "box_sigma": 0.0, "false_positive_rate": 0.0}

# Stage 2 (low-confidence recovery) must stay STRICTER than stage 1 (bytetrack.py's own
# invariant, pinned by test_stage_2s_gate_is_tighter_than_stage_1s) -- preserved across
# the sweep by keeping a fixed ratio to whatever max_cost Phase A is trying, rather than
# treating it as a second free axis the brief never asked Task 9 to tune.
_MAX_COST_LOW_RATIO = 0.625  # bytetrack.py's own shipped ratio: 0.5 / 0.8

# Per-(cost, gate form) coarse grids (Constraint 1: each candidate's own scale, not a
# shared one). Ranges anchored on the cold-start costs a brisk (kalman.py's
# `_MAX_WALK_SPEED_MS`) walker's first post-birth match produces at 1-5 fps -- measured
# directly (task-9-report.md): iou/expansion_iou saturate at their 1.0 ceiling by 2 fps,
# giou ranges ~1.08-1.71, centre_distance ~0.22-0.69. Kappa's saturating-form range runs
# higher than its additive-form range because the saturating gate MULTIPLIES its
# widening by `(ceiling - max_cost)` (see bytetrack.py's `_gate`: `ceiling - (ceiling -
# max_cost) * exp(-kappa * gap)`), so an additive-scaled kappa would barely move it.
#
# `centre_distance_cost` has a saturating grid too (Fix round 1, Critical 2): it was
# wrongly treated as unbounded in the first pass -- both its terms are self-normalizing
# and it is in fact bounded by 2.0, the same ceiling as `giou_cost` (`cost.py`'s
# `COST_CEILING` docstring has the measurement) -- so a saturating gate is well-defined
# for it, and the omission meant Constraint 4 was never actually checked for the
# candidate the sweep went on to choose.
GATE_FORMS_BY_COST: dict[str, tuple[GateForm, ...]] = {
    "iou": ("additive", "saturating"),
    "giou": ("additive", "saturating"),
    "centre_distance": ("additive", "saturating"),
    "expansion_iou": ("additive", "saturating"),
}

_GRIDS: dict[str, dict[GateForm, tuple[tuple[float, ...], tuple[float, ...]]]] = {
    "iou": {
        "additive": ((0.5, 0.7, 0.85, 0.95), (0.0, 0.25, 0.5, 0.75, 1.0)),
        "saturating": ((0.3, 0.5, 0.7, 0.85), (0.5, 1.0, 2.0, 4.0, 8.0)),
    },
    "giou": {
        "additive": ((0.8, 1.1, 1.4, 1.7), (0.5, 1.0, 1.5, 2.0, 3.0)),
        "saturating": ((0.5, 0.8, 1.1, 1.4), (0.5, 1.0, 2.0, 4.0, 8.0)),
    },
    "centre_distance": {
        "additive": ((0.2, 0.4, 0.6, 0.8), (0.0, 0.3, 0.6, 1.0, 1.5)),
        "saturating": ((0.2, 0.4, 0.6, 0.8), (0.5, 1.0, 2.0, 4.0, 8.0)),
    },
    "expansion_iou": {
        "additive": ((0.5, 0.7, 0.85, 0.95), (0.0, 0.25, 0.5, 0.75, 1.0)),
        "saturating": ((0.3, 0.5, 0.7, 0.85), (0.5, 1.0, 2.0, 4.0, 8.0)),
    },
}


@dataclass(frozen=True, slots=True)
class Candidate:
    """One (cost function, gate form)'s gate settings -- Phase A's search variable."""

    cost_name: str
    gate_form: GateForm
    max_cost: float
    kappa: float


@dataclass(frozen=True, slots=True)
class Score:
    """One candidate's aggregate score over some slice of the sweep grid."""

    idsw: float
    never: float
    merges: float
    mostly_tracked: float


@dataclass(frozen=True, slots=True)
class OperatingPoint:
    """A `Candidate` plus what it scored -- Phase A's chosen row for one cost/gate-form."""

    candidate: Candidate
    score: Score


def _build_tracker(candidate: Candidate, n_init: int) -> ByteTrackTracker:
    return ByteTrackTracker(
        n_init=n_init,
        cost=COSTS[candidate.cost_name],
        max_cost=candidate.max_cost,
        max_cost_low=candidate.max_cost * _MAX_COST_LOW_RATIO,
        gate_widening_per_second=candidate.kappa,
        gate_form=candidate.gate_form,
    )


def _run(
    candidate: Candidate,
    scenario: str,
    fps: float,
    n_init: int,
    seed: int,
    *,
    corruption: dict[str, float],
) -> RunScore:
    run = walk_scenario(scenario, dt_s=1.0 / fps, seed=seed, **corruption)
    tracker = _build_tracker(candidate, n_init)
    published = [tracker.update(frame, dets) for frame, dets, _ in run]
    return score_run(
        published,
        [truth for _, _, truth in run],
        width=WIDTH,
        height=HEIGHT,
        box_sigma=corruption["box_sigma"],
    )


def _aggregate(
    candidate: Candidate,
    fps_values: tuple[float, ...],
    n_inits: tuple[int, ...],
    seeds: tuple[int, ...],
    *,
    clean: bool = False,
) -> Score:
    """Mean ID switches, never-confirmed, merges, and mostly-tracked, over the grid."""
    corruption = CLEAN if clean else CORRUPTED
    scores = [
        _run(candidate, scenario, fps, n_init, seed, corruption=corruption)
        for scenario in SCENARIOS
        for fps in fps_values
        for n_init in n_inits
        for seed in seeds
    ]
    n = len(scores)
    return Score(
        idsw=sum(s.id_switches for s in scores) / n,
        never=sum(s.never_confirmed for s in scores) / n,
        merges=sum(s.merges for s in scores) / n,
        mostly_tracked=sum(s.mostly_tracked for s in scores) / n,
    )


@cache
def _tune(cost_name: str, gate_form: GateForm) -> OperatingPoint:
    """Phase A: the `(max_cost, kappa)` cell minimising never-confirmed for this
    (cost, gate form), tie-broken against inflating ID switches and merges.

    Selection is lexicographic, in the ADR's own stated priority order: first the set of
    cells tied (exactly) for the lowest mean never-confirmed, then within that set the
    one with the lowest mean `id_switches + merges`, then -- among any cells still tied
    -- the smallest kappa and max_cost (Step 3's simplicity tiebreak: a more
    conservative gate is easier to reason about when nothing else distinguishes two
    cells).

    Aggregated jointly across `N_INIT_GRID`, not a single fixed `n_init` (Fix round 1,
    Important 6): tuning at `n_init=2` alone let every `centre_distance_cost` cell tie
    at `never-confirmed=0` (2-3 confirming hits arrive well within any of the grid's
    gates at that n_init), so selection silently fell through to the kappa tiebreak on
    a metric that was not actually discriminating between cells. `n_init` is still a
    Phase B axis in its own right (ADR-0014 decision #4) -- this does not tune it, it
    just stops Phase A's OWN selection from resting on a degenerate slice of the grid.
    """
    max_costs, kappas = _GRIDS[cost_name][gate_form]
    points = [
        OperatingPoint(candidate, _aggregate(candidate, FPS_GRID, N_INIT_GRID, TUNE_SEEDS))
        for max_cost, kappa in itertools.product(max_costs, kappas)
        for candidate in [Candidate(cost_name, gate_form, max_cost, kappa)]
    ]
    best_never = min(p.score.never for p in points)
    near_best = [p for p in points if p.score.never <= best_never + 1e-9]
    best_penalty = min(p.score.idsw + p.score.merges for p in near_best)
    tied = [p for p in near_best if (p.score.idsw + p.score.merges) <= best_penalty + 1e-9]
    tied.sort(key=lambda p: (p.candidate.kappa, p.candidate.max_cost))
    return tied[0]


def _all_operating_points() -> list[OperatingPoint]:
    return [
        _tune(cost_name, gate_form)
        for cost_name, forms in GATE_FORMS_BY_COST.items()
        for gate_form in forms
    ]


def _print_row(cost_name: str, gate_form: str, prefix: str, score: Score) -> None:
    print(
        f"{cost_name:<18}{gate_form:<12}{prefix}{score.idsw:>8.2f}{score.never:>8.2f}"
        f"{score.merges:>8.2f}{score.mostly_tracked:>8.2f}"
    )


@pytest.mark.slow
def test_phase_a_tune_each_candidate() -> None:
    """Not an assertion -- the study. Run with `-s`; the printed table is task-9-
    report.md's Phase A, in full."""
    print(
        f"\n{'cost':<18}{'gate':<12}{'max_cost':>10}{'kappa':>8}{'IDSW':>8}{'NEVER':>8}{'MERGE':>8}{'MT':>8}"
    )
    for point in _all_operating_points():
        prefix = f"{point.candidate.max_cost:>10.3f}{point.candidate.kappa:>8.3f}"
        _print_row(point.candidate.cost_name, point.candidate.gate_form, prefix, point.score)


@pytest.mark.slow
def test_phase_b_compare_tuned_candidates() -> None:
    """Not an assertion -- the study. Run with `-s`; the printed table is task-9-
    report.md's Phase B, the table Task 10 pastes into the ADR."""
    print(
        f"\n{'cost':<18}{'gate':<12}{'fps':>5}{'n_init':>8}{'IDSW':>8}{'NEVER':>8}{'MERGE':>8}{'MT':>8}"
    )
    for point in _all_operating_points():
        for fps, n_init in itertools.product(FPS_GRID, N_INIT_GRID):
            score = _aggregate(point.candidate, (fps,), (n_init,), SEEDS)
            prefix = f"{fps:>5.0f}{n_init:>8}"
            _print_row(point.candidate.cost_name, point.candidate.gate_form, prefix, score)


def _grid_sum(candidate: Candidate, seed: int) -> float:
    """One seed's total `id_switches + merges`, summed over the whole fps x n_init x
    scenario grid -- the per-seed observation the significance test pairs on."""
    return sum(
        s.id_switches + s.merges
        for scenario in SCENARIOS
        for fps in FPS_GRID
        for n_init in N_INIT_GRID
        for s in [_run(candidate, scenario, fps, n_init, seed, corruption=CORRUPTED)]
    )


@pytest.mark.slow
def test_significance_of_the_ranking() -> None:
    """Not an assertion -- the study. Fix round 1, Important 3: the first pass reported
    a ranking (by mean full-grid ID-switches + merges) without checking whether the
    seeds it ran actually separated the candidates. They did not, at 5 seeds (paired t,
    p~0.20 against the runner-up) -- a real property of the noise floor at that seed
    count, not a mistake, but the report stated the ranking as settled fact regardless.

    Runs a paired t-test (`scipy.stats.ttest_rel`, one observation per seed: that seed's
    total `id_switches + merges` across the whole fps x n_init x scenario grid) between
    the lowest-total candidate and every other candidate, at `SEEDS`'s full 40. Paired,
    not independent-samples, because the same 40 seeds generate both candidates' runs --
    the walker geometry and detection corruption are identical between the two, so only
    the tracker's cost/gate differs, exactly what a paired test isolates for.
    """
    points = _all_operating_points()

    def _penalty(point: OperatingPoint) -> float:
        score = _aggregate(point.candidate, FPS_GRID, N_INIT_GRID, SEEDS)
        return score.idsw + score.merges

    winner = min(points, key=_penalty)
    winner_vals = [_grid_sum(winner.candidate, seed) for seed in SEEDS]

    print(f"\nwinner: {winner.candidate.cost_name}/{winner.candidate.gate_form}")
    print(f"{'rival':<18}{'gate':<12}{'winner mean':>12}{'rival mean':>12}{'t':>8}{'p':>10}")
    for point in points:
        if point is winner:
            continue
        rival_vals = [_grid_sum(point.candidate, seed) for seed in SEEDS]
        t_stat, p_value = ttest_rel(winner_vals, rival_vals)
        print(
            f"{point.candidate.cost_name:<18}{point.candidate.gate_form:<12}"
            f"{sum(winner_vals) / len(winner_vals):>12.2f}"
            f"{sum(rival_vals) / len(rival_vals):>12.2f}{t_stat:>8.2f}{p_value:>10.4f}"
        )


@pytest.mark.slow
def test_clean_input_is_a_tie() -> None:
    """Constraint 2's sanity check, kept in the suite but never used to rank.

    Every candidate's own Phase A operating point, replayed under clean (uncorrupted)
    input: an all-tie `NEVER`/`IDSW` result here is evidence the harness is not
    fabricating discrimination it has no basis for -- exact constant-velocity motion
    makes the Kalman prediction exact, leaving nothing for any cost to disambiguate.
    """
    print(f"\n{'cost':<18}{'gate':<12}{'IDSW':>8}{'NEVER':>8}{'MERGE':>8}{'MT':>8}")
    for point in _all_operating_points():
        score = _aggregate(point.candidate, FPS_GRID, N_INIT_GRID, TUNE_SEEDS, clean=True)
        _print_row(point.candidate.cost_name, point.candidate.gate_form, "", score)


# --- The fast regression guard -------------------------------------------------------
#
# Frozen from Phase A's own measurement (task-9-report.md), not recomputed here: the
# fast path must not re-run the grid search on every commit, and a hardcoded, documented
# baseline is what makes this a REGRESSION guard rather than a second copy of the study.
_TUNED_IOU_CANDIDATE = Candidate(cost_name="iou", gate_form="additive", max_cost=0.95, kappa=0.25)
"""`iou_cost`'s own Phase A winner (task-9-report.md, Phase A table) -- the fair
baseline this guard holds the shipped default to. Comparing against `iou_cost` run at
the SHIPPED DEFAULT's gate settings would measure whose gate the defaults happen to
fit, not whether the chosen cost function is actually better once IoU gets its own fair
tuning too.

`max_cost` was wrongly frozen as `0.85` in the first pass -- transcribed from an
intermediate run rather than the Phase A table actually printed alongside it, which
already said `0.95` (Fix round 1, Important 7). At the wrong value the guard's own
"real margin" was partly an artifact of an under-tuned baseline (NEVER=1.375, MT=0.042
at 0.85 vs NEVER=0.000, MT=0.448 at the true 0.95, against this candidate's own
GUARD_SEEDS numbers) -- a guard that flatters the shipped default by comparing it to a
strawman is worse than no guard. Fixed to match Phase A's actual, reproducible output."""


def test_the_chosen_cost_still_beats_a_tuned_iou_baseline() -> None:
    """ADR-0014's decision, held in place. If this fails, the ADR must be revisited."""
    chosen = ByteTrackTracker().cost
    assert chosen is not iou_cost, "the default must not be the control"
    chosen_name = next(name for name, fn in COSTS.items() if fn is chosen)
    default_tracker = ByteTrackTracker()
    n_init = default_tracker._n_init  # the guard must test the shipped default

    chosen_candidate = Candidate(
        cost_name=chosen_name,
        gate_form=default_tracker._gate_form,
        max_cost=default_tracker._max_cost,
        kappa=default_tracker._gate_widening_per_second,
    )
    chosen_score = _aggregate(chosen_candidate, (2.0,), (n_init,), GUARD_SEEDS)
    iou_score = _aggregate(_TUNED_IOU_CANDIDATE, (2.0,), (n_init,), GUARD_SEEDS)

    assert chosen_score.never <= iou_score.never
    assert chosen_score.mostly_tracked > iou_score.mostly_tracked
    assert chosen_score.idsw + chosen_score.merges <= iou_score.idsw + iou_score.merges + 1.0, (
        "a permissive gate that only fixes NEVER by swapping identities has moved the "
        "error, not removed it (ADR-0014's own stated downside of a permissive gate)"
    )
