"""What the engine said, measured against what a human saw.

This module is the one boundary where media time becomes UTC. A truth file counts seconds
from the start of the clip; a ``MetricRow`` carries a wall-clock minute bucket, because
replaying a clip stamps frames with the clock of the replay rather than of the original
recording. :func:`score` reconciles the two by subtracting the instant the replay began —
and refuses when that instant is not on a minute boundary, because a clip started at
09:30:20 has a media minute 0 that no wall-clock bucket corresponds to.

Scope is count error, on the two quantities a clicker's crossings can answer for:
``footfall`` (inward visits, what the gate is specified on) and ``line_cross`` (traffic
in both directions). Tracking quality, dwell and queue need a different observation than
a truth file carries, so nothing here computes them.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from math import ceil

from mesopic.errors import TruthError
from mesopic.truth.clips import ClipManifest, SceneReference, gate_eligible
from mesopic.truth.labels import DRAFT_RATER, TruthFile, check_pairing
from mesopic.types import ClipId, Direction, MetricName, MetricRow, MinuteBucket

SECONDS_PER_MINUTE = 60

DEFAULT_GATE_SCENE = SceneReference.GOOD_DOORWAY
"""The scene v1's accuracy gate is specified against (accuracy-targets-and-sla.md §1.1).

A default rather than a constant because the gated scene genuinely varies: M1 gates the
good doorway, while the v1.0-committed targets M5 answers to are stated against
``typical``. Hardcoding one would make the other unexpressible.
"""


@dataclass(frozen=True, slots=True)
class Score:
    """One clip's accuracy, and whether it was permitted to back a released claim.

    ``gating`` is carried on the result rather than left at the call site so an artefact
    read later still says whether it was a gate run or a measurement.
    """

    clip_id: ClipId
    gating: bool
    metric: MetricName
    """Which quantity was measured. Carried for the same reason as ``gating``, and for a
    sharper one: footfall and ``line_cross`` are different counts read against different
    bands, so a total that does not name its metric cannot be read once it outlives the
    call site — a doorway watched while a building empties scores 11 on one and 2 on the
    other."""

    minutes: int
    truth_total: int
    predicted_total: float
    total_error_pct: float
    """Absolute count error over the whole clip — the shape the published number takes."""

    mape_pct: float
    """Mean absolute percentage error across minute buckets. Harsher than the total: it
    catches an engine that is right overall by being wrong in both directions."""

    stray_minutes: int
    """Minutes the engine attributed counts to that the clip does not span.

    Counted separately from `minutes` because the clip's length is a fact and this is a
    symptom: a non-zero value means the run emitted counts outside the footage, which is
    either a teardown flush or a misaligned `stream_start`. The error percentages already
    include the stray counts; this says where to look for them.
    """


def _empty_minutes(truth: TruthFile) -> dict[int, int]:
    """Every minute the clip spans, at zero.

    **Including the empty ones.** A quiet minute is an observation the engine can get
    wrong, and dropping it would let a quiet hour score perfectly by default.
    """
    minutes = max(1, ceil(truth.duration_s / SECONDS_PER_MINUTE))
    return dict.fromkeys(range(minutes), 0)


def footfall_per_minute(truth: TruthFile) -> dict[int, int]:
    """Reduce labelled crossings to a per-minute footfall series, in media time.

    Footfall counts inward crossings only. An exit is a real event and not a visit; it is
    carried by the ``line_cross`` metric.
    """
    series = _empty_minutes(truth)
    for crossing in truth.crossings:
        if crossing.direction is Direction.IN:
            series[int(crossing.t_s // SECONDS_PER_MINUTE)] += 1
    return series


def line_crossings_per_minute(truth: TruthFile) -> dict[int, int]:
    """Reduce labelled crossings to a per-minute traffic series, in media time.

    Both directions, because an exit is traffic even though it is not a visit. This is
    the series to score a clip on when its crossings are mostly outward — a doorway
    watched while a building empties has plenty of events and almost no footfall, and
    scoring it on footfall would measure two of eleven observations.
    """
    series = _empty_minutes(truth)
    for crossing in truth.crossings:
        series[int(crossing.t_s // SECONDS_PER_MINUTE)] += 1
    return series


def _traffic(row: MetricRow) -> float:
    """`line_cross` carries the signed net in `value`; its traffic is `sample_count`.

    `LineCrossPlugin` stores the net on purpose — the direction is the information, and a
    net is what makes a line usable as an occupancy integrator. The truth series counts
    every crossing, because an exit is traffic even though it is not a visit. Folding
    `value` would therefore compare a net against a total and read a minute of three
    arrivals and one departure — which the engine got exactly right — as a 50%
    under-count. `sample_count` is the field that means the same thing on both sides.
    """
    return float(row.sample_count)


def _value(row: MetricRow) -> float:
    return row.value


_PREDICTED_QUANTITY: dict[MetricName, Callable[[MetricRow], float]] = {
    MetricName.FOOTFALL: _value,
    MetricName.LINE_CROSS: _traffic,
}
"""Which field of a run's row corresponds to each truth series.

Kept beside `_TRUTH_SERIES` because the two have to agree: a metric that names a truth
series and reads the wrong column of the run scores a correct engine as a broken one.
"""


_TRUTH_SERIES = {
    MetricName.FOOTFALL: footfall_per_minute,
    MetricName.LINE_CROSS: line_crossings_per_minute,
}
"""The metrics a truth file can be scored on, and how each reduces to minute buckets.

Only these two: a truth file records line crossings, so it can answer for visits and for
traffic and for nothing else. Dwell, queue and occupancy need a different observation
than a clicker produces.
"""


def mape(predicted: Mapping[int, float], truth: Mapping[int, int]) -> float:
    """Mean absolute percentage error across the truth's windows, as a percentage.

    The denominator is ``max(true, 1)``: an empty window is common in real footage and
    would otherwise divide by zero. A window the engine emitted nothing for counts as a
    prediction of zero — silence is a claim that nothing happened, not an absence of one.
    """
    if not truth:
        message = "cannot average an error over no windows"
        raise TruthError(message)

    total = sum(
        abs(predicted.get(minute, 0.0) - actual) / max(actual, 1)
        for minute, actual in truth.items()
    )
    return 100.0 * total / len(truth)


def gate_blockers(
    truth: TruthFile, manifest: ClipManifest, *, scene: SceneReference
) -> tuple[str, ...]:
    """Every reason this clip and these labels may not back a published number.

    Two different questions, kept separate on purpose. :func:`gate_eligible` answers the
    *permission* one — may we publish from footage of these people at all — and is
    settled by ADR-0015; it is unchanged and still means exactly what the ADR says. What
    follows it here is the *validity* question: does this clip measure the thing the gate
    is specified on, and did a human stand behind the labels. A clip can pass the first
    and fail the second, which is not a hypothetical — the 2.1 m own-rig home clips are
    gate-eligible, `hard`, and would have scored without complaint.

    Returns every failure rather than the first, so a clip that is wrong three ways says
    so in one pass instead of one refusal per fix.
    """
    blockers: list[str] = []
    if not gate_eligible(manifest):
        blockers.append(
            f"provenance {manifest.provenance.kind} with consent "
            f"{manifest.consent.model_release} is not gate-eligible"
        )
    if manifest.scene.reference is not scene:
        blockers.append(
            f"scene is {manifest.scene.reference}, but this gate is specified on {scene}"
        )
    if truth.labelled_by == DRAFT_RATER:
        blockers.append(f"labels are unverified ({DRAFT_RATER})")
    return tuple(blockers)


def score(
    truth: TruthFile,
    manifest: ClipManifest,
    metrics: Iterable[MetricRow],
    *,
    stream_start: MinuteBucket,
    gating: bool,
    gate_scene: SceneReference = DEFAULT_GATE_SCENE,
    metric: MetricName = MetricName.FOOTFALL,
) -> Score:
    """Measure the engine's count against a labelled clip.

    ``gating`` is explicit and unforgiving: asking to gate on a clip that may not back a
    published number raises rather than returning one. Measuring the same clip with
    ``gating=False`` is always fine, and is what metric development runs against all day
    — every refusal here is about publishing, never about measuring.

    ``gate_scene`` names which reference scene is being gated, because a target quoted
    outside its scene is void and the scene differs by milestone.

    ``metric`` selects **both** sides of the comparison — which crossings the truth series
    counts, and which of the run's rows are read. It defaults to footfall because that is
    what the gate is specified on; ``line_cross`` scores traffic instead, which is the
    honest choice for a clip whose crossings are mostly outward. The two are different
    quantities and the band each is read against differs, so a result carries the metric
    it was measured on.

    What is *not* checked, and is left to the caller: clip length. The gate is stated at
    hour grain, so a three-minute clip cannot produce it — but whether a given run needs
    an hour, thirty minutes, or a minute-grain series is an open call, and encoding a
    guess here would settle it by accident.
    """
    check_pairing(truth, manifest)
    if gating and (blockers := gate_blockers(truth, manifest, scene=gate_scene)):
        message = f"clip {manifest.clip_id} may not back a published number: " + "; ".join(blockers)
        raise TruthError(message)

    if metric not in _TRUTH_SERIES:
        supported = ", ".join(sorted(name.value for name in _TRUTH_SERIES))
        message = f"a truth file cannot answer for {metric.value}; it records {supported}"
        raise TruthError(message)

    expected = _TRUTH_SERIES[metric](truth)
    predicted = _predicted_per_minute(metrics, stream_start, metric)

    # Every minute the engine spoke about is a window it can be wrong in, including the
    # ones outside the clip. Reading only the truth's minutes would score a run that
    # invented a hundred crossings after the footage ended as flawless — an over-count
    # the gate cannot see is worse than no gate, because it reads as a pass.
    stray = sorted(set(predicted) - set(expected))
    windows = {**expected, **dict.fromkeys(stray, 0)}

    truth_total = sum(expected.values())
    predicted_total = sum(predicted.values())

    return Score(
        clip_id=truth.clip_id,
        gating=gating,
        metric=metric,
        minutes=len(expected),
        truth_total=truth_total,
        predicted_total=predicted_total,
        total_error_pct=100.0 * abs(predicted_total - truth_total) / max(truth_total, 1),
        mape_pct=mape(predicted, windows),
        stray_minutes=len(stray),
    )


def _predicted_per_minute(
    metrics: Iterable[MetricRow], stream_start: MinuteBucket, metric: MetricName
) -> dict[int, float]:
    """Fold one metric's rows onto media-time minute indices. The one UTC conversion."""
    if stream_start.tzinfo is None:
        message = "stream_start must be timezone-aware UTC"
        raise TruthError(message)
    if (stream_start.second, stream_start.microsecond) != (0, 0):
        # Media minute 0 spans the clip's first 60 seconds; a MinuteBucket is a wall-clock
        # minute. Off a boundary the two do not correspond, and picking one silently
        # attributes crossings to the wrong minute.
        message = "stream_start must fall on a minute boundary for media time to align"
        raise TruthError(message)

    selected = [row for row in metrics if row.metric is metric]
    _refuse_mixed_scopes(selected)

    read = _PREDICTED_QUANTITY[metric]
    predicted: dict[int, float] = {}
    for row in selected:
        minute = int((row.bucket - stream_start).total_seconds() // SECONDS_PER_MINUTE)
        predicted[minute] = predicted.get(minute, 0.0) + read(row)
    return predicted


def _refuse_mixed_scopes(rows: list[MetricRow]) -> None:
    """One series per run, and the caller says which.

    Rows scoped to different lines sum correctly — two doors on one camera are two rows
    and one visit count. A camera-wide row is that same quantity counted a second way, so
    summing it alongside the scoped rows doubles it. Whether P2.4 emits per-line rows, a
    camera-wide row, or both is P2.4's decision; what must not happen is a mix being
    scored and the doubling being reported as an engine accuracy failure.
    """
    scoped = any(row.scope_id is not None for row in rows)
    camera_wide = any(row.scope_id is None for row in rows)
    if scoped and camera_wide:
        message = (
            "metric rows mix per-scope and camera-wide totals; these are the same count "
            "measured twice, so scoring both would double it — pass one series"
        )
        raise TruthError(message)
