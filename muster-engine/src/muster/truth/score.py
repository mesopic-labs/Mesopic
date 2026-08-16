"""What the engine said, measured against what a human saw.

This module is the one boundary where media time becomes UTC. A truth file counts seconds
from the start of the clip; a ``MetricRow`` carries a wall-clock minute bucket, because
replaying a clip stamps frames with the clock of the replay rather than of the original
recording. :func:`score` reconciles the two by subtracting the instant the replay began —
and refuses when that instant is not on a minute boundary, because a clip started at
09:30:20 has a media minute 0 that no wall-clock bucket corresponds to.

Scope is the gate's number and nothing else: footfall count error. Tracking quality,
dwell and queue are recorded elsewhere and do not gate, so nothing here computes them.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import ceil

from muster.errors import TruthError
from muster.truth.clips import ClipManifest, gate_eligible
from muster.truth.labels import TruthFile, check_pairing
from muster.types import ClipId, Direction, MetricName, MetricRow, MinuteBucket

SECONDS_PER_MINUTE = 60


@dataclass(frozen=True, slots=True)
class Score:
    """One clip's accuracy, and whether it was permitted to back a released claim.

    ``gating`` is carried on the result rather than left at the call site so an artefact
    read later still says whether it was a gate run or a measurement.
    """

    clip_id: ClipId
    gating: bool
    minutes: int
    truth_total: int
    predicted_total: float
    total_error_pct: float
    """Absolute count error over the whole clip — the shape the published number takes."""

    mape_pct: float
    """Mean absolute percentage error across minute buckets. Harsher than the total: it
    catches an engine that is right overall by being wrong in both directions."""


def footfall_per_minute(truth: TruthFile) -> dict[int, int]:
    """Reduce labelled crossings to a per-minute footfall series, in media time.

    Keyed by minute index from the start of the clip, and **every minute the clip spans
    is present**, including the empty ones. A quiet minute is an observation the engine
    can get wrong, and dropping it would let a quiet hour score perfectly by default.

    Footfall counts inward crossings only. An exit is a real event and not a visit; it is
    carried by the ``line_cross`` metric, which is recorded and does not gate.
    """
    minutes = max(1, ceil(truth.duration_s / SECONDS_PER_MINUTE))
    series = dict.fromkeys(range(minutes), 0)
    for crossing in truth.crossings:
        if crossing.direction is Direction.IN:
            series[int(crossing.t_s // SECONDS_PER_MINUTE)] += 1
    return series


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


def score(
    truth: TruthFile,
    manifest: ClipManifest,
    metrics: Iterable[MetricRow],
    *,
    stream_start: MinuteBucket,
    gating: bool,
) -> Score:
    """Measure the engine's footfall against a labelled clip.

    ``gating`` is explicit and unforgiving: asking to gate on footage whose provenance or
    consent does not permit it raises rather than returning a number nobody may publish.
    Measuring the same clip with ``gating=False`` is fine, and is what metric development
    runs against all day.
    """
    check_pairing(truth, manifest)
    if gating and not gate_eligible(manifest):
        message = (
            f"clip {manifest.clip_id} is not gate-eligible: "
            f"provenance {manifest.provenance.kind}, consent {manifest.consent.model_release}"
        )
        raise TruthError(message)

    expected = footfall_per_minute(truth)
    predicted = _predicted_per_minute(metrics, stream_start)
    truth_total = sum(expected.values())
    predicted_total = sum(predicted.get(minute, 0.0) for minute in expected)

    return Score(
        clip_id=truth.clip_id,
        gating=gating,
        minutes=len(expected),
        truth_total=truth_total,
        predicted_total=predicted_total,
        total_error_pct=100.0 * abs(predicted_total - truth_total) / max(truth_total, 1),
        mape_pct=mape(predicted, expected),
    )


def _predicted_per_minute(
    metrics: Iterable[MetricRow], stream_start: MinuteBucket
) -> dict[int, float]:
    """Fold footfall rows onto media-time minute indices. The one UTC conversion."""
    if stream_start.tzinfo is None:
        message = "stream_start must be timezone-aware UTC"
        raise TruthError(message)
    if (stream_start.second, stream_start.microsecond) != (0, 0):
        # Media minute 0 spans the clip's first 60 seconds; a MinuteBucket is a wall-clock
        # minute. Off a boundary the two do not correspond, and picking one silently
        # attributes crossings to the wrong minute.
        message = "stream_start must fall on a minute boundary for media time to align"
        raise TruthError(message)

    predicted: dict[int, float] = {}
    for row in metrics:
        if row.metric is not MetricName.FOOTFALL:
            continue
        minute = int((row.bucket - stream_start).total_seconds() // SECONDS_PER_MINUTE)
        predicted[minute] = predicted.get(minute, 0.0) + row.value
    return predicted
