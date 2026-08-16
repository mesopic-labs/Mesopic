"""Scoring: what the engine said, against what a human saw.

The test that matters most here is the one that *refuses* — ``score(..., gating=True)``
on a clip whose provenance or consent does not permit it. That is what turns the
gate-eligibility rule from a paragraph into a property of the code.

The rest pin the arithmetic and the single time-conversion boundary. Truth is in media
time (offsets from clip start) and metric rows are in UTC minute buckets; exactly one
function reconciles them, and it refuses to guess when the alignment is ambiguous.

Written red-first.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from muster.errors import TruthError
from muster.truth import ClipManifest, TruthFile, footfall_per_minute, mape, score
from muster.types import CameraId, MetricName, MetricRow, MinuteBucket

CAMERA = CameraId("front-door")
STREAM_START = MinuteBucket(datetime(2026, 8, 16, 9, 30, 0, tzinfo=UTC))


def _truth(*crossings: tuple[float, str], duration_s: float = 180.0) -> TruthFile:
    return TruthFile.model_validate(
        {
            "schema_version": 1,
            "clip_id": "doorway-daylight-01",
            "labelled_by": "mark",
            "labelled_at_utc": "2026-08-17T14:02:11Z",
            "duration_s": duration_s,
            "crossings": [
                {"t_s": t_s, "line_id": "entrance", "direction": direction}
                for t_s, direction in crossings
            ],
        }
    )


def _manifest(*, kind: str = "own_rig", release: str = "obtained") -> ClipManifest:
    return ClipManifest.model_validate(
        {
            "schema_version": 1,
            "clip_id": "doorway-daylight-01",
            "sha256": "a" * 64,
            "duration_s": 180.0,
            "width": 1920,
            "height": 1080,
            "fps": 25.0,
            "provenance": {
                "kind": kind,
                "licence": "Owned",
                "licence_verified_utc": "2026-08-16",
                "url": None,
            },
            "consent": {"model_release": release, "note": None},
            "scene": {
                "reference": "good_doorway",
                "mount_height_m": 2.8,
                "mount_angle_deg": 42.0,
                "lighting": "even_daylight",
            },
        }
    )


def _row(minute: int, value: float) -> MetricRow:
    return MetricRow(
        camera_id=CAMERA,
        bucket=MinuteBucket(STREAM_START + timedelta(minutes=minute)),
        metric=MetricName.FOOTFALL,
        scope_id=None,
        value=value,
    )


# --- Deriving the truth series ----------------------------------------------


def test_crossings_land_in_the_minute_they_happened_in() -> None:
    truth = _truth((10.0, "in"), (59.9, "in"), (60.1, "in"))

    assert footfall_per_minute(truth) == {0: 2, 1: 1, 2: 0}


def test_outward_crossings_are_not_footfall() -> None:
    """Footfall is people coming in. An exit is a crossing, and not a visit."""
    truth = _truth((10.0, "in"), (20.0, "out"), (30.0, "out"))

    assert footfall_per_minute(truth)[0] == 1


def test_every_minute_the_clip_spans_is_present_even_when_empty() -> None:
    """An empty minute is a real observation, and the one MAPE's guard exists for. If
    quiet minutes were simply absent, the guard would never fire and a quiet hour would
    silently score perfectly."""
    truth = _truth((10.0, "in"), duration_s=180.0)

    assert footfall_per_minute(truth) == {0: 1, 1: 0, 2: 0}


def test_a_partial_final_minute_still_counts_as_a_minute() -> None:
    """A 19-second clip is one minute of observation, not zero."""
    truth = _truth((5.0, "in"), duration_s=19.8)

    assert footfall_per_minute(truth) == {0: 1}


# --- MAPE --------------------------------------------------------------------


def test_mape_is_zero_when_the_engine_agrees() -> None:
    assert mape({0: 10.0, 1: 5.0}, {0: 10, 1: 5}) == 0.0


def test_mape_averages_the_per_window_error() -> None:
    """10 vs 8 is 20%, 5 vs 5 is 0%, so the mean is 10%."""
    assert mape({0: 8.0, 1: 5.0}, {0: 10, 1: 5}) == pytest.approx(10.0)


def test_mape_guards_an_empty_window_against_dividing_by_zero() -> None:
    """Truth of 0 and a prediction of 2 is 200% against ``max(true, 1)``, not a crash."""
    assert mape({0: 2.0}, {0: 0}) == pytest.approx(200.0)


def test_mape_treats_a_missing_prediction_as_zero() -> None:
    """The engine emitting no row for a minute is a claim that nothing happened."""
    assert mape({}, {0: 4}) == pytest.approx(100.0)


def test_mape_of_nothing_is_an_error() -> None:
    """An average over no windows is not zero, and reporting it as zero would read as a
    perfect score."""
    with pytest.raises(TruthError):
        mape({}, {})


# --- Scoring, and the refusal that makes gate-eligibility real ---------------


def test_scoring_aligns_utc_buckets_to_media_time() -> None:
    truth = _truth((10.0, "in"), (70.0, "in"), (80.0, "in"))
    rows = [_row(0, 1.0), _row(1, 2.0)]

    result = score(truth, _manifest(), rows, stream_start=STREAM_START, gating=False)

    assert result.truth_total == 3
    assert result.predicted_total == 3.0
    assert result.mape_pct == 0.0


def test_scoring_reports_the_error_when_the_engine_undercounts() -> None:
    truth = _truth((10.0, "in"), (20.0, "in"), (30.0, "in"), (40.0, "in"))
    rows = [_row(0, 3.0)]

    result = score(truth, _manifest(), rows, stream_start=STREAM_START, gating=False)

    assert result.truth_total == 4
    assert result.predicted_total == 3.0
    assert result.total_error_pct == pytest.approx(25.0)


def test_scoring_ignores_metrics_that_are_not_footfall() -> None:
    truth = _truth((10.0, "in"))
    occupancy = MetricRow(
        camera_id=CAMERA,
        bucket=MinuteBucket(STREAM_START),
        metric=MetricName.OCCUPANCY,
        scope_id=None,
        value=99.0,
    )

    result = score(
        truth, _manifest(), [_row(0, 1.0), occupancy], stream_start=STREAM_START, gating=False
    )

    assert result.predicted_total == 1.0


def test_gating_on_stock_footage_is_refused() -> None:
    """The test that makes the rule real. Stock footage cannot back a released number,
    however good the clip is."""
    truth = _truth((10.0, "in"))

    with pytest.raises(TruthError):
        score(
            truth,
            _manifest(kind="stock", release="obtained"),
            [_row(0, 1.0)],
            stream_start=STREAM_START,
            gating=True,
        )


def test_gating_on_unknown_consent_is_refused() -> None:
    truth = _truth((10.0, "in"))

    with pytest.raises(TruthError):
        score(
            truth,
            _manifest(kind="own_rig", release="unknown"),
            [_row(0, 1.0)],
            stream_start=STREAM_START,
            gating=True,
        )


def test_gating_on_eligible_footage_is_allowed() -> None:
    truth = _truth((10.0, "in"))

    result = score(truth, _manifest(), [_row(0, 1.0)], stream_start=STREAM_START, gating=True)

    assert result.gating is True


def test_measuring_ineligible_footage_without_gating_is_fine() -> None:
    """Stock footage is a development fixture. Refusing to score it at all would leave
    plugin work with nothing to iterate against."""
    truth = _truth((10.0, "in"))

    result = score(
        truth,
        _manifest(kind="stock", release="unknown"),
        [_row(0, 1.0)],
        stream_start=STREAM_START,
        gating=False,
    )

    assert result.gating is False


def test_scoring_refuses_a_truth_file_labelled_against_a_different_cut() -> None:
    truth = _truth((10.0, "in"), duration_s=90.0)

    with pytest.raises(TruthError):
        score(truth, _manifest(), [_row(0, 1.0)], stream_start=STREAM_START, gating=False)


def test_scoring_refuses_an_unaligned_stream_start() -> None:
    """Media minute 0 spans the first 60 seconds of the clip; a UTC bucket is a wall-clock
    minute. If the replay did not start on a minute boundary the two do not correspond,
    and quietly picking one is how crossings get attributed to the wrong minute."""
    truth = _truth((10.0, "in"))

    with pytest.raises(TruthError):
        score(
            truth,
            _manifest(),
            [_row(0, 1.0)],
            stream_start=MinuteBucket(STREAM_START + timedelta(seconds=20)),
            gating=False,
        )


def test_scoring_refuses_a_naive_stream_start() -> None:
    truth = _truth((10.0, "in"))

    with pytest.raises(TruthError):
        score(
            truth,
            _manifest(),
            [_row(0, 1.0)],
            stream_start=MinuteBucket(datetime(2026, 8, 16, 9, 30, 0)),  # noqa: DTZ001 - the point
            gating=False,
        )
