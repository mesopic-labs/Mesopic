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
from pathlib import Path

import pytest

from muster.errors import TruthError
from muster.truth import (
    ClipManifest,
    SceneReference,
    TruthFile,
    footfall_per_minute,
    gate_blockers,
    gate_eligible,
    load_manifest,
    load_truth,
    mape,
    score,
)
from muster.types import CameraId, MetricName, MetricRow, MinuteBucket, ScopeId

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
CAMERA = CameraId("front-door")
STREAM_START = MinuteBucket(datetime(2026, 8, 16, 9, 30, 0, tzinfo=UTC))


def _truth(
    *crossings: tuple[float, str], duration_s: float = 180.0, labelled_by: str = "mark"
) -> TruthFile:
    return TruthFile.model_validate(
        {
            "schema_version": 1,
            "clip_id": "doorway-daylight-01",
            "labelled_by": labelled_by,
            "labelled_at_utc": "2026-08-17T14:02:11Z",
            "duration_s": duration_s,
            "crossings": [
                {"t_s": t_s, "line_id": "entrance", "direction": direction}
                for t_s, direction in crossings
            ],
        }
    )


def _manifest(
    *, kind: str = "own_rig", release: str = "obtained", scene: str = "good_doorway"
) -> ClipManifest:
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
                "reference": scene,
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


def test_gating_on_a_clip_outside_the_gated_scene_is_refused() -> None:
    """Consent is not the only way a clip can be the wrong basis for a number.

    A `hard` clip is footage we are allowed to publish from and still measures a
    different thing than the good-doorway target it would be read against. Before this
    check the 2.1 m home clips were gate-eligible and would have scored without complaint.
    """
    truth = _truth((10.0, "in"))

    with pytest.raises(TruthError, match="scene"):
        score(
            truth,
            _manifest(scene="hard"),
            [_row(0, 1.0)],
            stream_start=STREAM_START,
            gating=True,
        )


def test_gating_on_the_scene_actually_being_gated_is_allowed() -> None:
    """The gated scene is the caller's to name. M1 gates good_doorway; M5's committed
    targets are stated against `typical`, so hardcoding one scene would be wrong."""
    truth = _truth((10.0, "in"))

    result = score(
        truth,
        _manifest(scene="typical"),
        [_row(0, 1.0)],
        stream_start=STREAM_START,
        gating=True,
        gate_scene=SceneReference.TYPICAL,
    )

    assert result.gating is True


def test_gating_on_unverified_labels_is_refused() -> None:
    """`draft-unverified` is an admission, and the gate must not accept one.

    The rater name is the accountability mechanism: a draft can be promoted only by a
    human putting their own name on it, which is a deliberate act rather than a flag.
    """
    truth = _truth((10.0, "in"), labelled_by="draft-unverified")

    with pytest.raises(TruthError, match="unverified"):
        score(truth, _manifest(), [_row(0, 1.0)], stream_start=STREAM_START, gating=True)


def test_a_hard_clip_with_draft_labels_still_measures_without_gating() -> None:
    """Every refusal above is about publishing, never about measuring. Metric development
    runs against exactly this shape of clip all day and must stay unobstructed."""
    truth = _truth((10.0, "in"), labelled_by="draft-unverified")

    result = score(
        truth,
        _manifest(scene="hard"),
        [_row(0, 1.0)],
        stream_start=STREAM_START,
        gating=False,
    )

    assert result.gating is False


def test_gate_blockers_reports_every_reason_at_once() -> None:
    """A clip that fails three ways should say so in one pass, not one refusal per fix."""
    blockers = gate_blockers(
        _truth((10.0, "in"), labelled_by="draft-unverified"),
        _manifest(kind="stock", release="unknown", scene="hard"),
        scene=SceneReference.GOOD_DOORWAY,
    )

    assert len(blockers) == 3


def test_the_committed_own_rig_clips_cannot_gate() -> None:
    """The regression this guard exists for, pinned against the real files.

    `home-hallway-oblique-01` is genuinely gate-eligible — our own rig, consent obtained —
    and was shot on a 2.1 m mount, which makes it `hard`. Both halves matter: if a future
    edit promotes its scene, relaxes the check, or quietly widens eligibility, this fails
    before anybody reads a published number off eighty seconds of a domestic hallway.
    """
    manifest = load_manifest(FIXTURES / "clips" / "home-hallway-oblique-01.clip.json")
    truth = load_truth(FIXTURES / "truth" / "home-hallway-oblique-01.truth.json")

    assert gate_eligible(manifest) is True

    with pytest.raises(TruthError, match="scene"):
        score(truth, manifest, [], stream_start=STREAM_START, gating=True)


def test_gate_blockers_is_empty_for_a_clip_that_may_gate() -> None:
    assert gate_blockers(_truth((10.0, "in")), _manifest(), scene=SceneReference.GOOD_DOORWAY) == ()


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


# --- Predictions the clip cannot account for --------------------------------
#
# Added on review: the fold originally read only the minutes the truth spans, so footfall
# attributed outside the clip was invisible. A gate number that cannot see an over-count
# is worse than no gate number, because it reads as a pass.


def _scoped_row(minute: int, value: float, scope: str | None) -> MetricRow:
    return MetricRow(
        camera_id=CAMERA,
        bucket=MinuteBucket(STREAM_START + timedelta(minutes=minute)),
        metric=MetricName.FOOTFALL,
        scope_id=ScopeId(scope) if scope is not None else None,
        value=value,
    )


def test_footfall_attributed_past_the_end_of_the_clip_is_counted_as_error() -> None:
    """A teardown flush emitting one last bucket must not score as a perfect run."""
    truth = _truth((30.0, "in"))

    result = score(
        truth, _manifest(), [_row(0, 1.0), _row(30, 99.0)], stream_start=STREAM_START, gating=False
    )

    assert result.predicted_total == 100.0
    assert result.total_error_pct > 0.0


def test_a_stray_minute_is_reported_so_the_number_can_be_explained() -> None:
    """`total_error_pct` says the run was wrong; `stray_minutes` says where to look."""
    truth = _truth((30.0, "in"))

    result = score(
        truth, _manifest(), [_row(0, 1.0), _row(7, 2.0)], stream_start=STREAM_START, gating=False
    )

    assert result.stray_minutes == 1
    assert result.minutes == 3, "the clip's own length is a fact; a stray minute is not part of it"


def test_a_clean_run_reports_no_stray_minutes() -> None:
    truth = _truth((30.0, "in"))

    result = score(truth, _manifest(), [_row(0, 1.0)], stream_start=STREAM_START, gating=False)

    assert result.stray_minutes == 0
    assert result.total_error_pct == 0.0


def test_a_stray_minute_is_not_a_free_window_in_the_mape() -> None:
    """Averaging over the truth's windows alone would dilute a phantom minute to nothing."""
    truth = _truth((30.0, "in"))

    clean = score(truth, _manifest(), [_row(0, 1.0)], stream_start=STREAM_START, gating=False)
    strayed = score(
        truth, _manifest(), [_row(0, 1.0), _row(9, 5.0)], stream_start=STREAM_START, gating=False
    )

    assert strayed.mape_pct > clean.mape_pct


# --- Which footfall series is being scored ----------------------------------


def test_footfall_scoped_to_several_lines_sums() -> None:
    """Two doors on one camera are two rows and one visit count."""
    truth = _truth((10.0, "in"), (20.0, "in"))

    result = score(
        truth,
        _manifest(),
        [_scoped_row(0, 1.0, "front"), _scoped_row(0, 1.0, "side")],
        stream_start=STREAM_START,
        gating=False,
    )

    assert result.predicted_total == 2.0
    assert result.total_error_pct == 0.0


def test_mixing_scoped_and_camera_wide_footfall_is_refused() -> None:
    """The two are the same quantity counted twice, and summing them doubles it.

    Which of the two P2.4 emits is P2.4's decision. What must not happen is scoring a mix
    of both and reporting the resulting doubling as an engine accuracy failure.
    """
    truth = _truth((10.0, "in"))

    with pytest.raises(TruthError, match="camera-wide"):
        score(
            truth,
            _manifest(),
            [_scoped_row(0, 1.0, "front"), _scoped_row(0, 1.0, None)],
            stream_start=STREAM_START,
            gating=False,
        )
