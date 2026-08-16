"""A truth file is a human's account of what happened in a clip.

It is the only thing an accuracy number is measured against, so it is parsed strictly and
rejected on the first inconsistency: a label file that is quietly repaired produces a
number that looks fine and is wrong. Everything here is a rejection test except the two
that pin the accepted shape.

The one convention worth stating twice: ``t_s`` is **media time**, an offset from the
start of the clip, not a UTC instant. Replaying a clip stamps every frame with the wall
clock of the replay, so offsets are the only thing that aligns truth to engine output.

Written red-first.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from muster.errors import TruthError
from muster.truth import ClipManifest, check_pairing, load_manifest, load_truth
from muster.types import Direction, LineId

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_CLIPS = REPO_ROOT / "fixtures" / "clips"
FIXTURE_TRUTH = REPO_ROOT / "fixtures" / "truth"


def _truth_dict(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "schema_version": 1,
        "clip_id": "doorway-daylight-01",
        "labelled_by": "mark",
        "labelled_at_utc": "2026-08-17T14:02:11Z",
        "duration_s": 1800.0,
        "crossings": [
            {"t_s": 12.48, "line_id": "entrance", "direction": "in"},
            {"t_s": 31.2, "line_id": "entrance", "direction": "out"},
        ],
    }
    base.update(overrides)
    return base


def _write(tmp_path: Path, payload: dict[str, Any]) -> Path:
    path = tmp_path / "doorway-daylight-01.truth.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _manifest(duration_s: float = 1800.0) -> ClipManifest:
    return ClipManifest.model_validate(
        {
            "schema_version": 1,
            "clip_id": "doorway-daylight-01",
            "sha256": "a" * 64,
            "duration_s": duration_s,
            "width": 1920,
            "height": 1080,
            "fps": 25.0,
            "provenance": {
                "kind": "own_rig",
                "licence": "Owned",
                "licence_verified_utc": "2026-08-16",
                "url": None,
            },
            "consent": {"model_release": "obtained", "note": None},
            "scene": {
                "reference": "good_doorway",
                "mount_height_m": 2.8,
                "mount_angle_deg": 42.0,
                "lighting": "even_daylight",
            },
        }
    )


# --- The shape that is accepted ---------------------------------------------


def test_parses_a_well_formed_truth_file(tmp_path: Path) -> None:
    truth = load_truth(_write(tmp_path, _truth_dict()))

    assert truth.duration_s == 1800.0
    assert len(truth.crossings) == 2
    assert truth.crossings[0].direction is Direction.IN
    assert truth.crossings[0].line_id == LineId("entrance")
    assert truth.crossings[0].t_s == 12.48


def test_a_clip_with_no_crossings_is_valid_truth(tmp_path: Path) -> None:
    """ "Nobody walked through" is an observation, not a missing label."""
    truth = load_truth(_write(tmp_path, _truth_dict(crossings=[])))

    assert truth.crossings == ()


def test_labelled_at_is_stored_as_aware_utc(tmp_path: Path) -> None:
    truth = load_truth(_write(tmp_path, _truth_dict()))

    assert truth.labelled_at_utc == datetime(2026, 8, 17, 14, 2, 11, tzinfo=UTC)
    assert truth.labelled_at_utc.tzinfo is not None


# --- Rejections --------------------------------------------------------------


def test_rejects_a_naive_labelled_at(tmp_path: Path) -> None:
    """ "UTC everywhere" is not a comment; a naive stamp is an unanswerable question."""
    with pytest.raises(TruthError):
        load_truth(_write(tmp_path, _truth_dict(labelled_at_utc="2026-08-17T14:02:11")))


def test_rejects_an_unknown_direction(tmp_path: Path) -> None:
    payload = _truth_dict()
    payload["crossings"][0]["direction"] = "sideways"

    with pytest.raises(TruthError):
        load_truth(_write(tmp_path, payload))


def test_rejects_a_negative_timestamp(tmp_path: Path) -> None:
    payload = _truth_dict()
    payload["crossings"][0]["t_s"] = -0.5

    with pytest.raises(TruthError):
        load_truth(_write(tmp_path, payload))


def test_rejects_a_timestamp_past_the_end_of_the_clip(tmp_path: Path) -> None:
    payload = _truth_dict(duration_s=20.0)

    with pytest.raises(TruthError):
        load_truth(_write(tmp_path, payload))


def test_rejects_unsorted_crossings(tmp_path: Path) -> None:
    """Out-of-order marks mean the labelling tool or the labeller lost the plot; scoring
    them as-is would silently misattribute crossings to minutes."""
    payload = _truth_dict()
    payload["crossings"] = [
        {"t_s": 31.2, "line_id": "entrance", "direction": "out"},
        {"t_s": 12.48, "line_id": "entrance", "direction": "in"},
    ]

    with pytest.raises(TruthError):
        load_truth(_write(tmp_path, payload))


def test_accepts_two_crossings_at_the_same_instant(tmp_path: Path) -> None:
    """Two people abreast through a wide door is real. Non-decreasing, not increasing."""
    payload = _truth_dict()
    payload["crossings"] = [
        {"t_s": 12.48, "line_id": "entrance", "direction": "in"},
        {"t_s": 12.48, "line_id": "entrance", "direction": "in"},
    ]

    assert len(load_truth(_write(tmp_path, payload)).crossings) == 2


def test_rejects_an_unknown_schema_version(tmp_path: Path) -> None:
    """Dwell and occupancy arrive as version 2, not as optional fields nobody fills in."""
    with pytest.raises(TruthError):
        load_truth(_write(tmp_path, _truth_dict(schema_version=2)))


def test_rejects_a_missing_clip_id(tmp_path: Path) -> None:
    payload = _truth_dict()
    del payload["clip_id"]

    with pytest.raises(TruthError):
        load_truth(_write(tmp_path, payload))


def test_rejects_unknown_keys(tmp_path: Path) -> None:
    with pytest.raises(TruthError):
        load_truth(_write(tmp_path, _truth_dict(gate_eligible=True)))


def test_accepts_a_file_saved_with_a_utf8_bom(tmp_path: Path) -> None:
    """Windows editors and PowerShell's `Set-Content -Encoding utf8` prepend a byte-order
    mark. It is still UTF-8 and still valid JSON to every other tool; rejecting it would
    fail a labeller's file for a reason they cannot see in an editor."""
    path = tmp_path / "doorway-daylight-01.truth.json"
    path.write_text(json.dumps(_truth_dict()), encoding="utf-8-sig")

    assert load_truth(path).clip_id == "doorway-daylight-01"


# --- Pairing a truth file with the clip it claims to describe ---------------


def test_pairing_accepts_a_matching_manifest(tmp_path: Path) -> None:
    truth = load_truth(_write(tmp_path, _truth_dict()))

    check_pairing(truth, _manifest())


def test_pairing_rejects_a_duration_disagreement(tmp_path: Path) -> None:
    """The labeller watched a different cut of the clip than the one being scored."""
    truth = load_truth(_write(tmp_path, _truth_dict()))

    with pytest.raises(TruthError):
        check_pairing(truth, _manifest(duration_s=1795.0))


def test_pairing_rejects_a_different_clip(tmp_path: Path) -> None:
    truth = load_truth(_write(tmp_path, _truth_dict(clip_id="some-other-clip-01")))

    with pytest.raises(TruthError):
        check_pairing(truth, _manifest())


# --- The labels actually committed to this repository -----------------------


def test_every_committed_truth_file_pairs_with_its_manifest() -> None:
    """Catches the edit that renames a clip, or re-cuts one, and leaves labels behind
    describing footage that no longer exists in that form."""
    truth_files = sorted(FIXTURE_TRUTH.glob("*.truth.json"))

    assert truth_files, "no truth files committed"
    for path in truth_files:
        truth = load_truth(path)
        manifest = load_manifest(FIXTURE_CLIPS / f"{truth.clip_id}.clip.json")
        check_pairing(truth, manifest)
