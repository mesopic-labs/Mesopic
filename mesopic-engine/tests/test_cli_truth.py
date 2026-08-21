"""`mesopic truth validate` — the command a labeller runs before opening a pull request.

Its job is to make a bad label file cheap to find and a mislabelled *provenance* loud.
The gate-eligibility line in the output is the point: someone committing stock footage as
own-rig footage should see the word "no" printed at them long before a release does.

Written red-first.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from mesopic import cli

RUNNER = CliRunner()

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_CLIPS = REPO_ROOT / "fixtures" / "clips"


def _manifest_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "clip_id": "doorway-daylight-01",
        "sha256": "a" * 64,
        "duration_s": 180.0,
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
    payload.update(overrides)
    return payload


def _truth_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "clip_id": "doorway-daylight-01",
        "labelled_by": "mark",
        "labelled_at_utc": "2026-08-17T14:02:11Z",
        "duration_s": 180.0,
        "crossings": [{"t_s": 12.48, "line_id": "entrance", "direction": "in"}],
    }
    payload.update(overrides)
    return payload


def _write(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_validates_a_good_manifest(tmp_path: Path) -> None:
    path = _write(tmp_path / "doorway-daylight-01.clip.json", _manifest_payload())

    result = RUNNER.invoke(cli.app, ["truth", "validate", str(path)])

    assert result.exit_code == 0
    assert "ok" in result.stdout


def test_reports_gate_eligibility_for_a_manifest(tmp_path: Path) -> None:
    path = _write(tmp_path / "doorway-daylight-01.clip.json", _manifest_payload())

    result = RUNNER.invoke(cli.app, ["truth", "validate", str(path)])

    assert "gate-eligible: yes" in result.stdout


def test_says_no_out_loud_for_stock_footage(tmp_path: Path) -> None:
    payload = _manifest_payload()
    payload["provenance"]["kind"] = "stock"
    path = _write(tmp_path / "stock-01.clip.json", payload)

    result = RUNNER.invoke(cli.app, ["truth", "validate", str(path)])

    assert result.exit_code == 0
    assert "gate-eligible: no" in result.stdout


def test_an_eligible_clip_in_the_wrong_scene_does_not_read_as_a_clean_yes(
    tmp_path: Path,
) -> None:
    """The own-rig home clips are gate-eligible and `hard`. A bare "yes" invites exactly
    the misreading this line exists to prevent, so the scene is named alongside it."""
    payload = _manifest_payload()
    payload["scene"]["reference"] = "hard"
    path = _write(tmp_path / "home-01.clip.json", payload)

    result = RUNNER.invoke(cli.app, ["truth", "validate", str(path)])

    assert result.exit_code == 0
    assert "gate-eligible: yes" in result.stdout
    assert "hard" in result.stdout


def test_a_draft_truth_file_says_it_cannot_gate(tmp_path: Path) -> None:
    payload = _truth_payload()
    payload["labelled_by"] = "draft-unverified"
    path = _write(tmp_path / "doorway-daylight-01.truth.json", payload)

    result = RUNNER.invoke(cli.app, ["truth", "validate", str(path)])

    assert result.exit_code == 0
    assert "cannot gate" in result.stdout


def test_validates_a_good_truth_file(tmp_path: Path) -> None:
    path = _write(tmp_path / "doorway-daylight-01.truth.json", _truth_payload())

    result = RUNNER.invoke(cli.app, ["truth", "validate", str(path)])

    assert result.exit_code == 0
    assert "1 crossing" in result.stdout


def test_a_bad_file_exits_non_zero(tmp_path: Path) -> None:
    path = _write(tmp_path / "doorway-daylight-01.truth.json", _truth_payload(schema_version=7))

    result = RUNNER.invoke(cli.app, ["truth", "validate", str(path)])

    assert result.exit_code == 1


def test_validates_several_files_and_fails_on_any(tmp_path: Path) -> None:
    good = _write(tmp_path / "a-01.clip.json", _manifest_payload(clip_id="a-01"))
    bad = _write(tmp_path / "b-01.truth.json", _truth_payload(schema_version=7))

    result = RUNNER.invoke(cli.app, ["truth", "validate", str(good), str(bad)])

    assert result.exit_code == 1
    assert "a-01" in result.stdout


def test_an_unrecognised_suffix_is_an_error(tmp_path: Path) -> None:
    path = _write(tmp_path / "notes.json", _manifest_payload())

    result = RUNNER.invoke(cli.app, ["truth", "validate", str(path)])

    assert result.exit_code == 1


def test_the_committed_fixture_manifests_validate() -> None:
    """The command a reviewer runs to check this repository's own fixtures."""
    paths = [str(path) for path in sorted(FIXTURE_CLIPS.glob("*.clip.json"))]

    result = RUNNER.invoke(cli.app, ["truth", "validate", *paths])

    assert result.exit_code == 0
