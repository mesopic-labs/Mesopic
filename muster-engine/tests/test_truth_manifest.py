"""A clip manifest describes footage the repository does not contain.

The load-bearing property here is negative: **no released accuracy claim may be backed
by footage of people who did not consent to it.** That is enforced by derivation rather
than by a field — there is no ``gate_eligible`` key a hurried afternoon could set to
``true`` — so these tests are the thing that makes the rule real rather than documentary.

The second property is that parsing a manifest never touches a video. The clip bytes
live outside the repository, so if ``load_manifest`` needed them, this whole suite would
be a nightly job instead of a per-PR gate.

Written red-first.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from muster.errors import TruthError
from muster.truth import gate_eligible, load_manifest, resolve_clip
from muster.types import ClipId

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_CLIPS = REPO_ROOT / "fixtures" / "clips"


def _manifest_dict(**overrides: Any) -> dict[str, Any]:
    """A minimal valid manifest. Tests override exactly the key under test."""
    base: dict[str, Any] = {
        "schema_version": 1,
        "clip_id": "doorway-daylight-01",
        "sha256": "a" * 64,
        "duration_s": 1800.0,
        "width": 1920,
        "height": 1080,
        "fps": 25.0,
        "provenance": {
            "kind": "own_rig",
            "licence": "Owned — recorded by us",
            "licence_verified_utc": "2026-08-16",
            "url": None,
        },
        "consent": {"model_release": "obtained", "note": "signed, held offline"},
        "scene": {
            "reference": "good_doorway",
            "mount_height_m": 2.8,
            "mount_angle_deg": 42.0,
            "lighting": "even_daylight",
        },
    }
    base.update(overrides)
    return base


def _write(tmp_path: Path, payload: dict[str, Any], name: str = "clip.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# --- Gate-eligibility, the rule this file exists for -------------------------


def test_own_rig_with_consent_is_gate_eligible(tmp_path: Path) -> None:
    manifest = load_manifest(_write(tmp_path, _manifest_dict()))

    assert gate_eligible(manifest) is True


def test_pilot_footage_with_consent_is_gate_eligible(tmp_path: Path) -> None:
    payload = _manifest_dict()
    payload["provenance"]["kind"] = "pilot"
    manifest = load_manifest(_write(tmp_path, payload))

    assert gate_eligible(manifest) is True


def test_consent_not_required_is_gate_eligible(tmp_path: Path) -> None:
    """Footage with no identifiable person needs no release to be honest about."""
    payload = _manifest_dict()
    payload["consent"] = {"model_release": "not_required", "note": "no people in frame"}
    manifest = load_manifest(_write(tmp_path, payload))

    assert gate_eligible(manifest) is True


def test_stock_footage_is_never_gate_eligible(tmp_path: Path) -> None:
    """However good it looks. The stock licence covers copyright, never consent."""
    payload = _manifest_dict()
    payload["provenance"]["kind"] = "stock"
    payload["consent"] = {"model_release": "obtained", "note": "claimed by the uploader"}
    manifest = load_manifest(_write(tmp_path, payload))

    assert gate_eligible(manifest) is False


def test_synthetic_footage_is_never_gate_eligible(tmp_path: Path) -> None:
    payload = _manifest_dict()
    payload["provenance"]["kind"] = "synthetic"
    payload["consent"] = {"model_release": "not_required", "note": "generated"}
    manifest = load_manifest(_write(tmp_path, payload))

    assert gate_eligible(manifest) is False


def test_unknown_consent_is_never_gate_eligible(tmp_path: Path) -> None:
    payload = _manifest_dict()
    payload["consent"] = {"model_release": "unknown", "note": None}
    manifest = load_manifest(_write(tmp_path, payload))

    assert gate_eligible(manifest) is False


def test_manifest_has_no_gate_eligible_field(tmp_path: Path) -> None:
    """A stored boolean can be set to ``true``. A derived one cannot."""
    payload = _manifest_dict()
    payload["gate_eligible"] = True

    with pytest.raises(TruthError):
        load_manifest(_write(tmp_path, payload))


# --- Parsing is strict, and never touches a video ---------------------------


def test_rejects_unknown_schema_version(tmp_path: Path) -> None:
    with pytest.raises(TruthError):
        load_manifest(_write(tmp_path, _manifest_dict(schema_version=2)))


def test_rejects_unknown_provenance_kind(tmp_path: Path) -> None:
    payload = _manifest_dict()
    payload["provenance"]["kind"] = "borrowed"

    with pytest.raises(TruthError):
        load_manifest(_write(tmp_path, payload))


def test_rejects_non_positive_duration(tmp_path: Path) -> None:
    with pytest.raises(TruthError):
        load_manifest(_write(tmp_path, _manifest_dict(duration_s=0.0)))


def test_rejects_malformed_sha256(tmp_path: Path) -> None:
    with pytest.raises(TruthError):
        load_manifest(_write(tmp_path, _manifest_dict(sha256="not-a-hash")))


def test_rejects_clip_id_that_is_not_a_slug(tmp_path: Path) -> None:
    """``clip_id`` becomes a filename in ``resolve_clip``; a path separator in it is how
    that turns into a traversal."""
    with pytest.raises(TruthError):
        load_manifest(_write(tmp_path, _manifest_dict(clip_id="../../etc/passwd")))


def test_load_manifest_does_not_need_the_video(tmp_path: Path) -> None:
    """The whole point of splitting parse from resolve: CI has no clips on disk."""
    manifest = load_manifest(_write(tmp_path, _manifest_dict()))

    assert manifest.clip_id == ClipId("doorway-daylight-01")
    assert manifest.duration_s == 1800.0


# --- Resolving the video, the one place bytes are verified ------------------


def test_resolve_clip_returns_the_path_when_the_hash_matches(tmp_path: Path) -> None:
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    video = clips_dir / "doorway-daylight-01.mp4"
    video.write_bytes(b"not really a video, but it hashes")
    payload = _manifest_dict(sha256=_sha256_of(video))
    manifest = load_manifest(_write(tmp_path, payload))

    assert resolve_clip(manifest, clips_dir) == video


def test_resolve_clip_raises_on_a_hash_mismatch(tmp_path: Path) -> None:
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    (clips_dir / "doorway-daylight-01.mp4").write_bytes(b"a different clip entirely")
    manifest = load_manifest(_write(tmp_path, _manifest_dict()))

    with pytest.raises(TruthError):
        resolve_clip(manifest, clips_dir)


def test_resolve_clip_raises_when_the_clip_is_absent(tmp_path: Path) -> None:
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    manifest = load_manifest(_write(tmp_path, _manifest_dict()))

    with pytest.raises(TruthError):
        resolve_clip(manifest, clips_dir)


def _sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- The manifests actually committed to this repository --------------------


def test_every_committed_manifest_parses() -> None:
    manifests = sorted(FIXTURE_CLIPS.glob("*.clip.json"))

    assert manifests, "no clip manifests committed"
    for path in manifests:
        load_manifest(path)


def test_committed_stock_manifests_are_not_gate_eligible() -> None:
    """The development fixture is stock footage, and stock can never gate a release."""
    for path in sorted(FIXTURE_CLIPS.glob("*.clip.json")):
        manifest = load_manifest(path)
        if manifest.provenance.kind == "stock":
            assert gate_eligible(manifest) is False


def test_committed_manifests_record_a_licence_verification_date() -> None:
    """A licence that was never checked on a stated day is a licence nobody checked."""
    for path in sorted(FIXTURE_CLIPS.glob("*.clip.json")):
        manifest = load_manifest(path)
        assert isinstance(manifest.provenance.licence_verified_utc, date)
