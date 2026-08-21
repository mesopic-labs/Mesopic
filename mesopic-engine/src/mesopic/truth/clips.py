"""Clip manifests: what the footage is, where it came from, and whether it may gate.

The repository describes clips it does not contain. A manifest is committed; the video it
names is resolved at run time from a directory the operator points at, and verified by
hash. That split is not bookkeeping — this repository is public, and committing footage of
real people would contradict the product's central claim.

**Gate-eligibility is derived, never stored.** There is no ``gate_eligible`` field to set,
because the manifest is closed to unknown keys and the answer is computed from provenance
and consent every time it is asked for. Stock footage carries a copyright licence, not a
model release; footage of people who did not consent cannot back a published accuracy
number, however good the clip looks.

Parsing is deliberately separate from resolving. ``load_manifest`` reads JSON and touches
no video, so the manifest suite runs in CI where no clip exists; ``resolve_clip`` is the
only place bytes are read and the only place a hash is checked.
"""

from __future__ import annotations

import hashlib
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from mesopic.errors import TruthError
from mesopic.truth._json import parse, read_json
from mesopic.types import ClipId

CLIP_ID_PATTERN = r"^[a-z0-9]+(?:-[a-z0-9]+)*$"
"""Lowercase slug. ``clip_id`` becomes a filename in :func:`resolve_clip`, so anything
that could be a path separator, a parent reference, or a drive letter is rejected at the
schema rather than defended against later."""

SHA256_PATTERN = r"^[0-9a-f]{64}$"

MAX_MANIFEST_BYTES = 64 * 1024
"""A manifest is a few hundred bytes of JSON. The limit is here because reading an
attacker-chosen file into memory is the cheapest thing to get wrong."""

_HASH_CHUNK_BYTES = 1024 * 1024
"""Videos are large; hash them in chunks so memory does not scale with clip length."""


class ProvenanceKind(StrEnum):
    """Where the footage came from. Half of the gate-eligibility answer."""

    OWN_RIG = "own_rig"
    """Recorded by us, on our own rig, with consent under our control."""

    PILOT = "pilot"
    """From a pilot deployment, under an agreement that permits this use."""

    STOCK = "stock"
    """Commercially licensed stock. Development fixture only — a stock licence addresses
    copyright and says nothing about whether the people in frame agreed to be there."""

    SYNTHETIC = "synthetic"
    """Generated (``testsrc2`` and friends). No people, no consent question, and no
    ground truth worth labelling — these exist for performance work, not accuracy."""


class ModelRelease(StrEnum):
    """Whether the identifiable people in the clip consented. The other half."""

    OBTAINED = "obtained"
    NOT_REQUIRED = "not_required"
    """No identifiable person is in frame."""

    UNKNOWN = "unknown"
    """Nobody has established this. Treated exactly as badly as "no"."""


class SceneReference(StrEnum):
    """Which reference condition the clip represents.

    Accuracy targets differ per scene, so a clip that claims the wrong one produces a
    number measured against the wrong bar.
    """

    GOOD_DOORWAY = "good_doorway"
    TYPICAL = "typical"
    HARD = "hard"


class TruthModel(BaseModel):
    """Base for every ground-truth artefact: closed to unknown keys, and immutable.

    ``extra="forbid"`` is what keeps ``gate_eligible`` from ever being a field. Someone
    adding it to a JSON file gets a validation error rather than a stored opinion.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


class Provenance(TruthModel):
    """Where the clip came from, and when somebody last checked that in person."""

    kind: ProvenanceKind
    licence: str = Field(min_length=1, max_length=200)
    licence_verified_utc: date
    """The day a human read the licence. A licence nobody checked on a stated day is a
    licence nobody checked."""

    url: str | None = None


class Consent(TruthModel):
    """The consent position for the identifiable people in the clip."""

    model_release: ModelRelease
    note: str | None = Field(default=None, max_length=500)


class Scene(TruthModel):
    """The physical conditions, so a number can be read against the right target."""

    reference: SceneReference
    mount_height_m: float | None = Field(default=None, gt=0.0, le=20.0)
    mount_angle_deg: float | None = Field(default=None, ge=0.0, le=90.0)
    """Degrees below horizontal. The reference envelope wants 30 to 60."""

    lighting: str = Field(min_length=1, max_length=100)


class ClipManifest(TruthModel):
    """A clip this repository describes and does not contain."""

    schema_version: Literal[1]
    clip_id: Annotated[ClipId, Field(pattern=CLIP_ID_PATTERN, max_length=64)]
    sha256: Annotated[str, Field(pattern=SHA256_PATTERN)]
    duration_s: float = Field(gt=0.0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fps: float = Field(gt=0.0)
    provenance: Provenance
    consent: Consent
    scene: Scene


def gate_eligible(manifest: ClipManifest) -> bool:
    """May a released accuracy claim be backed by this clip?

    Derived, never stored. Both halves must hold: the footage must be ours or a
    pilot's, *and* the people in it must have consented or not needed to. Stock footage
    fails the first test no matter how good it looks, and unknown consent fails the second
    — "nobody checked" is treated exactly as badly as "no".
    """
    return manifest.provenance.kind in {
        ProvenanceKind.OWN_RIG,
        ProvenanceKind.PILOT,
    } and manifest.consent.model_release in {
        ModelRelease.OBTAINED,
        ModelRelease.NOT_REQUIRED,
    }


def load_manifest(path: Path) -> ClipManifest:
    """Parse a clip manifest. Reads JSON only — never opens the video it describes."""
    return parse(ClipManifest, read_json(path, MAX_MANIFEST_BYTES), path)


def resolve_clip(manifest: ClipManifest, clips_dir: Path) -> Path:
    """Locate the video this manifest describes, and verify it is that video.

    The only place in the engine that reads clip bytes, and the only place a hash is
    checked. A mismatch is an error rather than a warning: silently scoring against a
    re-encoded clip produces an accuracy number that is precise and meaningless.
    """
    root = clips_dir.resolve()
    candidate = (root / f"{manifest.clip_id}.mp4").resolve()
    if not candidate.is_relative_to(root):
        message = f"clip {manifest.clip_id} resolves outside the clips directory"
        raise TruthError(message)
    if not candidate.is_file():
        message = f"clip {manifest.clip_id} is not in the clips directory"
        raise TruthError(message)

    actual = _sha256_of(candidate)
    if actual != manifest.sha256:
        # Neither hash is a secret, but printing them is noise: the actionable fact is
        # which clip disagreed with its manifest.
        message = f"clip {manifest.clip_id} does not match the sha256 in its manifest"
        raise TruthError(message)
    return candidate


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()
