"""Ground truth: what actually happened in a clip, and what the engine said happened.

This package handles labels, never pixels. An import-linter contract forbids it from
reaching for ``muster.ingest``, ``av`` or ``cv2``, so "the truth set is numbers" is a
property of the dependency graph rather than a convention.

Public surface:

* :class:`ClipManifest` / :func:`load_manifest` — what a clip is and where it came from
* :func:`gate_eligible` — may this footage back a released accuracy claim?
* :func:`resolve_clip` — find the video and verify it is the one the manifest names
* :class:`TruthFile` / :func:`load_truth` — what a human saw happen in it
* :func:`check_pairing` — confirm the labels and the clip describe the same footage
* :func:`footfall_per_minute`, :func:`mape`, :func:`score` — the accuracy arithmetic
"""

from __future__ import annotations

from muster.truth.clips import (
    ClipManifest,
    Consent,
    ModelRelease,
    Provenance,
    ProvenanceKind,
    Scene,
    SceneReference,
    gate_eligible,
    load_manifest,
    resolve_clip,
)
from muster.truth.labels import Crossing, TruthFile, check_pairing, load_truth
from muster.truth.score import Score, footfall_per_minute, mape, score

__all__ = [
    "ClipManifest",
    "Consent",
    "Crossing",
    "ModelRelease",
    "Provenance",
    "ProvenanceKind",
    "Scene",
    "SceneReference",
    "Score",
    "TruthFile",
    "check_pairing",
    "footfall_per_minute",
    "gate_eligible",
    "load_manifest",
    "load_truth",
    "mape",
    "resolve_clip",
    "score",
]
