"""Fetch, export, quantize, and cache the model artefact.

The engine ships **without bundled weights**. A model is a swappable, separately
licensed asset fetched at runtime — which is exactly what keeps the model's licence
separable from the MIT engine code (ADR-0008, ADR-0013).

The lifecycle: download -> export to ONNX -> INT8 post-training quantization -> cache,
content-addressed by `(model, quant, opset, runtime)`. Writes are **atomic**: killed
mid-quantize must leave no half-file, because the appliance cannot be SSH'd into to clean
up. The weight's licence is recorded in cache metadata and surfaced by `muster doctor`
and `/healthz`.

The default path is AGPL-free end to end: Apache-2.0 weights, MIT quantization tools.

Implements P1.4.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ModelArtefact:
    """A cached, ready-to-load model and the licence it came under."""

    path: Path
    model_name: str
    quantization: str
    licence: str


class ModelManager:
    """Owns the model cache directory and the fetch/export/quantize pipeline."""

    def __init__(self, cache_dir: Path) -> None:
        raise NotImplementedError

    def ensure(self, model_name: str) -> ModelArtefact:
        """Return a cached artefact, building it if absent. Idempotent and atomic."""
        raise NotImplementedError
