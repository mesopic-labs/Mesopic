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

import hashlib
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import httpx

from muster.errors import ModelError

DOWNLOAD_TIMEOUT_S = 60.0
"""Explicit, always. A hung fetch on first run must fail rather than wedge the engine."""

MAX_MODEL_BYTES = 256 * 1024 * 1024
"""Nothing we ship is close to this. It bounds a hostile or misconfigured endpoint."""

OPSET = 12
"""ONNX opset the cached artefact is built against. Part of the cache key."""


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Where a model comes from and what licence it carries.

    The licence travels with the artefact all the way to ``/healthz`` so a commercial
    adopter is never surprised about what they are running (ADR-0013).
    """

    name: str
    url: str
    sha256: str
    licence: str
    input_size: int
    """Square input edge the graph was exported at. A property of the artefact, not a
    tunable — feeding a 640 frame to a 416 graph is a shape error, not a slow path."""


# The default detector is permissively licensed, and that is load-bearing rather than
# incidental: MIT engine + Apache-2.0 weights + MIT runtime means the default install
# carries no AGPL and there is no combined work for the §13 network clause to attach to
# (ADR-0013). Ultralytics is reachable only through the opt-in `[ultralytics]` extra.
#
# Apache-2.0 is a hard filter applied *before* the N100 benchmark, never after.
MODELS: dict[str, ModelSpec] = {
    "yolox-nano": ModelSpec(
        name="yolox-nano",
        # Pinned to an immutable release asset, digest verified against the downloaded
        # bytes (3,659,407 B) rather than copied from documentation.
        url=(
            "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/"
            "0.1.1rc0/yolox_nano.onnx"
        ),
        sha256="c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d",
        licence="Apache-2.0",
        input_size=416,
    ),
}

Downloader = Callable[[ModelSpec, Path], None]
"""Fetch a model's FP32 ONNX to ``dest``. Injected so the cache is testable offline."""

Quantizer = Callable[[Path, Path], None]
"""INT8-quantize ``src`` into ``dest``."""


def _download_https(spec: ModelSpec, dest: Path) -> None:
    """Stream a model to ``dest``, verifying its digest before the caller sees it.

    The digest check is not optional and not advisory: an unverified model is arbitrary
    code the inference runtime will happily execute.
    """
    if not spec.url or not spec.sha256:
        message = (
            f"model {spec.name!r} has no verified download URL or checksum; "
            "it cannot be fetched safely"
        )
        raise ModelError(message)

    digest = hashlib.sha256()
    written = 0
    with (
        httpx.stream(
            "GET", spec.url, timeout=DOWNLOAD_TIMEOUT_S, follow_redirects=True
        ) as response,
        dest.open("wb") as handle,
    ):
        response.raise_for_status()
        for chunk in response.iter_bytes():
            written += len(chunk)
            if written > MAX_MODEL_BYTES:
                message = f"model {spec.name!r} exceeded {MAX_MODEL_BYTES} bytes"
                raise ModelError(message)
            digest.update(chunk)
            handle.write(chunk)

    actual = digest.hexdigest()
    if actual != spec.sha256:
        message = f"checksum mismatch for {spec.name!r}: expected {spec.sha256}, got {actual}"
        raise ModelError(message)


def _quantize_int8(src: Path, dest: Path) -> None:
    """Post-training dynamic INT8 quantization.

    Dynamic rather than static: it needs no calibration set, which keeps first run fully
    offline after the weight download and avoids shipping representative imagery we
    would then have to license and store (ADR-0004 — the appliance has no operator).
    """
    # Imported here, not at module scope, because quantization only ever runs on a cache
    # miss. A box with a warm cache — which is every box after first run, and every
    # appliance shipped with the cache pre-warmed — should not pay to import the
    # quantization toolchain (and its `onnx` dependency) just to load a model.
    from onnxruntime.quantization import QuantType, quantize_dynamic  # noqa: PLC0415

    quantize_dynamic(str(src), str(dest), weight_type=QuantType.QUInt8)


def _is_usable(path: Path) -> bool:
    """Whether a cache entry can be loaded, as opposed to merely existing.

    A zero-byte entry is what an earlier run leaves behind if it died between creating
    the file and filling it. Treating it as a cache hit hands the runtime a file that
    fails to load on every frame, forever, with no way out but a manual cache wipe.
    """
    return path.exists() and path.stat().st_size > 0


@dataclass(frozen=True, slots=True)
class ModelArtefact:
    """A cached, ready-to-load model and the licence it came under."""

    path: Path
    model_name: str
    quantization: str
    licence: str


class ModelManager:
    """Owns the model cache directory and the fetch/export/quantize pipeline."""

    def __init__(
        self,
        cache_dir: Path,
        *,
        download: Downloader | None = None,
        quantize: Quantizer | None = None,
    ) -> None:
        self._cache_dir = cache_dir
        self._download = download if download is not None else _download_https
        self._quantize = quantize if quantize is not None else _quantize_int8

    def ensure(self, model_name: str) -> ModelArtefact:
        """Return a cached artefact, building it if absent. Idempotent and atomic."""
        spec = MODELS.get(model_name)
        if spec is None:
            known = ", ".join(sorted(MODELS))
            message = f"unknown model {model_name!r}; known models: {known}"
            raise ModelError(message)

        path = self._artefact_path(spec)
        if not _is_usable(path):
            self._build(spec, path)
        return ModelArtefact(
            path=path,
            model_name=spec.name,
            quantization="int8",
            licence=spec.licence,
        )

    def _artefact_path(self, spec: ModelSpec) -> Path:
        """Content-addressed by everything that changes the bytes.

        Keying on runtime and opset as well as model and quantization means switching
        runtime or upgrading the engine re-quantizes into a *new* file rather than
        clobbering one a running process may still have mapped.
        """
        return self._cache_dir / f"{spec.name}.int8.op{OPSET}.ort.onnx"

    def _build(self, spec: ModelSpec, path: Path) -> None:
        """Download, quantize, and publish atomically.

        Everything happens in a scratch directory alongside the cache; only the final
        ``os.replace`` is visible. A process killed at any point before that leaves the
        cache untouched, which is the property the appliance depends on — there is no
        one there to clear a half-written model by hand.
        """
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=self._cache_dir, prefix=".build-") as scratch:
            scratch_dir = Path(scratch)
            fp32 = scratch_dir / "fp32.onnx"
            int8 = scratch_dir / "int8.onnx"
            self._download(spec, fp32)
            self._quantize(fp32, int8)
            int8.replace(path)
