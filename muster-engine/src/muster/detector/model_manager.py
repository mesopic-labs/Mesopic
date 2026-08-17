"""Fetch and cache the model artefact.

The engine ships **without bundled weights**. A model is a swappable, separately
licensed asset fetched at runtime — which is exactly what keeps the model's licence
separable from the MIT engine code (ADR-0008, ADR-0013).

The lifecycle: download -> verify digest -> transform -> cache, content-addressed by
`(model, format, opset, runtime)`. Writes are **atomic**: a killed build must leave no
half-file, because the appliance cannot be SSH'd into to clean up. The weight's licence
is recorded in cache metadata and surfaced by `muster doctor` and `/healthz`.

The transform step is currently the identity: the artefact is the downloaded FP32 graph.
`_DYNAMIC_QUANTIZATION_NOTE` records why the INT8 step that used to live here was
removed, and what a correct replacement needs.

The default path is AGPL-free end to end: Apache-2.0 weights, MIT runtime.

Implements P1.4.
"""

from __future__ import annotations

import hashlib
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import httpx

from muster.detector.runtime import DEFAULT_RUNTIME, artefact_tag
from muster.errors import ModelError
from muster.types import Runtime

DEFAULT_MODEL = "yolox-nano"
"""The Apache-2.0 default (ADR-0013). Provisional until P1.7's N100 table lands."""

MODEL_CACHE_ENV_VAR = "MUSTER_MODEL_CACHE"
DEFAULT_MODEL_CACHE = Path.home() / ".cache" / "muster" / "models"
"""Where artefacts land. Here rather than in `cli`, because the CLI is not the only
composition root any more: a camera worker builds a detector without going through it.
"""

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

Transform = Callable[[Path, Path], None]
"""Turn the downloaded FP32 graph at ``src`` into the artefact published at ``dest``.

A seam rather than a straight copy because a *correct* INT8 step belongs here when one
exists (see ``_DYNAMIC_QUANTIZATION_NOTE``), and because the atomicity tests need a
step that can be made to fail partway.
"""

_DYNAMIC_QUANTIZATION_NOTE = """
Why this pipeline publishes FP32 and does not quantize.

Until 2026-08-12 this step ran `quantize_dynamic(..., QuantType.QUInt8)`. It was chosen
because dynamic quantization needs no calibration set, which kept first run offline and
avoided shipping representative imagery we would have to license and store.

It does not work on this graph. Dynamic quantization quantizes weights for MatMul-shaped
ops and leaves convolution activations uncalibrated, so a conv-heavy detector comes out
producing noise: on a 1080p frame containing one clearly visible person, the INT8
artefact's best person score was 0.0023 against the FP32 graph's 0.725 — no detection at
any threshold. The engine had therefore never detected anything, and nothing caught it
because the detector tests use fakes and the only stream available locally was a
synthetic pattern with no people in it.

Correct INT8 here means *static* quantization, which needs calibration frames. The
options, none free: ship licensed imagery (the problem dynamic quantization was chosen
to dodge), or calibrate on the box's own camera at first run — representative by
construction and licence-free, but it makes the artefact box-specific and breaks the
content-addressed cache, so it is an ADR-level decision rather than a code change.

FP32 is ~3.5 MB against ~1 MB and, measured locally, was *faster* than the broken INT8
path, which paid dequantize overhead on every convolution. Revisit against P1.7's N100
table, not against intuition.
"""


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


def _publish_unchanged(src: Path, dest: Path) -> None:
    """Publish the downloaded FP32 graph as the artefact, byte for byte.

    The engine ships the weights exactly as the pinned release asset provides them, whose
    digest `_download_https` has already verified. See `_DYNAMIC_QUANTIZATION_NOTE` for
    why there is no quantization step here and what a correct one would require.
    """
    src.replace(dest)


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
        transform: Transform | None = None,
    ) -> None:
        self._cache_dir = cache_dir
        self._download = download if download is not None else _download_https
        self._transform = transform if transform is not None else _publish_unchanged

    def ensure(self, model_name: str, *, runtime: Runtime = DEFAULT_RUNTIME) -> ModelArtefact:
        """Return a cached artefact, building it if absent. Idempotent and atomic."""
        spec = MODELS.get(model_name)
        if spec is None:
            known = ", ".join(sorted(MODELS))
            message = f"unknown model {model_name!r}; known models: {known}"
            raise ModelError(message)

        path = self._artefact_path(spec, runtime)
        if not _is_usable(path):
            self._build(spec, path)
        return ModelArtefact(
            path=path,
            model_name=spec.name,
            quantization="fp32",
            licence=spec.licence,
        )

    def _artefact_path(self, spec: ModelSpec, runtime: Runtime) -> Path:
        """Content-addressed by everything that changes the bytes.

        Keying on runtime and opset as well as model and quantization means switching
        runtime or upgrading the engine re-quantizes into a *new* file rather than
        clobbering one a running process may still have mapped.

        Runtimes that consume identical bytes share an entry, though: ORT and OpenVINO
        both load the same quantized ONNX graph, and separating them would buy a
        re-download that produces the same file (ADR-0012). The tag names the artefact's
        *format*, so a runtime that genuinely needs its own — a Coral `.tflite`, a
        TensorRT engine — still gets one.
        """
        return self._cache_dir / f"{spec.name}.{artefact_tag(runtime)}.op{OPSET}.onnx"

    def _build(self, spec: ModelSpec, path: Path) -> None:
        """Download, transform, and publish atomically.

        Everything happens in a scratch directory alongside the cache; only the final
        ``os.replace`` is visible. A process killed at any point before that leaves the
        cache untouched, which is the property the appliance depends on — there is no
        one there to clear a half-written model by hand.
        """
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=self._cache_dir, prefix=".build-") as scratch:
            scratch_dir = Path(scratch)
            downloaded = scratch_dir / "downloaded.onnx"
            artefact = scratch_dir / "artefact.onnx"
            self._download(spec, downloaded)
            self._transform(downloaded, artefact)
            artefact.replace(path)
