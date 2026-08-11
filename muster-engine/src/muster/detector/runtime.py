"""Which native runtime executes the detector's graph (ADR-0012).

ORT-CPU is the guaranteed path: a hard dependency, portable across Intel, AMD and ARM, and
the path the N100 perf gate is measured against. Everything else is an accelerator —
selected when asked for, never required, and its absence is a normal condition rather than
an error (ADR-0003).

Auto-selection deliberately resolves to ORT-CPU even where an accelerator is present.
ADR-0012 gates promoting one to the default on a measured win at equal accuracy, and that
measurement is owed by the perf harness (implementation-plan P1.7). Choosing on anything
less would also mean the M0 gate is measured on a path most boxes will not run. The policy
lives in `resolve_runtime` alone, so promoting an accelerator later is a change to one
function — and to the ADR.

Deciding *what was asked for* is kept separate from probing *whether it can be honoured*:
that split is what makes the whole selection path testable on a box with no accelerator.

Implements ADR-0012.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from muster.errors import ModelError
from muster.types import Runtime

if TYPE_CHECKING:
    from collections.abc import Mapping

DEFAULT_RUNTIME: Final = Runtime.ORT_CPU
"""The guaranteed path. What an operator gets when they express no preference."""

RUNTIME_ENV_VAR: Final = "MUSTER_DETECTOR_RUNTIME"
"""Operator-facing override. P2.1 will route `muster.yaml` to the same resolution."""

_ARTEFACT_TAGS: Final[dict[Runtime, str]] = {
    # Both runtimes load the same quantized ONNX graph, so they share a cache entry.
    # Keying them apart would force a re-download and re-quantize to produce identical
    # bytes. A runtime needing its own artefact — a Coral `.tflite`, a TensorRT engine —
    # returns a different tag here and gets its own file, which is the collision
    # ADR-0012's content-addressing exists to prevent.
    Runtime.ORT_CPU: "int8",
    Runtime.OPENVINO: "int8",
}


@dataclass(frozen=True, slots=True)
class RuntimeSelection:
    """Which runtime to use, and whether the operator named it.

    The distinction is the entire fallback policy: an explicit request that cannot be
    honoured is an error, while an automatic one degrades quietly to the guaranteed path.
    An operator who asked for an accelerator and silently got CPU has a performance
    mystery with nothing anywhere to explain it.
    """

    runtime: Runtime
    explicit: bool


def artefact_tag(runtime: Runtime) -> str:
    """The cache-key segment naming the *format* of the bytes this runtime consumes."""
    return _ARTEFACT_TAGS[runtime]


def is_available(runtime: Runtime) -> bool:
    """Whether this box can actually execute a graph on `runtime`."""
    if runtime is Runtime.ORT_CPU:
        return True

    try:
        # Imported here, not at module scope, because the extra is absent on most boxes
        # and its absence is the expected answer. A box that never opted in should not
        # pay an import to be told so.
        import openvino  # noqa: PLC0415

        devices = openvino.Core().available_devices
    except Exception:  # noqa: BLE001 - a probe must never propagate; no accelerator is not an error
        return False
    # A package that imports but enumerates no device is not a runtime we can use.
    return bool(devices)


def available_runtimes() -> tuple[Runtime, ...]:
    """Every runtime this box could execute, in declaration order.

    What `muster doctor` renders, and the answer to "why is this box slower than that
    one" before anyone reaches for a profiler.
    """
    return tuple(runtime for runtime in Runtime if is_available(runtime))


def resolve_runtime(
    requested: Runtime | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> RuntimeSelection:
    """Decide which runtime was asked for. Does not probe, and does not fall back.

    `env` is injectable so the resolution is testable without mutating the process
    environment.
    """
    if requested is not None:
        return RuntimeSelection(requested, explicit=True)

    source = os.environ if env is None else env
    # An exported-but-empty variable is how a shell says "no value", not a request for a
    # runtime whose name is the empty string.
    raw = source.get(RUNTIME_ENV_VAR, "").strip()
    if not raw:
        return RuntimeSelection(DEFAULT_RUNTIME, explicit=False)

    try:
        runtime = Runtime(raw.lower())
    except ValueError:
        known = ", ".join(sorted(candidate.value for candidate in Runtime))
        message = f"unknown detector runtime {raw!r}; known runtimes: {known}"
        raise ModelError(message) from None
    return RuntimeSelection(runtime, explicit=True)
