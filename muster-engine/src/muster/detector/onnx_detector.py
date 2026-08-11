"""Detection via ONNX Runtime, CPU execution provider by default (ADR-0012).

ORT-CPU is the portable default and always works. OpenVINO is available as an explicit
opt-in on Intel boxes and is pure upside, never a dependency (ADR-0003) — it is not
auto-selected, because promoting an accelerator to the default is gated on a measured win
the perf harness has yet to produce. Coral and CUDA/TensorRT are deferred by ADR-0012;
when they arrive they arrive behind the same `InferenceSession` seam, changing nothing
below.

Performance levers that belong here and nowhere else: input size, ROI crop to the union
of the camera's configured geometry, INT8 quantization, and thread counts.

This class is deliberately thin — session in, boxes out. The arithmetic that decides
whether the counts are right lives in `postprocess`, where it can be tested without a
model file.

Implements P1.4.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from muster.detector.openvino_session import OpenVinoSession
from muster.detector.postprocess import decode_yolox_output, letterbox
from muster.detector.runtime import DEFAULT_RUNTIME, RuntimeSelection, is_available, resolve_runtime
from muster.errors import ModelError
from muster.types import DecodedFrame, Detection, Runtime

if TYPE_CHECKING:
    from numpy.typing import NDArray

DEFAULT_CONFIDENCE = 0.35
DEFAULT_IOU_THRESHOLD = 0.45
_SPATIAL_AXES = 2
"""An NCHW input has exactly two spatial axes. Anything else is not an image graph."""


class InferenceSession(Protocol):
    """The slice of an ORT session this detector actually uses.

    Narrow on purpose: it is what lets a test supply a fake in ten lines, and what would
    let a different runtime slot in without touching this file (ADR-0012).
    """

    def get_inputs(self) -> list[Any]: ...

    def run(self, output_names: list[str] | None, input_feed: dict[str, Any]) -> list[Any]: ...


class OnnxDetector:
    """A `Detector` backed by an ONNX graph on ONNX Runtime."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        confidence: float = DEFAULT_CONFIDENCE,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        intra_op_threads: int | None = None,
        runtime: Runtime | None = None,
        session: InferenceSession | None = None,
    ) -> None:
        """`runtime` is ignored when `session` is injected — that session is already one."""
        self._confidence = confidence
        self._iou_threshold = iou_threshold

        if session is not None:
            self._session: InferenceSession | None = session
            self._runtime = DEFAULT_RUNTIME
        else:
            self._session, self._runtime = _open_session(
                Path(model_path),
                intra_op_threads=intra_op_threads,
                selection=resolve_runtime(runtime),
            )

        model_input = self._session.get_inputs()[0]
        self._input_name: str = model_input.name
        self._input_size = _square_input_size(model_input.shape)

    @property
    def input_size(self) -> int:
        """The edge length the graph was exported at, read from the graph itself.

        Not a tunable. Changing the input size means re-exporting the model, and
        guessing it wrong is a shape error on the first frame rather than a slow path.
        """
        return self._input_size

    @property
    def runtime(self) -> Runtime:
        """Which runtime is actually executing the graph — never merely the one requested.

        What `muster doctor` and `/healthz` report, and the first answer to "why is this
        box slower than that one".
        """
        return self._runtime

    def detect(self, frame: DecodedFrame) -> list[Detection]:
        """Return person detections in the frame's pixel space, person class only."""
        if self._session is None:
            message = "detector is closed"
            raise RuntimeError(message)

        tensor, ratio = letterbox(frame.image, self._input_size)
        outputs = self._session.run(None, {self._input_name: tensor})

        raw: NDArray[Any] = outputs[0]
        return decode_yolox_output(
            raw,
            input_size=self._input_size,
            ratio=ratio,
            frame_width=frame.width,
            frame_height=frame.height,
            confidence=self._confidence,
            iou_threshold=self._iou_threshold,
        )

    def close(self) -> None:
        """Release the inference session."""
        self._session = None


def _open_session(
    model_path: Path,
    *,
    intra_op_threads: int | None,
    selection: RuntimeSelection,
) -> tuple[InferenceSession, Runtime]:
    """Open the graph on the selected runtime, reporting which one actually opened it.

    An explicit request is a promise: if the operator named a runtime and it cannot be
    honoured, that is an error rather than a quiet downgrade to CPU, because a box slower
    than the one someone configured is otherwise undiagnosable. An automatic selection
    carries no such promise and degrades to the guaranteed path (ADR-0012).
    """
    if selection.runtime is Runtime.ORT_CPU:
        return _open_ort_cpu_session(model_path, intra_op_threads=intra_op_threads), Runtime.ORT_CPU

    if not is_available(selection.runtime):
        if selection.explicit:
            message = (
                f"runtime {selection.runtime.value!r} was requested but is not available "
                "on this box; install the matching extra, or leave the runtime unset to "
                "use the CPU baseline"
            )
            raise ModelError(message)
        return _fall_back_to_cpu(model_path, intra_op_threads=intra_op_threads)

    try:
        return OpenVinoSession(model_path, num_threads=intra_op_threads), selection.runtime
    except ModelError:
        # Available but unloadable is the same question as unavailable, one step later.
        if selection.explicit:
            raise
    return _fall_back_to_cpu(model_path, intra_op_threads=intra_op_threads)


def _fall_back_to_cpu(
    model_path: Path, *, intra_op_threads: int | None
) -> tuple[InferenceSession, Runtime]:
    """An accelerator is upside; upside that failed to materialise is still a working box."""
    return _open_ort_cpu_session(model_path, intra_op_threads=intra_op_threads), Runtime.ORT_CPU


def _open_ort_cpu_session(model_path: Path, *, intra_op_threads: int | None) -> InferenceSession:
    """Load the graph on the CPU execution provider.

    Imported lazily so that constructing a detector with an injected session — which is
    what the fast tests do — does not pay to import the runtime.
    """
    import onnxruntime as ort  # noqa: PLC0415

    options = ort.SessionOptions()
    if intra_op_threads is not None:
        # Each camera is its own process (engine §9). Left unbounded, every worker sizes
        # its thread pool to the whole box and they fight each other for the same cores.
        options.intra_op_num_threads = intra_op_threads

    try:
        session: InferenceSession = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
    except Exception as exc:  # ORT raises a wide, unstable set of types here
        message = f"could not load model {model_path.name!r}: {exc}"
        raise ModelError(message) from exc
    return session


def _square_input_size(shape: list[Any]) -> int:
    """Pull the spatial edge out of an NCHW input shape.

    A dynamic axis comes back as a string or None, which we cannot letterbox against —
    better to say so at construction than to fail on every frame.
    """
    spatial = shape[2:]
    if len(spatial) != _SPATIAL_AXES or not all(isinstance(axis, int) for axis in spatial):
        message = f"model input shape {shape!r} is not a fixed NCHW square"
        raise ModelError(message)
    if spatial[0] != spatial[1]:
        message = f"model input {shape!r} is not square; letterboxing assumes it is"
        raise ModelError(message)
    return int(spatial[0])
