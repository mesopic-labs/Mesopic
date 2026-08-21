"""OpenVINO wearing ONNX Runtime's session interface (ADR-0012).

This is an adapter, not a second detector. `OnnxDetector` talks to a two-method
`InferenceSession` protocol, so presenting OpenVINO through that protocol means the
letterbox → infer → decode path is the *same code* on both runtimes — no parallel
implementation to keep in step, and no second place for the arithmetic to drift.

The model file is the same quantized ONNX graph either way; OpenVINO consumes ONNX
directly, so nothing is re-exported and nothing is cached twice.

The native `openvino` package is used rather than ONNX Runtime's OpenVINO execution
provider: that provider ships in the `onnxruntime-openvino` wheel, which supplies the same
`onnxruntime` module as the plain wheel the engine already depends on, and installing both
is a broken environment.

Implements ADR-0012.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from mesopic.errors import ModelError

if TYPE_CHECKING:
    from pathlib import Path

DEVICE = "CPU"
"""Not ``AUTO``: it may migrate the graph to an iGPU mid-run, which would move the
numerics under a perf measurement. Selecting an iGPU is its own decision, not this one."""

THREAD_CONFIG_KEY = "INFERENCE_NUM_THREADS"
"""OpenVINO's counterpart to ORT's ``intra_op_num_threads``, and it exists for the same
reason: each camera is its own process, and unbounded pools fight over the same cores."""


@dataclass(frozen=True, slots=True)
class InputInfo:
    """The two attributes `OnnxDetector` reads off a graph input."""

    name: str
    shape: list[int | str]


class OpenVinoSession:
    """An OpenVINO compiled model presented as an `InferenceSession`."""

    def __init__(self, model_path: Path, *, num_threads: int | None = None) -> None:
        # Imported here, not at module scope, because `openvino` is an opt-in extra that
        # most boxes will not have. Importing it eagerly would make this module unusable
        # to import on the default install, which is the opposite of opportunistic.
        try:
            import openvino  # noqa: PLC0415

            config: dict[str, Any] = {} if num_threads is None else {THREAD_CONFIG_KEY: num_threads}
            self._compiled = openvino.Core().compile_model(
                str(model_path), device_name=DEVICE, config=config
            )
        except Exception as exc:  # OpenVINO raises a wide, unstable set of types here
            message = f"could not load model {model_path.name!r} on openvino: {exc}"
            raise ModelError(message) from exc

    def get_inputs(self) -> list[Any]:
        """Describe the graph's inputs in the shape ORT would report them.

        A dynamic axis is surfaced as a non-integer rather than raised on here, so the
        detector's existing "not a fixed NCHW square" guard produces one error for the
        problem instead of each runtime inventing its own.
        """
        return [
            InputInfo(
                name=node.get_any_name(),
                shape=[dim.get_length() if dim.is_static else "?" for dim in node.partial_shape],
            )
            for node in self._compiled.inputs
        ]

    def run(self, output_names: list[str] | None, input_feed: dict[str, Any]) -> list[Any]:
        """Infer one batch. `output_names` is accepted for protocol parity and unused.

        The caller always passes `None`, and the pinned detection graph has a single
        output that `decode_yolox_output` reads as `outputs[0]`.
        """
        del output_names
        results = self._compiled(input_feed)
        return [results[self._compiled.output(0)]]
