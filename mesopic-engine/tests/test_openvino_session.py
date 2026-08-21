"""The OpenVINO adapter, driven by a fake `openvino` module.

`openvino` is an opt-in extra and is absent from the dev environment and from CI, so every
fast test here injects a fake `Core` into `sys.modules` — the same trick as
`test_onnx_detector`'s `FakeSession`, one layer further out. The fake mirrors the shape of
the real API (`Core.compile_model`, `CompiledModel.inputs`, `.output(0)`, and calling the
compiled model) so the adapter is exercised rather than merely imported.

What this cannot prove is that the real OpenVINO binding matches that shape. The `slow`
test at the bottom is what proves it, and it skips wherever the extra is not installed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from mesopic.detector.onnx_detector import OnnxDetector
from mesopic.detector.openvino_session import OpenVinoSession
from mesopic.errors import ModelError
from mesopic.types import Runtime

INPUT_SIZE = 416


class _FakeDim:
    """One dimension of an OpenVINO `PartialShape`."""

    def __init__(self, length: int | None) -> None:
        self._length = length

    @property
    def is_static(self) -> bool:
        return self._length is not None

    def get_length(self) -> int:
        assert self._length is not None
        return self._length


class _FakeNode:
    def __init__(self, name: str, shape: list[int | None]) -> None:
        self._name = name
        self.partial_shape = [_FakeDim(dim) for dim in shape]

    def get_any_name(self) -> str:
        return self._name


class _FakeCompiledModel:
    def __init__(self, name: str, shape: list[int | None], output: np.ndarray) -> None:
        self.inputs = [_FakeNode(name, shape)]
        self._output_key = object()
        self._output = output
        self.feeds: list[dict[str, Any]] = []

    def output(self, index: int) -> object:
        assert index == 0
        return self._output_key

    def __call__(self, input_feed: dict[str, Any]) -> dict[object, np.ndarray]:
        self.feeds.append(input_feed)
        return {self._output_key: self._output}


class _FakeCore:
    """Records what it was asked to compile, so config assertions are possible."""

    last: _FakeCore | None = None

    def __init__(
        self,
        *,
        shape: list[int | None] | None = None,
        output: np.ndarray | None = None,
        explode: bool = False,
    ) -> None:
        default_shape: list[int | None] = [1, 3, INPUT_SIZE, INPUT_SIZE]
        self._shape = shape if shape is not None else default_shape
        self._output = output if output is not None else np.zeros((1, 8, 85), dtype=np.float32)
        self._explode = explode
        self.compiled_with: dict[str, Any] = {}

    def compile_model(
        self, model: str, device_name: str, config: dict[str, Any] | None = None
    ) -> _FakeCompiledModel:
        if self._explode:
            message = "no plugin for device"
            raise RuntimeError(message)
        self.compiled_with = {"model": model, "device_name": device_name, "config": config}
        return _FakeCompiledModel("images", self._shape, self._output)


def _install_fake_openvino(
    monkeypatch: pytest.MonkeyPatch,
    *,
    shape: list[int | None] | None = None,
    output: np.ndarray | None = None,
    explode: bool = False,
) -> list[_FakeCore]:
    """Put a fake `openvino` in `sys.modules` and hand back the cores it constructs."""
    built: list[_FakeCore] = []

    def core_factory() -> _FakeCore:
        core = _FakeCore(shape=shape, output=output, explode=explode)
        built.append(core)
        return core

    monkeypatch.setitem(sys.modules, "openvino", SimpleNamespace(Core=core_factory))
    return built


def test_get_inputs_reports_the_graph_name_and_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    """The detector reads the input name off the graph rather than guessing it."""
    _install_fake_openvino(monkeypatch)
    session = OpenVinoSession(Path("model.onnx"))

    (model_input,) = session.get_inputs()

    assert model_input.name == "images"
    assert model_input.shape == [1, 3, INPUT_SIZE, INPUT_SIZE]


def test_a_dynamic_axis_fails_the_same_way_on_both_runtimes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A graph we cannot letterbox against must be rejected identically either side.

    The adapter surfaces a dynamic dimension as a non-integer and lets the detector's
    existing guard raise, rather than inventing a second error for the same problem.
    """
    _install_fake_openvino(monkeypatch, shape=[1, 3, INPUT_SIZE, None])
    session = OpenVinoSession(Path("model.onnx"))

    with pytest.raises(ModelError, match="not a fixed NCHW square"):
        OnnxDetector(Path("model.onnx"), session=session)


def test_run_returns_the_first_output_as_an_array(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = np.ones((1, 8, 85), dtype=np.float32)
    _install_fake_openvino(monkeypatch, output=raw)
    session = OpenVinoSession(Path("model.onnx"))

    outputs = session.run(None, {"images": np.zeros((1, 3, INPUT_SIZE, INPUT_SIZE))})

    assert len(outputs) == 1
    assert isinstance(outputs[0], np.ndarray)
    assert np.array_equal(outputs[0], raw)


def test_it_compiles_for_cpu_not_auto(monkeypatch: pytest.MonkeyPatch) -> None:
    """`AUTO` can migrate the model to an iGPU mid-run, changing numerics underneath a
    perf measurement. Which device an accelerator uses is its own decision (ADR-0012)."""
    built = _install_fake_openvino(monkeypatch)
    OpenVinoSession(Path("model.onnx"))

    assert built[0].compiled_with["device_name"] == "CPU"


def test_num_threads_is_passed_to_the_compiler(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every camera is its own process; an unbounded pool per worker fights for cores."""
    built = _install_fake_openvino(monkeypatch)

    OpenVinoSession(Path("model.onnx"), num_threads=2)

    assert built[0].compiled_with["config"] == {"INFERENCE_NUM_THREADS": 2}


def test_no_thread_config_is_sent_when_unbounded(monkeypatch: pytest.MonkeyPatch) -> None:
    built = _install_fake_openvino(monkeypatch)

    OpenVinoSession(Path("model.onnx"))

    assert built[0].compiled_with["config"] == {}


def test_compile_failure_becomes_a_model_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A runtime that cannot load the graph is a `ModelError`, like every other one."""
    _install_fake_openvino(monkeypatch, explode=True)

    with pytest.raises(ModelError, match=r"model\.onnx"):
        OpenVinoSession(Path("model.onnx"))


def test_a_missing_openvino_package_is_a_model_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Constructing the adapter without the extra installed must say so plainly."""
    monkeypatch.setitem(sys.modules, "openvino", None)

    with pytest.raises(ModelError, match="openvino"):
        OpenVinoSession(Path("model.onnx"))


@pytest.mark.slow
def test_real_openvino_session_runs_the_pinned_model(tmp_path: Path) -> None:
    """The unfaked path: the real binding, the real quantized graph, one real frame.

    Everything above fakes `openvino`, which proves the adapter's wiring but never that
    the binding looks the way the adapter assumes. This is the test that would catch a
    renamed attribute on the real API.
    """
    pytest.importorskip("openvino")

    from datetime import UTC, datetime  # noqa: PLC0415 - only this slow path needs it

    from mesopic.detector.model_manager import ModelManager  # noqa: PLC0415
    from mesopic.types import CameraId, DecodedFrame, FrameTs  # noqa: PLC0415

    artefact = ModelManager(tmp_path).ensure("yolox-nano", runtime=Runtime.OPENVINO)
    detector = OnnxDetector(artefact.path, session=OpenVinoSession(artefact.path))

    assert detector.input_size == INPUT_SIZE

    frame = DecodedFrame(
        camera_id=CameraId("cam-1"),
        ts=FrameTs(datetime(2026, 8, 11, 12, 0, tzinfo=UTC)),
        image=np.zeros((1080, 1920, 3), dtype=np.uint8),
        width=1920,
        height=1080,
    )

    # Flat zeros should not produce people. A sanity check on the confidence gate, not an
    # accuracy claim — that is P2.9's job against labelled footage.
    assert detector.detect(frame) == []
    detector.close()
