"""The detector's orchestration, with the inference session injected.

The arithmetic is covered in `test_detector_postprocess`; what is left here is the wiring
— reading the graph's real input shape and name, feeding it a correctly-shaped tensor,
and handing back boxes in the frame's coordinates. A fake session makes all of that
testable in milliseconds, and the real graph is exercised by a `slow` test.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from mesopic.detector.model_manager import ModelManager
from mesopic.detector.onnx_detector import OnnxDetector, _open_session
from mesopic.detector.runtime import RuntimeSelection
from mesopic.errors import ModelError
from mesopic.types import CameraId, DecodedFrame, FrameTs, Runtime

INPUT_SIZE = 64
ANCHORS = 8 * 8 + 4 * 4 + 2 * 2


class FakeSession:
    """Stands in for an ORT session: reports a graph shape, returns a fixed tensor."""

    def __init__(self, raw: np.ndarray, *, input_name: str = "images") -> None:
        self._raw = raw
        self._input_name = input_name
        self.feeds: list[dict[str, Any]] = []
        self.closed = False

    def get_inputs(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name=self._input_name, shape=[1, 3, INPUT_SIZE, INPUT_SIZE])]

    def run(self, output_names: list[str] | None, input_feed: dict[str, Any]) -> list[np.ndarray]:
        self.requested_outputs = output_names
        self.feeds.append(input_feed)
        return [self._raw]


def _frame(width: int = 64, height: int = 64) -> DecodedFrame:
    return DecodedFrame(
        camera_id=CameraId("cam-1"),
        ts=FrameTs(datetime(2026, 8, 7, 12, 0, tzinfo=UTC)),
        image=np.zeros((height, width, 3), dtype=np.uint8),
        width=width,
        height=height,
    )


def _raw_with_person() -> np.ndarray:
    raw = np.zeros((1, ANCHORS, 85), dtype=np.float32)
    raw[0, 0, 0:2] = 0.5
    raw[0, 0, 2:4] = float(np.log(2.0))
    raw[0, 0, 4] = 1.0
    raw[0, 0, 5] = 1.0
    return raw


def test_detect_returns_person_boxes_in_frame_coordinates() -> None:
    detector = OnnxDetector(Path("unused.onnx"), session=FakeSession(_raw_with_person()))

    detections = detector.detect(_frame())

    assert len(detections) == 1
    assert detections[0].box == (0, 0, 12, 12)
    assert detections[0].score == pytest.approx(1.0)


def test_detect_feeds_the_graph_its_declared_input_name_and_shape() -> None:
    """The input is named by the graph, not guessed.

    YOLOX calls it `images`; another permissive model will not. Reading the name from
    the session is what makes swapping the weight a config change rather than a patch.
    """
    session = FakeSession(_raw_with_person(), input_name="input.1")
    detector = OnnxDetector(Path("unused.onnx"), session=session)

    detector.detect(_frame(width=128, height=96))

    (feed,) = session.feeds
    assert list(feed) == ["input.1"]
    tensor = feed["input.1"]
    assert tensor.shape == (1, 3, INPUT_SIZE, INPUT_SIZE)
    assert tensor.dtype == np.float32


def test_input_size_is_read_from_the_graph_not_assumed() -> None:
    """A 416-px graph fed 640-px tensors is a shape error, not a slow path.

    The stub defaulted to 640 while the pinned YOLOX-Nano export is 416, so this is the
    exact mismatch the property exists to make impossible.
    """
    detector = OnnxDetector(Path("unused.onnx"), session=FakeSession(_raw_with_person()))

    assert detector.input_size == INPUT_SIZE


def test_detect_returns_nothing_when_the_frame_is_empty() -> None:
    empty = np.zeros((1, ANCHORS, 85), dtype=np.float32)
    detector = OnnxDetector(Path("unused.onnx"), session=FakeSession(empty))

    assert detector.detect(_frame()) == []


def test_detect_scales_boxes_for_a_frame_larger_than_the_graph() -> None:
    """A 1080p frame goes into a 64-px graph and the boxes must come back at 1080p."""
    detector = OnnxDetector(Path("unused.onnx"), session=FakeSession(_raw_with_person()))

    (detection,) = detector.detect(_frame(width=1920, height=1080))
    _, _, x2, y2 = detection.box

    # The frame is letterboxed by 64/1920, so a 12-px box in graph space is ~360 px here.
    assert x2 == pytest.approx(360, abs=2)
    assert y2 == pytest.approx(360, abs=2)


@pytest.mark.slow
def test_real_quantized_model_runs_on_a_1080p_frame(tmp_path: Path) -> None:
    """End to end on the pinned weight: fetch, quantize, load, infer.

    Everything above injects a session, which proves the wiring but never touches ONNX
    Runtime. This one does, so a graph whose input name or shape we guessed wrong fails
    here rather than on a user's first frame.
    """
    artefact = ModelManager(tmp_path).ensure("yolox-nano")
    detector = OnnxDetector(artefact.path)

    assert detector.input_size == 416

    detections = detector.detect(_frame(width=1920, height=1080))

    # A frame of flat zeros should not produce people. This is a sanity check on the
    # confidence gate, not an accuracy claim — that is P2.9's job against real footage.
    assert detections == []
    detector.close()


def test_injected_session_reports_the_default_runtime() -> None:
    """An injected session is a test double; claiming an accelerator would be fiction."""
    detector = OnnxDetector(Path("unused.onnx"), session=FakeSession(_raw_with_person()))

    assert detector.runtime is Runtime.ORT_CPU


def test_explicit_unavailable_runtime_is_an_error_not_a_silent_downgrade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Asking for an accelerator and silently getting CPU is an undebuggable mystery.

    The operator gets a box slower than the one they configured, with nothing anywhere
    saying why. An explicit request is a promise: honour it or fail loudly.
    """
    monkeypatch.setattr("mesopic.detector.onnx_detector.is_available", lambda _runtime: False)

    with pytest.raises(ModelError, match="openvino"):
        _open_session(
            Path("unused.onnx"),
            intra_op_threads=None,
            selection=RuntimeSelection(Runtime.OPENVINO, explicit=True),
        )


def test_automatic_unavailable_runtime_falls_back_to_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accelerators are pure upside (ADR-0003); their absence is never fatal."""
    monkeypatch.setattr("mesopic.detector.onnx_detector.is_available", lambda _runtime: False)
    opened: list[str] = []

    def fake_cpu_session(model_path: Path, *, intra_op_threads: int | None) -> FakeSession:
        opened.append(str(model_path))
        assert intra_op_threads is None
        return FakeSession(_raw_with_person())

    monkeypatch.setattr("mesopic.detector.onnx_detector._open_ort_cpu_session", fake_cpu_session)

    session, runtime = _open_session(
        Path("unused.onnx"),
        intra_op_threads=None,
        selection=RuntimeSelection(Runtime.OPENVINO, explicit=False),
    )

    assert opened == ["unused.onnx"]
    assert isinstance(session, FakeSession)
    # The reported runtime must be the one actually executing, not the one asked for —
    # a `/healthz` that names an accelerator the box fell back off is worse than silence.
    assert runtime is Runtime.ORT_CPU


def test_an_accelerator_that_fails_to_load_is_fatal_only_when_asked_for(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Available but unloadable is the same question as unavailable, one step later."""
    monkeypatch.setattr("mesopic.detector.onnx_detector.is_available", lambda _runtime: True)

    def exploding_session(model_path: Path, *, num_threads: int | None) -> None:
        del model_path, num_threads
        message = "compile failed"
        raise ModelError(message)

    def fake_cpu_session(model_path: Path, *, intra_op_threads: int | None) -> FakeSession:
        del model_path, intra_op_threads
        return FakeSession(_raw_with_person())

    monkeypatch.setattr("mesopic.detector.onnx_detector.OpenVinoSession", exploding_session)
    monkeypatch.setattr("mesopic.detector.onnx_detector._open_ort_cpu_session", fake_cpu_session)

    with pytest.raises(ModelError, match="compile failed"):
        _open_session(
            Path("unused.onnx"),
            intra_op_threads=None,
            selection=RuntimeSelection(Runtime.OPENVINO, explicit=True),
        )

    fell_back, runtime = _open_session(
        Path("unused.onnx"),
        intra_op_threads=None,
        selection=RuntimeSelection(Runtime.OPENVINO, explicit=False),
    )

    assert isinstance(fell_back, FakeSession)
    assert runtime is Runtime.ORT_CPU


def test_close_releases_the_session() -> None:
    detector = OnnxDetector(Path("unused.onnx"), session=FakeSession(_raw_with_person()))

    detector.close()

    with pytest.raises(RuntimeError, match="closed"):
        detector.detect(_frame())
