"""Detection via ONNX Runtime, CPU execution provider by default (ADR-0012).

ORT-CPU is the portable default and always works. OpenVINO (Intel), Coral, and
CUDA/TensorRT are probed at load time and are pure upside, never a dependency (ADR-0003).

Performance levers that belong here and nowhere else: input size, ROI crop to the union
of the camera's configured geometry, INT8 quantization, and thread counts.

Implements P1.4.
"""

from __future__ import annotations

from muster.types import DecodedFrame, Detection


class OnnxDetector:
    """A `Detector` backed by an ONNX graph on ONNX Runtime."""

    def __init__(self, model_path: str, *, input_size: int = 640, confidence: float = 0.35) -> None:
        raise NotImplementedError

    def detect(self, frame: DecodedFrame) -> list[Detection]:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError
