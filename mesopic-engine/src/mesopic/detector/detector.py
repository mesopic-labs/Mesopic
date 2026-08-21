"""The narrow interface behind which the model is swappable.

This protocol is load-bearing: it is what makes "swap the detector" a swap rather than a
rewrite — for the permissive/Ultralytics choice (ADR-0013), for the vertical-model
marketplace, and for the native-extension escape hatch (ADR-0002).
"""

from __future__ import annotations

from typing import Protocol

from mesopic.types import DecodedFrame, Detection


class Detector(Protocol):
    """Detects people in a single frame."""

    def detect(self, frame: DecodedFrame) -> list[Detection]:
        """Return person detections in the frame's pixel space, person class only."""
        ...

    def close(self) -> None:
        """Release the inference session."""
        ...
