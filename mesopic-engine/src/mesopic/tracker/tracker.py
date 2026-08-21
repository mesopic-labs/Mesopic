"""The tracker interface.

ADR-0014 makes this protocol load-bearing rather than incidental: plain IoU association
collapses below ~3 fps, so both the association cost function and the tracker itself are
benchmark-gated choices we must be able to swap.

No appearance model, no Re-ID embedding — that would blow the CPU budget and edge toward
the biometrics we explicitly refuse.
"""

from __future__ import annotations

from typing import Protocol

from mesopic.types import DecodedFrame, Detection, Track


class Tracker(Protocol):
    """Associates detections across frames into tracks."""

    def update(self, frame: DecodedFrame, detections: list[Detection]) -> list[Track]:
        """Advance the tracker one tick and return the currently live tracks."""
        ...
