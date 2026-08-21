"""Person boxes from a frame; owns the model lifecycle (engine-architecture.md §6).

Must not track, and must not know that zones or lines exist.
"""

from __future__ import annotations

from mesopic.detector.detector import Detector

__all__ = ["Detector"]
