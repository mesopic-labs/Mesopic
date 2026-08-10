"""ByteTrack wrapper — the default tracker, benchmark-gated (ADR-0014).

This module owns two things nothing else may duplicate:

* **Foot-point derivation** — bottom-centre of the bounding box, never the centroid.
* **Pixel -> normalized conversion** — the *only* place in the engine it happens.

Implements P1.5.
"""

from __future__ import annotations

from muster.types import DecodedFrame, Detection, NormPoint, PixelBox, Track


def foot_point(box: PixelBox, frame_width: int, frame_height: int) -> NormPoint:
    """Bottom-centre of `box`, in normalized `[0, 1]` coordinates.

    The single most important geometric convention in the engine: approximately where
    the person meets the floor, which is what every metric is actually about. Defined in
    algorithms.md; this is its one implementation.

    Args:
        box: `(x1, y1, x2, y2)` in inference-frame pixels.
        frame_width: Inference-frame width in pixels.
        frame_height: Inference-frame height in pixels.

    Returns:
        `(x, y)` in `[0.0, 1.0]`, origin top-left.

    Raises:
        ValueError: If either frame dimension is not positive.
    """
    if frame_width <= 0 or frame_height <= 0:
        msg = f"frame dimensions must be positive, got {frame_width}x{frame_height}"
        raise ValueError(msg)
    x1, _, x2, y2 = box
    x = (x1 + x2) / 2.0 / frame_width
    y = y2 / frame_height
    return (min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0))


class ByteTrackTracker:
    """A `Tracker` implementing BYTE association with a pluggable cost (ADR-0014)."""

    def __init__(self, *, track_thresh: float = 0.5, max_iou_cost: float = 0.8) -> None:
        # `max_iou_cost` gates the association COST (1 - IoU), not the IoU. 0.8 means a
        # minimum IoU of 0.2. The upstream name (`match_thresh`) invites exactly the
        # wrong reading, and that misreading was a real bug in the design drafts.
        raise NotImplementedError

    def update(self, frame: DecodedFrame, detections: list[Detection]) -> list[Track]:
        raise NotImplementedError
