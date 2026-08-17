"""Encode one frame for the calibration view, and nothing else.

This is engine-architecture.md §13's single exception to "frames never hit disk": a
snapshot is grabbed on demand, streamed once to the browser to draw zones and lines on,
and never persisted. Nothing in this module opens a file, and nothing that calls it may
either — `test_frame_lifetime.py` asserts the whole round trip writes zero bytes.

It lives in `muster.supervisor` rather than `muster.api` deliberately. The import
contract forbids the web layer from importing a codec, so the encode happens on the side
of the boundary that already owns the pixels and only bytes cross.

Implements P3.8.
"""

from __future__ import annotations

import cv2
import numpy as np

from muster.types import DecodedFrame

MAX_EDGE_PX = 960
"""Longest edge of the encoded snapshot.

The browser needs a backdrop to draw normalized geometry on, and geometry is normalized
precisely so the backdrop's resolution does not matter (§3). Sending a 4K frame would put
megabytes through a queue for a canvas a fraction of the size."""

JPEG_QUALITY = 78
"""Enough to see a doorway and a floor edge. This is a drawing aid, not evidence."""


def encode_snapshot(
    frame: DecodedFrame, *, max_edge: int = MAX_EDGE_PX, quality: int = JPEG_QUALITY
) -> bytes:
    """One frame to JPEG bytes, scaled so its longest edge fits `max_edge`.

    Scaling is skipped when the frame already fits, so a small camera is not resampled
    into a softer image for no reason.
    """
    image = _fit(frame.image, max_edge)
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:  # pragma: no cover - imencode fails only on a malformed array
        msg = "could not encode the snapshot"
        raise RuntimeError(msg)
    return bytes(buffer.tobytes())


def _fit(image: np.ndarray, max_edge: int) -> np.ndarray:
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_edge:
        return image
    scale = max_edge / longest
    return cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


__all__ = ["JPEG_QUALITY", "MAX_EDGE_PX", "encode_snapshot"]
