"""Turning a YOLOX graph's raw output into person boxes.

Kept separate from the inference session on purpose. The session is I/O — load a file,
call `run` — but everything here is arithmetic, and it is the arithmetic that decides
whether the engine counts people correctly. Splitting them means the accuracy-critical
half is covered by fast tests with hand-built tensors instead of being inferred from
whether a real photograph produced plausible-looking output.

The graph emits ``(1, N, 85)``: 4 box terms, 1 objectness, 80 COCO class scores, over
N anchors — one per cell of three grids at strides 8/16/32. At 416 px that is
52² + 26² + 13² = 3549 anchors. The boxes are *not* final: each is an offset from its
grid cell in stride units, so decoding them is our job, not the model's.

Implements the post-processing half of P1.4.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from muster.types import BgrImage, Detection

STRIDES = (8, 16, 32)
"""The three feature-map strides YOLOX predicts at, coarsest last."""

PERSON_CLASS_ID = 0
"""COCO's person class. The only one the engine has any use for (engine §1)."""

PAD_VALUE = 114
"""Neutral grey. Matches the padding the model was trained and exported against."""

_BOX_TERMS = 4
_OBJECTNESS = 4
_CLASS_OFFSET = 5
_BATCHED_NDIM = 3
"""``(batch, anchors, terms)``. A 2-D tensor is the same thing with the batch stripped."""


def letterbox(image: BgrImage, size: int) -> tuple[NDArray[np.float32], float]:
    """Fit a frame into a square model input without distorting it.

    Returns the NCHW tensor and the scale factor applied, which the caller needs to map
    boxes back afterwards. Aspect ratio is preserved and the remainder padded, because
    stretching a 16:9 frame into a square makes every standing person wider than the
    model was trained to expect.

    No normalization and no channel swap: YOLOX's exported graph takes raw BGR in
    ``[0, 255]``, which is what the decoder hands us anyway.
    """
    height, width = image.shape[:2]
    ratio = min(size / height, size / width)
    scaled_h, scaled_w = int(height * ratio), int(width * ratio)

    resized = cv2.resize(image, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), PAD_VALUE, dtype=np.uint8)
    canvas[:scaled_h, :scaled_w] = resized

    tensor = canvas.transpose(2, 0, 1).astype(np.float32)[np.newaxis]
    return np.ascontiguousarray(tensor), ratio


def decode_yolox_output(
    raw: NDArray[np.float32],
    *,
    input_size: int,
    ratio: float,
    frame_width: int,
    frame_height: int,
    confidence: float,
    iou_threshold: float,
) -> list[Detection]:
    """Raw graph output to person boxes in the frame's own pixel space.

    Boxes come back in the coordinates of the frame that was passed in, not the padded
    model square — the letterbox is the detector's private business, and leaking it
    would make every downstream consumer undo a transform it was never told about.
    """
    predictions = raw[0] if raw.ndim == _BATCHED_NDIM else raw
    grids, strides = _grids_and_strides(input_size)

    centres = (predictions[:, 0:2] + grids) * strides
    sizes = np.exp(predictions[:, 2:_BOX_TERMS]) * strides
    # Objectness gates the class score: a confident "person" on an anchor the model does
    # not believe contains an object at all is not a detection.
    scores = predictions[:, _OBJECTNESS] * predictions[:, _CLASS_OFFSET + PERSON_CLASS_ID]

    confident = scores >= confidence
    if not confident.any():
        return []

    boxes = _to_corners(centres[confident], sizes[confident]) / ratio
    boxes = _clamp(boxes, frame_width, frame_height)
    scores = scores[confident]

    kept = _non_max_suppression(boxes, scores, iou_threshold)
    rounded = np.round(boxes[kept]).astype(int)
    return [
        Detection(box=(int(x1), int(y1), int(x2), int(y2)), score=float(score))
        for (x1, y1, x2, y2), score in zip(rounded, scores[kept], strict=True)
    ]


def _grids_and_strides(input_size: int) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Per-anchor grid coordinates and the stride each was predicted at.

    Concatenated coarsest-last, matching the order the graph emits them in — get this
    order wrong and every box lands in the wrong part of the frame.
    """
    grids: list[NDArray[np.float32]] = []
    strides: list[NDArray[np.float32]] = []
    for stride in STRIDES:
        cells = input_size // stride
        xs, ys = np.meshgrid(np.arange(cells), np.arange(cells))
        grids.append(np.stack((xs, ys), axis=2).reshape(-1, 2).astype(np.float32))
        strides.append(np.full((cells * cells, 1), stride, dtype=np.float32))
    return np.concatenate(grids), np.concatenate(strides)


def _to_corners(centres: NDArray[np.float32], sizes: NDArray[np.float32]) -> NDArray[np.float32]:
    half = sizes / 2.0
    return np.concatenate([centres - half, centres + half], axis=1)


def _clamp(boxes: NDArray[np.float32], width: int, height: int) -> NDArray[np.float32]:
    """Hold boxes inside the frame.

    A box running off the edge puts a foot-point outside ``[0, 1]`` once the tracker
    normalizes it, which every metric downstream would then have to defend against.
    """
    boxes[:, 0] = boxes[:, 0].clip(0, width)
    boxes[:, 1] = boxes[:, 1].clip(0, height)
    boxes[:, 2] = boxes[:, 2].clip(0, width)
    boxes[:, 3] = boxes[:, 3].clip(0, height)
    return boxes


def _non_max_suppression(
    boxes: NDArray[np.float32], scores: NDArray[np.float32], iou_threshold: float
) -> list[int]:
    """Greedy NMS: keep the best box, drop everything that overlaps it too much.

    Several neighbouring anchors routinely fire on one person. Without this, one person
    standing still counts as three, and footfall is wrong in a way that looks like a
    tracking bug rather than a detection one.
    """
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    order = scores.argsort()[::-1]

    kept: list[int] = []
    while order.size > 0:
        best = int(order[0])
        kept.append(best)
        if order.size == 1:
            break

        rest = order[1:]
        left = np.maximum(boxes[best, 0], boxes[rest, 0])
        top = np.maximum(boxes[best, 1], boxes[rest, 1])
        right = np.minimum(boxes[best, 2], boxes[rest, 2])
        bottom = np.minimum(boxes[best, 3], boxes[rest, 3])
        intersection = np.maximum(0.0, right - left) * np.maximum(0.0, bottom - top)
        union = areas[best] + areas[rest] - intersection
        # A degenerate zero-area box overlaps nothing; guard the division rather than
        # letting it produce a nan that silently compares false.
        iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
        order = rest[iou <= iou_threshold]

    return kept
