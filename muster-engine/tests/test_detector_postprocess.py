"""YOLOX post-processing, tested as pure arithmetic.

The detector's accuracy lives almost entirely in these two functions, and neither needs
a model to exercise: `letterbox` is a resize with padding, and `decode_yolox_output`
turns a raw ``(1, N, 85)`` tensor into boxes. Testing them directly means the tricky
parts — grid/stride decoding, objectness times class score, NMS, and undoing the
letterbox — are pinned by fast tests with hand-built inputs, rather than inferred from
whether a real image happened to produce plausible-looking boxes.

Anchor counts here use a deliberately tiny ``input_size`` so the expected values can be
worked out by hand: at 64 px the strides 8/16/32 give 8²+4²+2² = 84 anchors.
"""

from __future__ import annotations

import numpy as np
import pytest

from muster.detector.postprocess import decode_yolox_output, letterbox
from muster.types import Detection

INPUT_SIZE = 64
ANCHORS = 8 * 8 + 4 * 4 + 2 * 2
PERSON = 0
DOG = 16


def _raw(anchors: int = ANCHORS) -> np.ndarray:
    """An all-zero prediction tensor: every anchor present, none confident."""
    return np.zeros((1, anchors, 85), dtype=np.float32)


def _set_anchor(
    raw: np.ndarray,
    index: int,
    *,
    offset: tuple[float, float] = (0.5, 0.5),
    log_size: float = float(np.log(2.0)),
    objectness: float = 1.0,
    class_id: int = PERSON,
    class_score: float = 1.0,
) -> None:
    raw[0, index, 0:2] = offset
    raw[0, index, 2:4] = log_size
    raw[0, index, 4] = objectness
    raw[0, index, 5 + class_id] = class_score


def _decode(
    raw: np.ndarray,
    *,
    ratio: float = 1.0,
    frame_width: int = INPUT_SIZE,
    frame_height: int = INPUT_SIZE,
    confidence: float = 0.35,
    iou_threshold: float = 0.45,
) -> list[Detection]:
    return decode_yolox_output(
        raw,
        input_size=INPUT_SIZE,
        ratio=ratio,
        frame_width=frame_width,
        frame_height=frame_height,
        confidence=confidence,
        iou_threshold=iou_threshold,
    )


def test_letterbox_preserves_aspect_ratio_and_pads_to_square() -> None:
    image = np.full((100, 200, 3), 255, dtype=np.uint8)

    tensor, ratio = letterbox(image, INPUT_SIZE)

    assert tensor.shape == (1, 3, INPUT_SIZE, INPUT_SIZE)
    assert tensor.dtype == np.float32
    # 200 px of width has to fit in 64, so everything scales by 64/200.
    assert ratio == pytest.approx(0.32)
    # The image occupies the top-left 32 rows; the rest is padding, not image.
    assert tensor[0, :, :32, :64].max() == pytest.approx(255.0)
    assert tensor[0, :, 33:, :].max() == pytest.approx(114.0)


def test_decode_returns_one_box_for_a_confident_person_anchor() -> None:
    raw = _raw()
    _set_anchor(raw, 0)  # first anchor of the stride-8 grid, cell (0, 0)

    detections = _decode(raw)

    assert len(detections) == 1
    # centre = (0 + 0.5) * 8 = 4; size = exp(log 2) * 8 = 16; so the box is (-4,-4,12,12)
    # before clamping to the frame.
    assert detections[0].box == (0, 0, 12, 12)
    assert detections[0].score == pytest.approx(1.0)


def test_decode_ignores_every_class_except_person() -> None:
    """A dog is not footfall.

    The 80-class COCO head is a property of the upstream weight; everything downstream
    counts people, so the filter belongs here rather than being someone else's problem.
    """
    raw = _raw()
    _set_anchor(raw, 0, class_id=DOG)

    assert _decode(raw) == []


def test_decode_drops_anchors_below_the_confidence_threshold() -> None:
    raw = _raw()
    # Score is objectness times class probability, so 0.5 * 0.5 = 0.25 — under the bar
    # even though neither factor is obviously low on its own.
    _set_anchor(raw, 0, objectness=0.5, class_score=0.5)

    assert _decode(raw) == []
    assert len(_decode(raw, confidence=0.2)) == 1


def test_decode_suppresses_duplicate_detections_of_the_same_person() -> None:
    """Neighbouring anchors fire on one person; NMS has to collapse them to one box."""
    raw = _raw()
    _set_anchor(raw, 0, offset=(0.5, 0.5), objectness=1.0)
    _set_anchor(raw, 1, offset=(-0.4, 0.5), objectness=0.9)  # near-identical box

    detections = _decode(raw)

    assert len(detections) == 1
    # The survivor is the higher-scoring one.
    assert detections[0].score == pytest.approx(1.0)


def test_decode_keeps_two_people_who_do_not_overlap() -> None:
    raw = _raw()
    _set_anchor(raw, 0)
    _set_anchor(raw, 63)  # far corner of the stride-8 grid

    assert len(_decode(raw)) == 2


def test_decode_maps_boxes_back_through_the_letterbox_ratio() -> None:
    """Boxes come back in the frame's pixel space, not the model's 416-px square.

    The letterbox is the detector's own business. Handing a caller coordinates in
    padded model space would make every downstream consumer undo a transform it should
    never have been told about.
    """
    raw = _raw()
    _set_anchor(raw, 0)

    detections = _decode(raw, ratio=0.5, frame_width=128, frame_height=128)

    # Same box as the unscaled case, divided by the ratio.
    assert detections[0].box == (0, 0, 24, 24)


def test_decode_clamps_boxes_to_the_frame() -> None:
    """A box running off the edge would put a foot-point outside [0,1] after
    normalization, which every metric downstream would then have to defend against."""
    raw = _raw()
    _set_anchor(raw, ANCHORS - 1, log_size=float(np.log(8.0)))  # huge box, bottom-right

    (detection,) = _decode(raw)
    x1, y1, x2, y2 = detection.box

    assert x1 >= 0
    assert y1 >= 0
    assert x2 <= INPUT_SIZE
    assert y2 <= INPUT_SIZE


def test_decode_handles_an_empty_frame() -> None:
    assert _decode(_raw()) == []
