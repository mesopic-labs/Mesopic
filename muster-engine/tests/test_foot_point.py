"""The engine's one pixel->normalized conversion, tested as pure arithmetic.

Every metric is computed from the foot-point and nothing else, so a sign error or an
off-by-one here is a silent, uniform bias in every number the product reports. It is
worth more tests than its four lines suggest.
"""

from __future__ import annotations

import pytest

from muster.tracker.bytetrack import foot_point


def test_foot_point_is_bottom_centre_not_centroid() -> None:
    """The whole convention: where the person meets the floor, not their middle."""
    assert foot_point((0, 0, 100, 200), 100, 200) == (0.5, 1.0)


def test_foot_point_normalizes_against_frame_size() -> None:
    """Zones authored once must survive a resolution change."""
    assert foot_point((320, 0, 640, 540), 1920, 1080) == (0.25, 0.5)


def test_foot_point_is_resolution_agnostic() -> None:
    """The same box, expressed at two resolutions, is the same normalized point."""
    assert foot_point((80, 0, 160, 270), 480, 270) == foot_point((320, 0, 640, 1080), 1920, 1080)


@pytest.mark.parametrize(
    ("box", "expected"),
    [
        ((-40, 0, 40, 1200), (0.0, 1.0)),  # overhangs left and bottom
        ((1900, 0, 2000, 100), (1.0, 0.0925925925925926)),  # overhangs right
        ((-100, 0, -20, 200), (0.0, 200 / 1080)),  # entirely left, exercises x lower clamp
        ((0, -300, 40, -100), (20 / 1920, 0.0)),  # entirely above, exercises y lower clamp
    ],
)
def test_foot_point_clamps_boxes_that_overhang_the_frame(
    box: tuple[int, int, int, int], expected: tuple[float, float]
) -> None:
    """A detector may return a box past the edge; downstream contracts say [0, 1]."""
    x, y = foot_point(box, 1920, 1080)
    assert (x, y) == pytest.approx(expected)


def test_foot_point_rejects_a_degenerate_frame() -> None:
    """A zero-width frame is a programming error, not something to divide by."""
    with pytest.raises(ValueError, match="frame dimensions"):
        foot_point((0, 0, 10, 10), 0, 1080)
