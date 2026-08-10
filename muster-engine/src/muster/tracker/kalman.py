"""Constant-velocity Kalman filter over `[cx, cy, w, h]` (algorithms.md §3.1).

Two deliberate departures from the 2021 SORT/ByteTrack reference, both from §3.1:

* **Width/height, not aspect/height.** Aspect-ratio *velocity* is unstable — a small
  change in either dimension produces a large, noisy derivative, worst for
  partially-occluded boxes whose aspect jumps when legs are clipped. That is exactly
  Muster's case at the till and in the queue.
* **Real Δt, not unit steps.** fps is adaptive, so `F` and `Q` both scale with the
  measured gap. A person moves ~5x further between frames at 1 fps than at 5 fps, and
  scaling `Q` correctly widens the filter's uncertainty when we sample coarsely.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

State = NDArray[np.float64]
"""`[cx, cy, w, h, vcx, vcy, vw, vh]`."""

_DIM = 4

# Noise is scaled by the box height, because a pixel of error means something very
# different for a person at the door than for one at the back of the room.
_STD_POSITION = 1.0 / 20.0
_STD_VELOCITY = 1.0 / 160.0
_STD_MEASUREMENT = 1.0 / 20.0


def to_state(box: NDArray[np.float64]) -> NDArray[np.float64]:
    """`(x1, y1, x2, y2)` -> `[cx, cy, w, h]` (the observed half of the state)."""
    x1, y1, x2, y2 = box
    state: NDArray[np.float64] = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1])
    return state


def to_box(measurement: NDArray[np.float64]) -> NDArray[np.float64]:
    """`[cx, cy, w, h]` -> `(x1, y1, x2, y2)`."""
    cx, cy, w, h = measurement[:_DIM]
    box: NDArray[np.float64] = np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0])
    return box


class BoxKalmanFilter:
    """Tracks one box. Owns no identity, no lifecycle — just the motion estimate."""

    def __init__(self, box: NDArray[np.float64]) -> None:
        self.mean: State = np.concatenate([to_state(box), np.zeros(_DIM)])
        # A new track's velocity is unknown, not zero: the position block starts tight
        # and the velocity block starts loose, so the second observation dominates it.
        height = self.mean[3]
        self.covariance: NDArray[np.float64] = np.diag(
            np.square(
                np.concatenate(
                    [
                        np.full(_DIM, 2.0 * _STD_POSITION * height),
                        np.full(_DIM, 10.0 * _STD_VELOCITY * height),
                    ]
                )
            )
        )

    @property
    def box(self) -> NDArray[np.float64]:
        """The current estimate as `(x1, y1, x2, y2)`."""
        return to_box(self.mean)

    def predict(self, dt_s: float) -> None:
        """Advance the estimate by `dt_s` seconds of constant velocity.

        Raises:
            ValueError: If `dt_s` is negative.
        """
        if dt_s < 0.0:
            msg = f"dt_s must not be negative, got {dt_s}"
            raise ValueError(msg)
        transition = np.eye(2 * _DIM)
        transition[:_DIM, _DIM:] = dt_s * np.eye(_DIM)

        height = max(self.mean[3], 1.0)
        process_noise = np.diag(
            np.square(
                dt_s
                * np.concatenate(
                    [
                        np.full(_DIM, _STD_POSITION * height),
                        np.full(_DIM, _STD_VELOCITY * height),
                    ]
                )
            )
        )
        self.mean = transition @ self.mean
        self.covariance = transition @ self.covariance @ transition.T + process_noise

    def update(self, box: NDArray[np.float64]) -> None:
        """Correct the estimate with an observed box."""
        observation = np.zeros((_DIM, 2 * _DIM))
        observation[:, :_DIM] = np.eye(_DIM)

        height = max(self.mean[3], 1.0)
        measurement_noise = np.diag(np.square(np.full(_DIM, _STD_MEASUREMENT * height)))

        projected = observation @ self.covariance @ observation.T + measurement_noise
        gain = self.covariance @ observation.T @ np.linalg.inv(projected)
        innovation = to_state(box) - observation @ self.mean

        self.mean = self.mean + gain @ innovation
        self.covariance = (np.eye(2 * _DIM) - gain @ observation) @ self.covariance
