"""Constant-velocity Kalman filter over `[cx, cy, w, h]` (algorithms.md §3.1).

Two deliberate departures from the 2021 SORT/ByteTrack reference, both from §3.1:

* **Width/height, not aspect/height.** Aspect-ratio *velocity* is unstable — a small
  change in either dimension produces a large, noisy derivative, worst for
  partially-occluded boxes whose aspect jumps when legs are clipped. That is exactly
  Muster's case at the till and in the queue.
* **Real Δt, not unit steps.** fps is adaptive, so `F` and `Q` both scale with the
  measured gap. A person moves ~5x further between frames at 1 fps than at 5 fps, and
  scaling `Q` correctly widens the filter's uncertainty when we sample coarsely.

## Noise-constant derivation (algorithms.md §3.3.1)

Every noise constant below is derived here from the physical model documented in
algorithms.md §3.3.1 (reference walker, 1.5 m/s average; §4/§13.4's reference adult,
1.7 m). None of it is copied from a published tracker's noise model — this is an MIT
repository, and the classic deep_sort/SORT noise constants are GPL-3.0, so they may
never be carried in here, even as innocuous-looking magic numbers.

The bridge from metres to pixels is the observed box height itself: a box `height_px`
tall belongs to a `_PERSON_HEIGHT_M` person, so

    scale = height_px / _PERSON_HEIGHT_M      # pixels per metre at this person's depth

is the local pixels-per-metre scale. Expressing every constant through `scale` is why
they stay correct for a person near the camera and one far from it alike.

* **Measurement noise (`R`).** The detector localises a box edge to about
  `_BOX_LOCALISATION_FRACTION` (5%) of the person's own height — an engineering
  estimate of detector edge accuracy, distinct from the walker-speed figures below.
  `std_meas = _BOX_LOCALISATION_FRACTION * height_px`.
* **Initial position covariance.** The first observation *is* the position estimate,
  so its uncertainty is exactly the measurement noise above:
  `std_pos_init = _BOX_LOCALISATION_FRACTION * height_px`.
* **Initial velocity covariance.** A single observation says nothing about *direction*
  of travel, so the prior must span the full envelope of plausible human movement, not
  a tight guess: `std_vel_init = _MAX_WALK_SPEED_MS * scale`, where `_MAX_WALK_SPEED_MS`
  (2.0 m/s) is a brisk-walk envelope above §3.3.1's 1.5 m/s reference average. This
  comes out far wider (in variance, ~20x for a typical box) than the position prior —
  by derivation, not a hand-tuned multiplier — because a new track's velocity is
  unknown and the *second* observation must be the one that pins it down.
* **Process noise (`Q`), white-noise-acceleration model.** An unmodelled acceleration
  `a` over `dt` seconds perturbs velocity by `a*dt` and position by `0.5*a*dt^2` — these
  scale *differently* with `dt` (linear vs quadratic), which is physically correct and
  is why `Q`'s two blocks are computed separately rather than by one shared factor.
  `a_px = _MAX_ACCEL_MS2 * scale`, with `_MAX_ACCEL_MS2` (1.0 m/s^2) covering a person
  starting, stopping, or turning:

    - `std_v(dt) = a_px * dt`
    - `std_p(dt) = 0.5 * a_px * dt ** 2`
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

State = NDArray[np.float64]
"""`[cx, cy, w, h, vcx, vcy, vw, vh]`."""

_DIM = 4

_PERSON_HEIGHT_M = 1.7
"""The reference adult (algorithms.md §4/§13.4) — the metres-to-pixels anchor."""

_MAX_WALK_SPEED_MS = 2.0
"""Brisk-walk envelope, above §3.3.1's 1.5 m/s reference average — a bound, not a mean."""

_MAX_ACCEL_MS2 = 1.0
"""Covers a person starting, stopping, or turning between sampled frames."""

_BOX_LOCALISATION_FRACTION = 0.05
"""Detector edge accuracy, as a fraction of the person's own height (~8.5 cm at 1.7 m)."""


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
        # (it *is* the observation) and the velocity block starts loose — spanning the
        # full walking-speed envelope — so the second observation dominates it.
        height = max(self.mean[3], 1.0)
        scale = height / _PERSON_HEIGHT_M
        std_position = _BOX_LOCALISATION_FRACTION * height
        std_velocity = _MAX_WALK_SPEED_MS * scale
        self.covariance: NDArray[np.float64] = np.diag(
            np.square(np.concatenate([np.full(_DIM, std_position), np.full(_DIM, std_velocity)]))
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

        # White-noise-acceleration model: an unmodelled a·dt perturbs velocity
        # linearly but position only by ½·a·dt² — the two blocks must not share a
        # scaling factor, or the filter overstates position uncertainty at short dt.
        height = max(self.mean[3], 1.0)
        accel = _MAX_ACCEL_MS2 * (height / _PERSON_HEIGHT_M)
        std_position = 0.5 * accel * dt_s**2
        std_velocity = accel * dt_s
        process_noise = np.diag(
            np.square(np.concatenate([np.full(_DIM, std_position), np.full(_DIM, std_velocity)]))
        )
        self.mean = transition @ self.mean
        self.covariance = transition @ self.covariance @ transition.T + process_noise

    def update(self, box: NDArray[np.float64]) -> None:
        """Correct the estimate with an observed box."""
        observation = np.zeros((_DIM, 2 * _DIM))
        observation[:, :_DIM] = np.eye(_DIM)

        height = max(self.mean[3], 1.0)
        std_measurement = _BOX_LOCALISATION_FRACTION * height
        measurement_noise = np.diag(np.square(np.full(_DIM, std_measurement)))

        projected = observation @ self.covariance @ observation.T + measurement_noise
        gain = self.covariance @ observation.T @ np.linalg.inv(projected)
        innovation = to_state(box) - observation @ self.mean

        self.mean = self.mean + gain @ innovation
        self.covariance = (np.eye(2 * _DIM) - gain @ observation) @ self.covariance
