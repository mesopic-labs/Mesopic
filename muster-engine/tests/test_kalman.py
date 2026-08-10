"""The constant-velocity filter, and specifically its Δt behaviour.

Classic ByteTrack advances one unit "step" per frame. Muster's fps is adaptive, so a
unit-step filter would under-predict displacement by 5x exactly when the controller has
dropped to 1 fps -- i.e. when the scene is busiest. algorithms.md §3.1 picks option (i),
feeding the real elapsed time, and these tests are what hold it to that.
"""

from __future__ import annotations

import numpy as np
import pytest

from muster.tracker.kalman import BoxKalmanFilter, to_box, to_state

BOX = np.array([100.0, 200.0, 140.0, 300.0])  # 40x100 at (100, 200)


def test_state_round_trips_through_box_form() -> None:
    assert to_box(to_state(BOX)) == pytest.approx(BOX)


def test_a_new_track_has_no_velocity() -> None:
    """One observation cannot imply motion -- the ADR's 'newly-born track' case."""
    kf = BoxKalmanFilter(BOX)
    kf.predict(0.5)
    assert kf.box == pytest.approx(BOX)


def test_prediction_advances_by_real_elapsed_time() -> None:
    """Two observations 0.5 s apart imply a velocity; the next 0.5 s extrapolates it."""
    kf = BoxKalmanFilter(BOX)
    kf.predict(0.5)
    kf.update(BOX + np.array([20.0, 0.0, 20.0, 0.0]))
    before = kf.box[0]
    kf.predict(0.5)
    assert kf.box[0] > before


def test_a_longer_gap_predicts_a_longer_displacement() -> None:
    """The property a unit-step filter gets wrong: 1 fps must move 5x further than 5 fps."""
    displacements = []
    for dt in (0.2, 1.0):
        kf = BoxKalmanFilter(BOX)
        kf.predict(dt)
        kf.update(BOX + np.array([10.0, 0.0, 10.0, 0.0]))
        start = kf.box[0]
        kf.predict(dt)
        displacements.append(kf.box[0] - start)
    assert displacements[1] > displacements[0] * 3.0


def test_uncertainty_grows_with_the_sampling_gap() -> None:
    """Sampling coarsely should make the filter *less* sure, not equally sure."""
    coarse, fine = BoxKalmanFilter(BOX), BoxKalmanFilter(BOX)
    coarse.predict(1.0)
    fine.predict(0.2)
    assert np.trace(coarse.covariance) > np.trace(fine.covariance)


def test_update_pulls_the_estimate_toward_the_observation() -> None:
    kf = BoxKalmanFilter(BOX)
    kf.predict(0.5)
    kf.update(np.array([200.0, 200.0, 240.0, 300.0]))
    assert 100.0 < kf.box[0] <= 200.0


def test_a_negative_gap_is_rejected() -> None:
    """Frames arrive in order; a backwards Δt means a clock bug upstream."""
    kf = BoxKalmanFilter(BOX)
    with pytest.raises(ValueError, match="dt_s"):
        kf.predict(-0.1)
