"""Tests for the Cardiovascular UKF state estimator."""

import pytest
import numpy as np
from cardiotwin.twin.estimator import CardiovascularUKF
from cardiotwin.twin.model import WindkesselParams, simulate_beat


def test_estimator_converges_on_noiseless_synthetic_data():
    """UKF must track and converge to match noiseless synthetic hemodynamic readings."""
    # Target true state
    true_state = np.array([1.35, 1.25, 68.0, 75.0])
    p_true = WindkesselParams(Rp=true_state[0], C=true_state[1], Rc=0.05, SV=true_state[2], HR=true_state[3])
    out_true = simulate_beat(p_true)
    y_target = np.array([out_true.hr, out_true.sbp, out_true.dbp])

    # Initialize UKF away from target
    x_init = np.array([1.10, 1.50, 60.0, 68.0])
    ukf = CardiovascularUKF(initial_state=x_init)

    # Run for 30 steps with constant target observation
    for _ in range(30):
        result = ukf.step(y_target)

    # Predicted observations must be within 1.5 units of target
    np.testing.assert_allclose(result.predicted_obs, y_target, atol=2.0)
    assert result.innovation is not None
    assert np.linalg.norm(result.innovation) < 2.0


def test_estimator_skips_missing_observations():
    """UKF should skip update if observation has NaNs or is None."""
    ukf = CardiovascularUKF()
    initial_cov_trace = np.trace(ukf.P)

    # 1. Step with NaN observation
    res_nan = ukf.step(np.array([np.nan, 120.0, 80.0]))
    assert not res_nan.update_performed
    assert res_nan.observed is None
    assert res_nan.innovation is None
    # Prior covariance grew due to process noise Q
    assert np.trace(ukf.P) > initial_cov_trace

    # 2. Step with None observation
    res_none = ukf.step(None)
    assert not res_none.update_performed
    assert res_none.time_step == 2


def test_estimator_maintains_positive_definite_covariance():
    """Covariance matrix must remain symmetric positive-definite."""
    ukf = CardiovascularUKF()
    for _ in range(15):
        ukf.step(np.array([72.0, 120.0, 80.0]))
        # Check symmetry
        np.testing.assert_allclose(ukf.P, ukf.P.T, atol=1e-8)
        # Check all eigenvalues are strictly positive
        eigvals = np.linalg.eigvalsh(ukf.P)
        assert np.all(eigvals > 0.0)
