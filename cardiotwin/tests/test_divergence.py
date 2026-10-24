"""Tests for the Cardiovascular Divergence Engine."""

import pytest
import numpy as np
from cardiotwin.twin.estimator import CardiovascularUKF
from cardiotwin.twin.divergence import DivergenceEngine


def test_divergence_score_rises_after_injected_anomaly():
    """Divergence score must rise sharply when an anomaly is injected."""
    ukf = CardiovascularUKF()
    engine = DivergenceEngine(persistence_n=2)

    # 1. Warm-up with normal baseline readings
    normal_obs = np.array([72.0, 120.0, 80.0])
    for _ in range(10):
        step_res = ukf.step(normal_obs)
        div_res = engine.evaluate_step(step_res, ukf)

    # Under baseline normal readings, score should be low and status Normal
    assert div_res.divergence_score < 2.5
    assert div_res.status == "Normal"

    # 2. Inject acute hypertensive vascular anomaly (SBP jumps to 175, DBP to 110)
    anomaly_obs = np.array([74.0, 175.0, 110.0])

    step_res_1 = ukf.step(anomaly_obs)
    div_res_1 = engine.evaluate_step(step_res_1, ukf)
    assert div_res_1.divergence_score > 3.0, f"Divergence score {div_res_1.divergence_score} didn't spike"

    # Step 2 of anomaly -> persistence kicks in -> status becomes Significant
    step_res_2 = ukf.step(anomaly_obs)
    div_res_2 = engine.evaluate_step(step_res_2, ukf)
    assert div_res_2.consecutive_high_count >= 2
    assert div_res_2.status == "Significant"


def test_ensemble_predictive_uncertainty_bands():
    """50-twin ensemble must return valid p5, median, and p95 percentiles."""
    ukf = CardiovascularUKF()
    engine = DivergenceEngine(ensemble_size=50)

    step_res = ukf.step(np.array([72.0, 120.0, 80.0]))
    div_res = engine.evaluate_step(step_res, ukf)

    bands = div_res.ensemble_bands
    assert "SBP" in bands and "DBP" in bands and "HR" in bands

    for metric in ["SBP", "DBP", "HR"]:
        b = bands[metric]
        assert b["p5"] <= b["median"] <= b["p95"], f"Inconsistent percentiles for {metric}: {b}"
        assert b["std"] > 0.0


def test_sensor_dropout_does_not_cause_false_alarm():
    """Missing observations should be flagged gracefully without false alarms."""
    ukf = CardiovascularUKF()
    engine = DivergenceEngine()

    step_res = ukf.step(None)
    div_res = engine.evaluate_step(step_res, ukf)

    assert div_res.is_missing_data
    assert div_res.status == "Normal"
    assert div_res.consecutive_high_count == 0
