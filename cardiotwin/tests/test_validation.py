"""Tests for the CardioTwin X Cohort Validation and Data Reliability Engines."""

import pytest
import numpy as np
from cardiotwin.twin.validation import (
    VALIDATION_NOTE,
    check_public_dataset,
    run_cohort_validation,
)
from cardiotwin.twin.state import TwinSessionState


def test_public_dataset_detection_when_missing():
    """Verify missing public benchmark dataset returns 'public dataset not loaded' status."""
    res = check_public_dataset(search_paths=["definitely_nonexistent_dataset_12345.csv"])
    assert res["loaded"] is False
    assert "public dataset not loaded" in res["message"].lower()


def test_cohort_validation_structure():
    """Verify cohort validation aggregates patients into required tables and calibration curves."""
    summary = run_cohort_validation(patients_per_preset=1, steps_per_patient=25, seed_base=8000)
    assert summary.total_patients == 3
    assert summary.note == VALIDATION_NOTE
    assert "SBP" in summary.overall_mae and "DBP" in summary.overall_mae and "HR" in summary.overall_mae
    assert "SBP" in summary.overall_rmse and "DBP" in summary.overall_rmse and "HR" in summary.overall_rmse
    assert summary.overall_mae["SBP"] > 0.0
    assert summary.overall_rmse["SBP"] >= summary.overall_mae["SBP"]
    assert 0.0 <= summary.observed_coverage_90 <= 100.0

    # Calibration curve
    cal = summary.calibration_curve
    assert "expected" in cal and "observed" in cal
    assert len(cal["expected"]) == len(cal["observed"])
    assert len(cal["expected"]) > 0

    # Benchmark comparison
    assert "tpr_pct" in summary.divergence_metrics
    assert "fpr_pct" in summary.divergence_metrics
    assert "tpr_pct" in summary.isoforest_metrics

    # Preset stratification table
    assert not summary.preset_table.empty
    assert "Preset" in summary.preset_table.columns
    assert "SBP_MAE" in summary.preset_table.columns
    assert len(summary.preset_table) == 3


def test_data_reliability_scoring():
    """Test Data Reliability indicator levels (High, Medium, Low) and uncertainty warning."""
    session = TwinSessionState(preset_key="healthy", seed=42)

    # Initially step several times with clean telemetry
    for _ in range(8):
        session.step_forward()

    level, reason, metrics = session.compute_data_reliability()
    assert level in ["High", "Medium"]
    assert metrics["valid_ratio"] >= 0.80

    # Ingest a missing/NaN frame -> Reliability must drop to Low
    session.step_forward(manual_obs=np.array([np.nan, np.nan, np.nan]))
    low_level, low_reason, _ = session.compute_data_reliability()
    assert low_level == "Low"
    assert "offline" in low_reason.lower() or "dropout" in low_reason.lower()


def test_explainable_alerts_storage_on_divergence():
    """Test that Watch or Significant alerts are recorded in alerts_log with required fields."""
    session = TwinSessionState(preset_key="healthy", seed=42)

    # Advance normally
    for _ in range(5):
        session.step_forward()

    # Inject acute anomaly to force Watch/Significant status
    for _ in range(4):
        session.step_forward(manual_obs=np.array([125.0, 185.0, 115.0]))

    # An alert should have been recorded
    assert len(session.alerts_log) > 0
    latest_alert = session.alerts_log[-1]
    assert latest_alert["status"] in ["Watch", "Significant"]
    assert "top_signals" in latest_alert
    assert "parameter_shift" in latest_alert
    assert "reliability_level" in latest_alert
    assert latest_alert["divergence_score"] > 2.0
