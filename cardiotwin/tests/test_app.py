"""Integration tests for Streamlit application components and session manager."""

import pytest
import numpy as np
import pandas as pd
from cardiotwin.twin.state import TwinSessionState
from cardiotwin.twin.whatif import WHATIF_SCENARIOS, ScenarioDelta, run_whatif_simulation
from cardiotwin.twin.model import WindkesselParams


def test_twin_session_state_lifecycle():
    """TwinSessionState must initialize cleanly and step forward smoothly."""
    state = TwinSessionState(preset_key="healthy", seed=101)

    assert state.patient_history.name == "Sarah Jenkins"
    assert len(state.steps) >= 15  # Warmed up

    # Advance 5 steps
    for _ in range(5):
        step_res = state.step_forward()
        assert step_res is not None
        assert step_res.divergence_result is not None
        assert step_res.explanation_report is not None

    df_telemetry = state.get_telemetry_dataframe(max_points=30)
    assert not df_telemetry.empty
    assert "SBP_pred" in df_telemetry.columns
    assert "SBP_p95" in df_telemetry.columns
    assert "divergence_score" in df_telemetry.columns


def test_manual_reading_injection():
    """Manual user measurements must be assimilated without error."""
    state = TwinSessionState(preset_key="healthy", seed=101)
    manual_reading = np.array([80.0, 130.0, 85.0])

    step_res = state.step_forward(manual_obs=manual_reading)
    assert step_res is not None
    assert step_res.episode == "Manual User Reading"
    assert step_res.observed is not None
    np.testing.assert_allclose(step_res.observed, manual_reading)


def test_whatif_scenarios_all_valid():
    """All what-if scenario presets must execute and produce comparative metrics."""
    params = WindkesselParams()

    for scen_key in WHATIF_SCENARIOS:
        comp = run_whatif_simulation(current_params=params, scenario_key=scen_key)
        assert comp.scenario_name is not None
        assert "Illustrative simulation" in comp.disclaimer
        assert comp.output_after.sbp > 0
        assert comp.output_after.dbp > 0
        assert comp.uncertainty_bands_after["SBP"]["p95"] >= comp.uncertainty_bands_after["SBP"]["p5"]


def test_whatif_custom_sliders():
    """Custom parameter deltas should be applied accurately."""
    params = WindkesselParams()
    custom = ScenarioDelta(
        name="Test Vasoconstriction",
        description="Testing resistance increase",
        rp_multiplier=1.40,
        c_multiplier=0.80,
        sv_delta=10.0,
        hr_delta=15.0,
    )
    comp = run_whatif_simulation(current_params=params, scenario_key="custom", custom_delta=custom)
    assert comp.output_after.sbp > comp.output_before.sbp
    assert comp.output_after.hr > comp.output_before.hr


def test_reminders_no_prescription_and_proper_labels():
    """User reminders must not prescribe aspirin and patient label must be synthetic sample."""
    state = TwinSessionState()
    for reminder in state.reminders:
        assert "Aspirin" not in reminder
        assert reminder.startswith("Your reminder:")
    assert state.patient_history.label == "Sample patient (synthetic data)"


def test_explanation_states_why_twin_differs():
    """Explanation summary must convey why the twin differs from observations."""
    state = TwinSessionState()
    # Inject divergent reading
    res = state.step_forward(manual_obs=np.array([75.0, 180.0, 115.0]))
    assert res is not None
    summary = res.explanation_report.summary
    assert "Twin differs from observations" in summary or "vasoconstriction" in summary.lower() or "resistance" in summary.lower()


def test_3d_heart_anatomical_and_lubdub_generation():
    """3D anatomical heart HTML generator must incorporate BP-driven lub-dub and vascular anatomy."""
    from cardiotwin.twin.heart3d import generate_3d_heart_html

    html = generate_3d_heart_html(
        sbp=145.0,
        dbp=92.0,
        hr=84.0,
        sv=68.0,
        div_status="Significant",
        div_score=3.8,
    )
    assert len(html) > 10000
    assert "three.min.js" in html
    assert "OrbitControls" in html
    assert "ventriclesGeom" in html
    assert "archCurve" in html  # Aortic arch
    assert "ptCurve" in html    # Pulmonary trunk
    assert "coronaryGroup" in html
    assert "playLubSound" in html
    assert "playDubSound" in html
    assert "phase-lub" in html
    assert "phase-dub" in html
    assert "145" in html  # SBP passed in config
    assert "92" in html   # DBP passed in config



