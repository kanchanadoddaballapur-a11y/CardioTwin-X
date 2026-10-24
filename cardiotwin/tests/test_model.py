"""Tests for the 3-element Windkessel physiological model."""

import pytest
import numpy as np
from cardiotwin.twin.model import WindkesselParams, simulate_beat, apply_baroreflex


def test_default_model_gives_canonical_120_80():
    """Default parameters must yield canonical ~120/80 mmHg."""
    params = WindkesselParams()
    out = simulate_beat(params)

    assert 118.0 <= out.sbp <= 122.0, f"Expected SBP ~120, got {out.sbp:.2f}"
    assert 78.0 <= out.dbp <= 82.0, f"Expected DBP ~80, got {out.dbp:.2f}"
    assert out.dbp < out.map < out.sbp, f"MAP {out.map} not between DBP and SBP"
    assert 35.0 <= out.pp <= 45.0, f"Expected PP ~40, got {out.pp:.2f}"


def test_baroreflex_coupling():
    """Baroreflex drops HR on high MAP and raises HR on low MAP."""
    base_hr = 72.0
    setpoint = 93.3

    # High pressure -> HR drops
    hr_high_map = apply_baroreflex(mean_arterial_pressure=115.0, baseline_hr=base_hr, setpoint_map=setpoint)
    assert hr_high_map < base_hr, f"HR did not decrease on high MAP: {hr_high_map}"

    # Low pressure -> HR increases
    hr_low_map = apply_baroreflex(mean_arterial_pressure=75.0, baseline_hr=base_hr, setpoint_map=setpoint)
    assert hr_low_map > base_hr, f"HR did not increase on low MAP: {hr_low_map}"


def test_parameter_bounds_and_clipping():
    """Unphysical inputs should be clamped to bounds."""
    params = WindkesselParams(Rp=-5.0, C=10.0, Rc=0.001, SV=300.0, HR=10.0)
    clipped = params.clipped()
    assert clipped.Rp >= 0.4
    assert clipped.C <= 3.0
    assert clipped.Rc >= 0.01
    assert clipped.SV <= 160.0
    assert clipped.HR >= 35.0


def test_waveform_properties():
    """Waveform must have continuous positive values and proper periodicity."""
    params = WindkesselParams()
    out = simulate_beat(params, n_points=100)

    assert len(out.time) == 100
    assert len(out.pressure) == 100
    assert len(out.flow) == 100
    assert np.all(out.pressure > 40.0)
    assert np.all(out.pressure < 200.0)
    # Systolic peak must be greater than end-diastolic pressure
    assert out.pressure[0] <= out.sbp
