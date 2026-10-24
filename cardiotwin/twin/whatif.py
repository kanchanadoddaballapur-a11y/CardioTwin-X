"""Cardiovascular What-If Simulation Engine.

Simulates physiological perturbation scenarios on top of the patient's
current digital twin state, predicting how interventions or physiological
stressors affect hemodynamics and pressure waveforms.

Pre-configured scenarios:
- Moderate Aerobic Exercise (vasodilation + inotropy + chronotropy)
- Acute Stress (sympathetic vasoconstriction + chronotropy)
- High Salt Intake (volume expansion + elevated Rp)
- Vasodilator-Like Effect (arteriolar relaxation / lowered Rp)
- Arterial Stiffening (loss of compliance C, wide pulse pressure)
- Custom Slider Adjustments

All outputs are strictly labeled:
"Illustrative simulation, not medical advice."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from cardiotwin.twin.model import WindkesselOutput, WindkesselParams, simulate_beat

MEDICAL_DISCLAIMER: str = (
    "Illustrative simulation, not medical advice. "
    "Demonstrates theoretical cardiovascular dynamics using a mathematical model."
)


@dataclass
class ScenarioDelta:
    """Multipliers and offsets applied to twin parameters."""
    name: str
    description: str
    rp_multiplier: float = 1.0     # Multiplier on peripheral resistance Rp
    c_multiplier: float = 1.0      # Multiplier on compliance C
    sv_delta: float = 0.0          # Additive delta in stroke volume (mL)
    hr_delta: float = 0.0          # Additive delta in heart rate (bpm)
    f_sys_delta: float = 0.0       # Additive change in systolic ejection fraction


WHATIF_SCENARIOS: Dict[str, ScenarioDelta] = {
    "moderate_exercise": ScenarioDelta(
        name="Moderate Aerobic Exercise",
        description="Increased metabolic demand drives chronotropic surge, enhanced stroke volume, and muscular vasodilation.",
        rp_multiplier=0.72,
        c_multiplier=0.92,
        sv_delta=18.0,
        hr_delta=42.0,
    ),
    "acute_stress": ScenarioDelta(
        name="Acute Stress / Sympathetic Surge",
        description="Fight-or-flight response induces peripheral arteriolar constriction, elevated heart rate, and increased afterload.",
        rp_multiplier=1.28,
        c_multiplier=0.90,
        sv_delta=5.0,
        hr_delta=24.0,
    ),
    "high_salt": ScenarioDelta(
        name="High Salt Intake (Fluid Retention)",
        description="Excess sodium induces fluid retention and vascular hyper-reactivity, increasing systemic resistance and filling volume.",
        rp_multiplier=1.22,
        c_multiplier=0.88,
        sv_delta=7.0,
        hr_delta=-2.0,
    ),
    "vasodilator": ScenarioDelta(
        name="Vasodilator-Like Relaxation Effect",
        description="Arteriolar smooth muscle relaxation drops systemic vascular resistance, reducing cardiac afterload.",
        rp_multiplier=0.74,
        c_multiplier=1.14,
        sv_delta=0.0,
        hr_delta=4.0,
    ),
    "arterial_stiffening": ScenarioDelta(
        name="Progressive Arterial Stiffening",
        description="Loss of aortic elastin reduces Windkessel buffering compliance, creating isolated systolic hypertension and high pulse pressure.",
        rp_multiplier=1.10,
        c_multiplier=0.52,
        sv_delta=0.0,
        hr_delta=0.0,
    ),
}


@dataclass
class WhatIfComparison:
    """Comparison bundle between baseline twin state and simulated scenario."""
    scenario_name: str
    description: str
    disclaimer: str
    params_before: WindkesselParams
    params_after: WindkesselParams
    output_before: WindkesselOutput
    output_after: WindkesselOutput
    uncertainty_bands_after: Dict[str, Dict[str, float]]
    delta_sbp: float
    delta_dbp: float
    delta_map: float
    delta_hr: float
    delta_pp: float

    def to_summary_dict(self) -> Dict[str, Any]:
        """Convert metrics to summary dictionary."""
        return {
            "scenario": self.scenario_name,
            "disclaimer": self.disclaimer,
            "SBP_before": float(self.output_before.sbp),
            "SBP_after": float(self.output_after.sbp),
            "delta_SBP": float(self.delta_sbp),
            "DBP_before": float(self.output_before.dbp),
            "DBP_after": float(self.output_after.dbp),
            "delta_DBP": float(self.delta_dbp),
            "MAP_before": float(self.output_before.map),
            "MAP_after": float(self.output_after.map),
            "delta_MAP": float(self.delta_map),
            "HR_before": float(self.output_before.hr),
            "HR_after": float(self.output_after.hr),
            "delta_HR": float(self.delta_hr),
            "PP_before": float(self.output_before.pp),
            "PP_after": float(self.output_after.pp),
            "delta_PP": float(self.delta_pp),
        }


def run_whatif_simulation(
    current_params: WindkesselParams,
    scenario_key: str = "moderate_exercise",
    custom_delta: Optional[ScenarioDelta] = None,
    current_cov: Optional[np.ndarray] = None,
    n_ensemble: int = 40,
) -> WhatIfComparison:
    """Run comparative simulation before vs after a physiological perturbation.

    Args:
        current_params: Active twin parameter baseline.
        scenario_key: Key in WHATIF_SCENARIOS or 'custom'.
        custom_delta: Custom ScenarioDelta if scenario_key == 'custom'.
        current_cov: Posterior covariance matrix P for uncertainty propagation.
        n_ensemble: Monte Carlo sample count for scenario predictive bands.

    Returns:
        WhatIfComparison containing waveforms, metrics, and uncertainty bands.
    """
    if scenario_key == "custom" and custom_delta is not None:
        delta = custom_delta
    else:
        delta = WHATIF_SCENARIOS.get(scenario_key, WHATIF_SCENARIOS["moderate_exercise"])

    # Baseline simulation
    p_before = current_params.clipped()
    out_before = simulate_beat(p_before)

    # Perturbed scenario parameters
    p_after = WindkesselParams(
        Rp=p_before.Rp * delta.rp_multiplier,
        C=p_before.C * delta.c_multiplier,
        Rc=p_before.Rc,
        SV=p_before.SV + delta.sv_delta,
        HR=p_before.HR + delta.hr_delta,
        f_sys=p_before.f_sys + delta.f_sys_delta,
    ).clipped()

    out_after = simulate_beat(p_after)

    # Propagate uncertainty bands on scenario outcome
    rng = np.random.default_rng(123)
    vec_after = p_after.to_vector()[:4]

    if current_cov is not None and current_cov.shape == (4, 4):
        # Scale covariance with parameter shifts
        scale_diag = np.array([delta.rp_multiplier, delta.c_multiplier, 1.0, 1.0])
        scen_cov = np.diag(scale_diag) @ current_cov @ np.diag(scale_diag)
        try:
            samples = rng.multivariate_normal(vec_after, scen_cov, size=n_ensemble, check_valid="ignore")
        except Exception:
            std = np.sqrt(np.maximum(np.diag(scen_cov), 1e-4))
            samples = vec_after + rng.normal(0, 1, size=(n_ensemble, 4)) * std
    else:
        # Default nominal standard error
        std_nominal = np.array([0.08, 0.08, 4.0, 3.0])
        samples = vec_after + rng.normal(0, 1, size=(n_ensemble, 4)) * std_nominal

    # Evaluate simulated outputs for ensemble
    ens_sbp, ens_dbp, ens_hr, ens_map = [], [], [], []
    for s in samples:
        # Fast hemodynamic approximation for ensemble points
        rp_s = float(np.clip(s[0], 0.4, 3.0))
        c_s = float(np.clip(s[1], 0.3, 3.0))
        sv_s = float(np.clip(s[2], 30.0, 160.0))
        hr_s = float(np.clip(s[3], 35.0, 200.0))
        co_s = (sv_s * hr_s) / 60.0
        map_s = co_s * (rp_s + 0.05)
        pp_s = (sv_s / c_s) * 0.75
        ens_sbp.append(map_s + 0.60 * pp_s)
        ens_dbp.append(map_s - 0.40 * pp_s)
        ens_map.append(map_s)
        ens_hr.append(hr_s)

    bands_after = {
        "SBP": {
            "p5": float(np.percentile(ens_sbp, 5)),
            "median": float(np.median(ens_sbp)),
            "p95": float(np.percentile(ens_sbp, 95)),
        },
        "DBP": {
            "p5": float(np.percentile(ens_dbp, 5)),
            "median": float(np.median(ens_dbp)),
            "p95": float(np.percentile(ens_dbp, 95)),
        },
        "MAP": {
            "p5": float(np.percentile(ens_map, 5)),
            "median": float(np.median(ens_map)),
            "p95": float(np.percentile(ens_map, 95)),
        },
        "HR": {
            "p5": float(np.percentile(ens_hr, 5)),
            "median": float(np.median(ens_hr)),
            "p95": float(np.percentile(ens_hr, 95)),
        },
    }

    return WhatIfComparison(
        scenario_name=delta.name,
        description=delta.description,
        disclaimer=MEDICAL_DISCLAIMER,
        params_before=p_before,
        params_after=p_after,
        output_before=out_before,
        output_after=out_after,
        uncertainty_bands_after=bands_after,
        delta_sbp=float(out_after.sbp - out_before.sbp),
        delta_dbp=float(out_after.dbp - out_before.dbp),
        delta_map=float(out_after.map - out_before.map),
        delta_hr=float(out_after.hr - out_before.hr),
        delta_pp=float(out_after.pp - out_before.pp),
    )
