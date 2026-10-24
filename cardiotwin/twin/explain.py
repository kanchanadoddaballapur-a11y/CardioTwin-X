"""Physiological Explainability Engine.

Translates mathematical divergence and posterior parameter shifts into
clear, plain-English physiological explanations for clinicians and users.

Functions:
- Computes percentage change and sigma deviations vs a rolling baseline.
- Identifies the driving physiological parameter shift (Rp, C, SV, HR).
- Generates plain-English narrative with confidence ratings.
- Prepares signal contribution breakdowns for visual bar charting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from cardiotwin.twin.divergence import DivergenceResult
from cardiotwin.twin.estimator import EstimatorStepResult


@dataclass
class ExplanationReport:
    """Narrative and statistical report explaining digital twin divergence."""
    time_step: int
    status: str
    divergence_score: float
    summary: str                           # One-line headline
    detailed_explanation: str              # Plain-English narrative
    confidence_label: str                  # 'High Confidence', 'Moderate Confidence', 'Preliminary'
    primary_driver_param: str              # e.g. 'Rp', 'C', 'SV', 'HR'
    primary_driver_description: str        # e.g. 'Peripheral resistance is up 18% vs baseline (2.1 sigma)'
    parameter_deltas: Dict[str, Dict[str, float]] # Per-param {pct_change, sigma, current, baseline}
    signal_contributions: Dict[str, float] # {HR: %, SBP: %, DBP: %}
    is_missing_data: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "time_step": self.time_step,
            "status": self.status,
            "divergence_score": float(self.divergence_score),
            "summary": self.summary,
            "detailed_explanation": self.detailed_explanation,
            "confidence_label": self.confidence_label,
            "primary_driver_param": self.primary_driver_param,
            "primary_driver_description": self.primary_driver_description,
            "parameter_deltas": self.parameter_deltas,
            "signal_contributions": self.signal_contributions,
            "is_missing_data": self.is_missing_data,
        }


class ExplainabilityEngine:
    """Analyzes posterior state evolution to generate physiological explanations."""

    PARAM_NAMES = {
        "Rp": "Peripheral vascular resistance (arteriolar tone)",
        "C": "Arterial compliance (vessel elasticity)",
        "SV": "Stroke volume (blood ejected per contraction)",
        "HR": "Heart rate (chronotropic drive)",
    }

    def __init__(self, baseline_window: int = 30) -> None:
        """Initialize explainability engine.

        Args:
            baseline_window: Number of initial normal steps used to compute baseline mean/std.
        """
        self.baseline_window = int(baseline_window)
        self.baseline_means: Optional[np.ndarray] = None
        self.baseline_stds: Optional[np.ndarray] = None
        self.collected_normal_states: List[np.ndarray] = []

    def update_baseline(self, state_mean: np.ndarray) -> None:
        """Add normal state observation to baseline accumulator."""
        if len(self.collected_normal_states) < self.baseline_window:
            self.collected_normal_states.append(state_mean.copy())
            if len(self.collected_normal_states) >= 5:
                arr = np.array(self.collected_normal_states)
                self.baseline_means = np.mean(arr, axis=0)
                self.baseline_stds = np.std(arr, axis=0) + 1e-4

    def explain(
        self,
        ukf_res: EstimatorStepResult,
        div_res: DivergenceResult,
    ) -> ExplanationReport:
        """Generate comprehensive plain-English explanation for current step."""
        t_step = ukf_res.time_step

        # If data is missing / sensor dropout
        if div_res.is_missing_data or not ukf_res.update_performed:
            return ExplanationReport(
                time_step=t_step,
                status="Sensor Dropout",
                divergence_score=0.0,
                summary="Sensor telemetry temporarily unavailable.",
                detailed_explanation="No incoming optical or pressure waveform detected. The digital twin continues internal hemodynamics without measurement update.",
                confidence_label="Uncertain (Sensor Offline)",
                primary_driver_param="None",
                primary_driver_description="Sensor dropout",
                parameter_deltas={},
                signal_contributions={"HR": 0.0, "SBP": 0.0, "DBP": 0.0},
                is_missing_data=True,
            )

        # Baseline calculation
        if self.baseline_means is None:
            self.baseline_means = ukf_res.state_mean.copy()
            self.baseline_stds = np.maximum(ukf_res.state_std.copy(), 0.01)
        elif div_res.status == "Normal":
            self.update_baseline(ukf_res.state_mean)

        cur_mean = ukf_res.state_mean
        cur_std = ukf_res.state_std
        base_mean = self.baseline_means
        base_std = self.baseline_stds

        param_keys = ["Rp", "C", "SV", "HR"]
        param_deltas: Dict[str, Dict[str, float]] = {}

        max_sigma = -1.0
        primary_param = "Rp"
        primary_desc = ""

        for idx, key in enumerate(param_keys):
            pct_chg = ((cur_mean[idx] - base_mean[idx]) / max(abs(base_mean[idx]), 1e-6)) * 100.0
            comb_std = np.sqrt(base_std[idx] ** 2 + cur_std[idx] ** 2)
            sigma_dist = (cur_mean[idx] - base_mean[idx]) / max(comb_std, 1e-6)

            param_deltas[key] = {
                "current": float(cur_mean[idx]),
                "baseline": float(base_mean[idx]),
                "pct_change": float(pct_chg),
                "sigma": float(sigma_dist),
                "uncertainty": float(cur_std[idx]),
            }

            if abs(sigma_dist) > max_sigma:
                max_sigma = abs(sigma_dist)
                primary_param = key
                direction = "up" if pct_chg > 0 else "down"
                primary_desc = f"{self.PARAM_NAMES[key]} estimate is {direction} {abs(pct_chg):.1f}% vs baseline ({abs(sigma_dist):.1f} sigma)"

        # Confidence label derivation based on UKF posterior uncertainty
        # If relative standard error of estimates is < 8% -> High Confidence
        mean_rel_err = np.mean(cur_std / np.maximum(cur_mean, 1e-3))
        if mean_rel_err < 0.08 and ukf_res.time_step > 5:
            confidence = "High Confidence (Stable Twin State)"
        elif mean_rel_err < 0.16:
            confidence = "Moderate Confidence"
        else:
            confidence = "Preliminary (Adapting)"

        # Synthesize plain-English explanation
        if div_res.status == "Significant":
            # Clinically significant divergence
            top_signal = max(div_res.signal_contributions.items(), key=lambda item: item[1])[0]
            sig_pct = div_res.signal_contributions[top_signal]

            if primary_param == "Rp":
                if param_deltas["Rp"]["pct_change"] > 0:
                    summary = f"Systemic vasoconstriction detected ({primary_desc})."
                    detail = (
                        f"The digital twin identifies a significant elevation in peripheral vascular resistance (Rp up {param_deltas['Rp']['pct_change']:.1f}%). "
                        f"This hemodynamic pattern accounts for {sig_pct:.0f}% of observed blood pressure divergence, characteristic of acute sympathetic tone, sodium excess, or renovascular response."
                    )
                else:
                    summary = f"Peripheral vasodilation detected ({primary_desc})."
                    detail = (
                        f"Peripheral resistance dropped {abs(param_deltas['Rp']['pct_change']):.1f}%, indicating systemic arteriolar relaxation or vasodilator effect."
                    )
            elif primary_param == "C":
                summary = f"Arterial stiffening detected ({primary_desc})."
                detail = (
                    f"Arterial compliance C dropped by {abs(param_deltas['C']['pct_change']):.1f}%, causing widening of pulse pressure and higher systolic loading."
                )
            elif primary_param == "HR":
                summary = f"Pronounced chronotropic shift ({primary_desc})."
                detail = (
                    f"Heart rate shifted by {param_deltas['HR']['pct_change']:.1f}%, dominating observed telemetry divergence ({sig_pct:.0f}% contribution from {top_signal})."
                )
            else:
                summary = f"Stroke volume shift ({primary_desc})."
                detail = (
                    f"Stroke volume varied by {param_deltas['SV']['pct_change']:.1f}%, altering cardiac output drive."
                )

        elif div_res.status == "Watch":
            summary = f"Mild hemodynamic fluctuation ({primary_desc})."
            detail = (
                f"Slight physiological drift observed (Divergence score: {div_res.divergence_score:.2f}). "
                f"Parameters remain within tolerable bounds; continuing close observation."
            )
        else:
            summary = "Digital twin in close physiological equilibrium."
            detail = (
                "Observed cardiovascular measurements align closely with the calibrated digital twin. "
                "Hemodynamic parameters and autoregulation reflect stable homeostatic balance."
            )

        return ExplanationReport(
            time_step=t_step,
            status=div_res.status,
            divergence_score=div_res.divergence_score,
            summary=summary,
            detailed_explanation=detail,
            confidence_label=confidence,
            primary_driver_param=primary_param,
            primary_driver_description=primary_desc,
            parameter_deltas=param_deltas,
            signal_contributions=div_res.signal_contributions,
            is_missing_data=False,
        )
