"""Session State and Streaming Manager for CardioTwin X.

Coordinates data streaming, UKF estimation, divergence detection,
and explainability across Streamlit execution cycles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from cardiotwin.twin.data import (
    PATIENT_PRESETS,
    PatientHistory,
    generate_24h_series,
)
from cardiotwin.twin.divergence import DivergenceEngine, DivergenceResult
from cardiotwin.twin.estimator import CardiovascularUKF, EstimatorStepResult
from cardiotwin.twin.explain import ExplainabilityEngine, ExplanationReport
from cardiotwin.twin.model import WindkesselParams, simulate_beat


@dataclass
class TwinCycleStep:
    """Consolidated outputs from a single step across all twin components."""
    step_idx: int
    timestamp: datetime
    observed: Optional[np.ndarray]           # [HR, SBP, DBP]
    ukf_result: EstimatorStepResult
    divergence_result: DivergenceResult
    explanation_report: ExplanationReport
    episode: str
    is_anomaly: bool


class TwinSessionState:
    """Manages digital twin runtime state and synthetic telemetry feed."""

    def __init__(
        self,
        preset_key: str = "healthy",
        seed: int = 42,
    ) -> None:
        """Initialize session state with selected patient profile."""
        self.preset_key = preset_key
        self.seed = seed
        self.reset(preset_key=preset_key, seed=seed)

    def reset(self, preset_key: str = "healthy", seed: int = 42) -> None:
        """Reset digital twin components with fresh patient baseline."""
        self.preset_key = preset_key
        self.seed = seed
        preset = PATIENT_PRESETS.get(preset_key, PATIENT_PRESETS["healthy"])

        self.patient_history: PatientHistory = preset["history"]
        self.base_params: WindkesselParams = preset["base_params"]

        # Generate 24-hour synthetic stream
        self.series_df: pd.DataFrame = generate_24h_series(preset_key=preset_key, seed=seed)
        self.total_steps = len(self.series_df)
        self.current_step = 0

        # Initialize UKF at patient baseline
        x0 = np.array([
            self.base_params.Rp,
            self.base_params.C,
            self.base_params.SV,
            self.base_params.HR,
        ], dtype=np.float64)
        P0 = np.diag([0.15**2, 0.20**2, 8.0**2, 4.0**2])
        self.ukf = CardiovascularUKF(initial_state=x0, initial_cov=P0)

        # Initialize Divergence Engine and Explainability
        self.divergence_engine = DivergenceEngine(random_state=seed)
        self.explain_engine = ExplainabilityEngine(baseline_window=20)

        # Step history buffer
        self.steps: List[TwinCycleStep] = []
        self.whatif_events: List[Dict[str, Any]] = []
        self.alerts_log: List[Dict[str, Any]] = []

        # Playback control

        self.is_live: bool = True
        self.playback_speed: float = 1.0  # seconds between ticks

        # User-entered reminders and Care Circle contacts
        self.reminders: List[str] = [
            "Your reminder: Drink 500 mL water post-morning walk",
            "Your reminder: Evening relaxation breathing session at 20:30",
        ]
        self.care_circle: List[Dict[str, Any]] = [
            {
                "name": "Dr. Aris Thorne",
                "role": "Attending Cardiologist",
                "clinic": "Metropolitan Heart Institute",
                "phone": "+1 (555) 019-2834",
                "alert_enabled": True,
            },
            {
                "name": "Marcus Jenkins",
                "role": "Spouse / Primary Contact",
                "clinic": "Home",
                "phone": "+1 (555) 014-9921",
                "alert_enabled": True,
            },
        ]

        # Warm up first 10 steps so UI starts with active vitals immediately
        self.warmup(n_steps=15)

    def warmup(self, n_steps: int = 15) -> None:
        """Pre-populate initial steps to populate charts on first page load."""
        for _ in range(n_steps):
            if self.current_step < self.total_steps:
                self.step_forward()

    def step_forward(self, manual_obs: Optional[np.ndarray] = None) -> Optional[TwinCycleStep]:
        """Advance twin simulation by 1 minute with next synthetic or manual reading."""
        if manual_obs is not None:
            # User manual reading entry
            row_time = datetime.now()
            obs = np.asarray(manual_obs, dtype=np.float64)
            episode = "Manual User Reading"
            is_anomaly = False
        else:
            step_in_day = self.current_step % self.total_steps
            day_offset = self.current_step // self.total_steps
            row = self.series_df.iloc[step_in_day]
            row_time = row["timestamp"] + timedelta(days=int(day_offset))
            episode = str(row["episode"])
            is_anomaly = bool(row["is_anomaly"])

            if np.isnan(row["HR"]) or np.isnan(row["SBP"]) or np.isnan(row["DBP"]):
                obs = None
            else:
                obs = np.array([row["HR"], row["SBP"], row["DBP"]], dtype=np.float64)

            self.current_step += 1


        # 1. UKF Estimation Step
        ukf_res = self.ukf.step(obs)

        # 2. Divergence Evaluation Step
        div_res = self.divergence_engine.evaluate_step(ukf_res, self.ukf)

        # 3. Explainability Step
        exp_res = self.explain_engine.explain(ukf_res, div_res)

        step_bundle = TwinCycleStep(
            step_idx=len(self.steps),
            timestamp=row_time,
            observed=obs,
            ukf_result=ukf_res,
            divergence_result=div_res,
            explanation_report=exp_res,
            episode=episode,
            is_anomaly=is_anomaly,
        )
        self.steps.append(step_bundle)

        # 4. Check & record explainable alert if status is Watch or Significant
        rel_level, rel_reason, _ = self.compute_data_reliability()
        if div_res.status in ["Watch", "Significant"]:
            top_sig_str = ", ".join([
                f"{k}: {v:.0f}%"
                for k, v in sorted(
                    div_res.signal_contributions.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )
            ])
            self.alerts_log.append({
                "id": len(self.alerts_log) + 1,
                "step": step_bundle.step_idx,
                "timestamp": row_time.strftime("%H:%M:%S"),
                "status": div_res.status,
                "divergence_score": float(div_res.divergence_score),
                "top_signals": top_sig_str,
                "signal_contributions": div_res.signal_contributions,
                "parameter_shift": exp_res.primary_driver_description or exp_res.summary,
                "reliability_level": rel_level,
                "reliability_reason": rel_reason,
                "detector_agreement": div_res.detector_agreement,
                "episode": episode,
                "detailed_explanation": exp_res.detailed_explanation,
            })

        return step_bundle

    def log_whatif_event(self, scenario_name: str, details: str = "") -> None:
        """Log a user counterfactual experiment onto the timeline."""
        latest = self.get_latest_step()
        ts = latest.timestamp if latest else datetime.now()
        self.whatif_events.append({
            "step": latest.step_idx if latest else len(self.steps),
            "timestamp": ts,
            "scenario": scenario_name,
            "details": details,
        })

    def compute_data_reliability(self, window_size: int = 10) -> Tuple[str, str, Dict[str, Any]]:
        """Evaluate telemetry data reliability: High, Medium, or Low.

        Evaluates:
        1. Number of recent valid readings in window (dropout ratio).
        2. Immediate sensor dropout on current frame.
        3. Parameter covariance size: Tr(P) trace metric.
        4. Divergence persistence count.

        Returns:
            Tuple of (level: 'High'|'Medium'|'Low', explanation: str, metrics: dict).
        """
        if not self.steps:
            return "Medium", "Initializing stream telemetry...", {}

        recent = self.steps[-window_size:]
        n_recent = len(recent)
        valid_count = sum(1 for s in recent if s.observed is not None and not np.any(np.isnan(s.observed)))
        valid_ratio = valid_count / max(n_recent, 1)

        latest = self.steps[-1]
        is_current_missing = (latest.observed is None) or np.any(np.isnan(latest.observed))

        cov_trace = float(np.trace(latest.ukf_result.state_cov))
        div_persistence = latest.divergence_result.consecutive_high_count
        div_score = latest.divergence_result.divergence_score

        metrics = {
            "valid_count": valid_count,
            "valid_ratio": valid_ratio,
            "is_current_missing": is_current_missing,
            "cov_trace": cov_trace,
            "div_persistence": div_persistence,
            "div_score": div_score,
        }

        # 1. Low Reliability criteria
        if is_current_missing:
            return (
                "Low",
                "Sensor telemetry is currently offline (missing signal dropout).",
                metrics,
            )
        if valid_ratio < 0.60:
            return (
                "Low",
                f"Severe signal dropout: only {valid_count}/{n_recent} valid readings in recent window.",
                metrics,
            )
        if cov_trace > 85.0:
            return (
                "Low",
                f"High estimator uncertainty: parameter covariance trace ({cov_trace:.1f}) exceeds stability threshold.",
                metrics,
            )
        if div_persistence >= 5 or div_score > 5.0:
            return (
                "Low",
                f"Extreme prolonged divergence (persistence: {div_persistence} beats, score: {div_score:.1f}).",
                metrics,
            )

        # 2. Medium Reliability criteria
        if valid_ratio < 0.85:
            return (
                "Medium",
                f"Intermittent telemetry: {n_recent - valid_count} frame dropouts in last {n_recent} readings.",
                metrics,
            )
        if cov_trace > 45.0:
            return (
                "Medium",
                f"Moderate parameter uncertainty (covariance trace: {cov_trace:.1f}) during dynamic adaptation.",
                metrics,
            )
        if div_persistence >= 2:
            return (
                "Medium",
                f"Elevated innovation divergence being monitored for persistence ({div_persistence} steps).",
                metrics,
            )

        # 3. High Reliability
        return (
            "High",
            "Continuous valid telemetry with well-conditioned posterior covariance.",
            metrics,
        )


    def get_latest_step(self) -> Optional[TwinCycleStep]:
        """Return the most recent processed step."""
        if not self.steps:
            return None
        return self.steps[-1]

    def get_telemetry_dataframe(self, max_points: int = 120) -> pd.DataFrame:
        """Produce dataframe of recent time series for UI charting."""
        if not self.steps:
            return pd.DataFrame()

        recent = self.steps[-max_points:]
        records = []
        for s in recent:
            obs = s.observed
            hr_obs = float(obs[0]) if obs is not None else np.nan
            sbp_obs = float(obs[1]) if obs is not None else np.nan
            dbp_obs = float(obs[2]) if obs is not None else np.nan

            bands = s.divergence_result.ensemble_bands
            rec = {
                "step": s.step_idx,
                "timestamp": s.timestamp,
                "HR_obs": hr_obs,
                "SBP_obs": sbp_obs,
                "DBP_obs": dbp_obs,
                "HR_pred": float(s.ukf_result.predicted_obs[0]),
                "SBP_pred": float(s.ukf_result.predicted_obs[1]),
                "DBP_pred": float(s.ukf_result.predicted_obs[2]),
                "SBP_p5": bands.get("SBP", {}).get("p5", np.nan),
                "SBP_p95": bands.get("SBP", {}).get("p95", np.nan),
                "DBP_p5": bands.get("DBP", {}).get("p5", np.nan),
                "DBP_p95": bands.get("DBP", {}).get("p95", np.nan),
                "HR_p5": bands.get("HR", {}).get("p5", np.nan),
                "HR_p95": bands.get("HR", {}).get("p95", np.nan),
                "divergence_score": s.divergence_result.divergence_score,
                "status": s.divergence_result.status,
                "Rp_est": float(s.ukf_result.state_mean[0]),
                "Rp_std": float(s.ukf_result.state_std[0]),
                "C_est": float(s.ukf_result.state_mean[1]),
                "C_std": float(s.ukf_result.state_std[1]),
                "SV_est": float(s.ukf_result.state_mean[2]),
                "SV_std": float(s.ukf_result.state_std[2]),
                "HR_est": float(s.ukf_result.state_mean[3]),
                "HR_std": float(s.ukf_result.state_std[3]),
                "episode": s.episode,
            }
            records.append(rec)

        return pd.DataFrame(records)
