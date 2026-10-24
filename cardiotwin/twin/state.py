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
        return step_bundle

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
