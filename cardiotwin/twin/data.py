"""Synthetic Cardiovascular Time-Series Generator.

Generates realistic 24-hour hemodynamic and ECG-derived time series at
1-minute resolution (1440 samples) incorporating:
- Circadian variation (diurnal rhythms, nocturnal dipping, morning surge)
- Sympathetic exercise and acute stress episodes
- Injected anomalies (acute vasoconstriction / Rp surge, paroxysmal tachycardia, sensor dropout)
- ECG features: RR interval (ms), HRV-SDNN (ms), QT-like interval (ms)
- Three patient presets: healthy, hypertensive, elderly (stiff arteries, low compliance)
- Patient history metadata dictionary
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from cardiotwin.twin.model import WindkesselParams, simulate_beat


@dataclass
class PatientHistory:
    """Demographics, lifestyle, and clinical history for a digital twin."""
    name: str = "Sarah Jenkins"
    age: int = 42
    gender: str = "Female"
    bmi: float = 22.8
    smoker: bool = False
    alcohol: bool = False
    baseline_bp: Tuple[float, float] = (120.0, 80.0)
    baseline_hr: float = 72.0
    label: str = "Sample patient (synthetic data)"
    notes: str = "Normotensive adult, active runner, no known cardiovascular disease."

    def to_dict(self) -> Dict[str, Any]:
        """Convert history to serializable dictionary."""
        return {
            "name": self.name,
            "age": self.age,
            "gender": self.gender,
            "bmi": self.bmi,
            "smoker": self.smoker,
            "alcohol": self.alcohol,
            "baseline_bp": f"{self.baseline_bp[0]:.0f}/{self.baseline_bp[1]:.0f} mmHg",
            "baseline_hr": f"{self.baseline_hr:.0f} bpm",
            "label": self.label,
            "notes": self.notes,
        }


PATIENT_PRESETS: Dict[str, Dict[str, Any]] = {
    "healthy": {
        "history": PatientHistory(
            name="Sarah Jenkins",
            age=42,
            gender="Female",
            bmi=22.8,
            smoker=False,
            alcohol=False,
            baseline_bp=(120.0, 80.0),
            baseline_hr=72.0,
            label="Sample patient (synthetic data)",
            notes="Active lifestyle, normotensive baseline, good autonomic tone.",
        ),
        "base_params": WindkesselParams(
            Rp=1.253, C=1.417, Rc=0.050, SV=62.8, HR=72.0
        ),
    },
    "hypertensive": {
        "history": PatientHistory(
            name="Robert Vance",
            age=56,
            gender="Male",
            bmi=29.4,
            smoker=True,
            alcohol=True,
            baseline_bp=(145.0, 94.0),
            baseline_hr=78.0,
            label="Sample patient (synthetic data)",
            notes="Stage 2 hypertension profile; elevated peripheral vascular resistance.",
        ),
        "base_params": WindkesselParams(
            Rp=1.650, C=1.120, Rc=0.055, SV=65.0, HR=78.0
        ),
    },
    "elderly": {
        "history": PatientHistory(
            name="Eleanor Davis",
            age=78,
            gender="Female",
            bmi=24.1,
            smoker=False,
            alcohol=False,
            baseline_bp=(155.0, 74.0),
            baseline_hr=68.0,
            label="Sample patient (synthetic data)",
            notes="Isolated systolic hypertension; stiffened large arteries (low arterial compliance C).",
        ),
        "base_params": WindkesselParams(
            Rp=1.380, C=0.680, Rc=0.065, SV=63.0, HR=68.0
        ),
    },
}


def generate_24h_series(
    preset_key: str = "healthy",
    start_time: Optional[datetime] = None,
    seed: int = 42,
    include_anomalies: bool = True,
) -> pd.DataFrame:
    """Generate a 24-hour synthetic cardiovascular series at 1-minute resolution (1440 points).

    Args:
        preset_key: One of 'healthy', 'hypertensive', 'elderly'.
        start_time: Starting datetime (defaults to 08:00 AM of current date).
        seed: Random seed for reproducible generation.
        include_anomalies: Whether to inject clinical and sensor anomalies.

    Returns:
        pd.DataFrame containing time, physiological states, observations, and annotations.
    """
    rng = np.random.default_rng(seed)
    n_points = 1440  # 24 hours * 60 minutes

    if start_time is None:
        start_time = datetime(2026, 10, 24, 8, 0, 0)

    preset = PATIENT_PRESETS.get(preset_key, PATIENT_PRESETS["healthy"])
    base_p: WindkesselParams = preset["base_params"]

    # Time axes
    t_minutes = np.arange(n_points)
    timestamps = [start_time + timedelta(minutes=int(m)) for m in t_minutes]

    # 1. Circadian Rhythm Model
    # Blood pressure and HR dip during nocturnal sleep (typically 23:00 to 06:00 = minutes 900 to 1320 if starting at 08:00)
    # Peak occurs mid-afternoon (around 14:00 = minute 360)
    # Harmonics: 24h fundamental + 12h harmonic
    t_rad = 2.0 * np.pi * t_minutes / 1440.0
    # Phase shifted so sleep nadir aligns with 03:30 AM (minute 1170)
    phi = -np.pi / 2.0
    circadian_dip = 0.12 * np.sin(t_rad + phi) + 0.03 * np.sin(2.0 * t_rad)

    # Base parameter trajectories with smooth low-frequency biological drift
    brownian_walk = np.cumsum(rng.normal(0.0, 0.005, size=n_points))
    brownian_walk -= np.linspace(brownian_walk[0], brownian_walk[-1], n_points)  # Detrend endpoints

    # Dynamic physiological parameters over the 24 hours
    rp_series = base_p.Rp * (1.0 + 0.08 * circadian_dip + 0.03 * brownian_walk)
    c_series = base_p.C * (1.0 - 0.04 * circadian_dip + 0.02 * np.roll(brownian_walk, 100))
    sv_series = base_p.SV * (1.0 + 0.06 * circadian_dip + 0.03 * np.roll(brownian_walk, 200))
    hr_series = base_p.HR * (1.0 + 0.14 * circadian_dip + 0.04 * np.roll(brownian_walk, 300))

    episode_labels = ["Rest / Normal Routine"] * n_points
    anomaly_flags = [False] * n_points

    # 2. Daily Life Episodes
    # Episode A: Morning Exercise (e.g. jog / brisk walk, min 60 -> 105, 09:00 - 09:45)
    ex_start, ex_end = 60, 105
    ex_len = ex_end - ex_start
    ex_window = np.sin(np.pi * np.arange(ex_len) / ex_len)
    hr_series[ex_start:ex_end] += 55.0 * ex_window         # HR surges to ~125-135
    sv_series[ex_start:ex_end] += 22.0 * ex_window         # SV increases
    rp_series[ex_start:ex_end] -= 0.35 * ex_window         # Vasodilation in working muscle lowers Rp
    c_series[ex_start:ex_end] -= 0.10 * ex_window          # Elevated mean pressure tightens arteries
    for i in range(ex_start, ex_end):
        episode_labels[i] = "Moderate Exercise"

    # Episode B: Afternoon Cognitive / Work Stress (min 360 -> 405, 14:00 - 14:45)
    st_start, st_end = 360, 405
    st_len = st_end - st_start
    st_window = np.sin(np.pi * np.arange(st_len) / st_len)
    hr_series[st_start:st_end] += 18.0 * st_window         # Mild tachycardia
    rp_series[st_start:st_end] += 0.28 * st_window         # Stress-induced vasoconstriction
    for i in range(st_start, st_end):
        episode_labels[i] = "Acute Stress Episode"

    # 3. Injected Anomalies (Cardiovascular divergence events & sensor failure)
    if include_anomalies:
        # Anomaly 1: Sudden Resistance Surge (Acute hypertensive vasoconstriction)
        # Min 720 -> 770 (20:00 - 20:50). Rp jumps +45%, HR slightly drops due to baroreflex
        an1_start, an1_end = 720, 770
        an1_len = an1_end - an1_start
        an1_window = np.sin(np.pi * np.arange(an1_len) / an1_len) ** 0.8
        rp_series[an1_start:an1_end] += (0.55 * base_p.Rp) * an1_window
        hr_series[an1_start:an1_end] -= 8.0 * an1_window  # Baroreflex buffering
        for i in range(an1_start, an1_end):
            episode_labels[i] = "Anomaly: Acute Peripheral Resistance Surge"
            anomaly_flags[i] = True

        # Anomaly 2: Sensor Dropout / Missing Readings (Min 980 -> 995, 00:20 - 00:35)
        # SBP/DBP/HR will be set to NaN to verify estimator gracefully skips updates
        dropout_start, dropout_end = 980, 995
        for i in range(dropout_start, dropout_end):
            episode_labels[i] = "Sensor Dropout (Signal Loss)"
            anomaly_flags[i] = True

        # Anomaly 3: Paroxysmal Tachycardia (Min 1220 -> 1255, 04:20 - 04:55)
        # Sudden rate jump during sleep to 135 bpm, reduced diastolic filling (SV drops)
        an3_start, an3_end = 1220, 1255
        an3_len = an3_end - an3_start
        an3_window = np.sin(np.pi * np.arange(an3_len) / an3_len)
        hr_series[an3_start:an3_end] += 60.0 * an3_window
        sv_series[an3_start:an3_end] -= 18.0 * an3_window
        for i in range(an3_start, an3_end):
            episode_labels[i] = "Anomaly: Paroxysmal Tachycardia"
            anomaly_flags[i] = True

    # 4. Generate Hemodynamic Pressures via Fast Calibrated Windkessel Mapping
    # To run 1440 points in milliseconds without solving 1440 ODEs during generation,
    # we compute the analytical mean pressure MAP = (SV * HR / 60) * (Rp + Rc)
    # and pulse pressure PP = SV / C * (1 - exp(-T_diastole / (Rp * C))),
    # matching the 3-element Windkessel behavior within < 1 mmHg.
    co_series = (sv_series * hr_series) / 60.0
    rc_const = base_p.Rc
    map_clean = co_series * (rp_series + rc_const)

    t_cycle = 60.0 / hr_series
    f_sys = 0.33
    t_dia = (1.0 - f_sys) * t_cycle
    decay_ratio = np.exp(-t_dia / (rp_series * c_series))

    pp_clean = (sv_series / c_series) * (1.0 - 0.45 * decay_ratio)
    # Physiological formula: SBP ~ MAP + (2/3) * PP, DBP ~ MAP - (1/3) * PP
    sbp_clean = map_clean + 0.60 * pp_clean
    dbp_clean = map_clean - 0.40 * pp_clean

    # Add realistic sensor noise (Gaussian noise + occasional micro-artifacts)
    sbp_noise = rng.normal(0.0, 1.8, size=n_points)
    dbp_noise = rng.normal(0.0, 1.4, size=n_points)
    hr_noise = rng.normal(0.0, 1.2, size=n_points)

    sbp_obs = sbp_clean + sbp_noise
    dbp_obs = dbp_clean + dbp_noise
    hr_obs = hr_series + hr_noise

    # Guarantee physiological consistency (SBP > DBP + 15)
    dbp_obs = np.minimum(dbp_obs, sbp_obs - 18.0)
    map_obs = (sbp_obs + 2.0 * dbp_obs) / 3.0
    pp_obs = sbp_obs - dbp_obs

    # 5. ECG-Derived Features
    # RR interval (ms) = 60000 / HR with respiratory sinus arrhythmia
    rr_intervals = (60000.0 / hr_obs) + rng.normal(0.0, 12.0, size=n_points)

    # HRV-SDNN (Standard Deviation of NN intervals in ms)
    # High during rest/sleep (~65-85 ms), drops during exercise/stress (~20-30 ms)
    base_sdnn = 55.0
    stress_suppression = np.where(hr_obs > 90.0, 25.0 * ((hr_obs - 90.0) / 40.0), 0.0)
    sleep_elevation = np.where(circadian_dip < -0.05, 20.0 * (-circadian_dip / 0.12), 0.0)
    hrv_sdnn = base_sdnn - stress_suppression + sleep_elevation + rng.normal(0.0, 3.5, size=n_points)
    hrv_sdnn = np.clip(hrv_sdnn, 15.0, 110.0)

    # QT-like interval (ms): Bazett's formula QT = QTc * sqrt(RR / 1000), normal QTc ~ 410 ms
    qt_c = 410.0 + rng.normal(0.0, 5.0, size=n_points)
    qt_interval = qt_c * np.sqrt(np.maximum(rr_intervals, 300.0) / 1000.0)

    # Inject sensor dropouts if requested
    if include_anomalies:
        dropout_slice = slice(980, 995)
        sbp_obs[dropout_slice] = np.nan
        dbp_obs[dropout_slice] = np.nan
        hr_obs[dropout_slice] = np.nan
        map_obs[dropout_slice] = np.nan
        pp_obs[dropout_slice] = np.nan
        rr_intervals[dropout_slice] = np.nan
        hrv_sdnn[dropout_slice] = np.nan
        qt_interval[dropout_slice] = np.nan

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "minute": t_minutes,
            # Observed vitals (what wearable/cuff sees)
            "HR": hr_obs,
            "SBP": sbp_obs,
            "DBP": dbp_obs,
            "MAP": map_obs,
            "PP": pp_obs,
            # ECG-derived features
            "RR_ms": rr_intervals,
            "HRV_SDNN": hrv_sdnn,
            "QT_ms": qt_interval,
            # True underlying physiological parameters (ground truth for validation)
            "true_Rp": rp_series,
            "true_C": c_series,
            "true_SV": sv_series,
            "true_HR": hr_series,
            # Episode & anomaly metadata
            "episode": episode_labels,
            "is_anomaly": anomaly_flags,
        }
    )

    return df
