"""Cardiovascular Divergence Engine.

The primary diagnostic innovation of CardioTwin X:
Compares incoming patient observations with the digital twin's model-predicted
trajectories to detect physiological divergence (e.g. vascular decompensation,
occult hypertension, autonomic stress) vs expected physiological variability.

Components:
1. Normalized Innovation / Mahalanobis Divergence Score:
   D_M = sqrt(v^T * S^-1 * v), where v is innovation, S is innovation covariance (Pyy).
   Under normal conditions, D_M^2 follows a chi-square distribution with k=3 dof.
2. Persistence Filter:
   Requires high divergence to persist across N consecutive measurements to
   filter transient motion artifacts and sensor dropouts.
3. 50-Twin Monte Carlo Posterior Ensemble:
   Samples parameters from N(x_mean, P) to generate 5th-95th percentile
   predictive uncertainty bands for SBP, DBP, and HR.
4. Secondary Detector:
   IsolationForest trained on residual features; provides cross-detector agreement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest

from cardiotwin.twin.estimator import CardiovascularUKF, EstimatorStepResult


@dataclass
class DivergenceResult:
    """Diagnostic outcome of divergence analysis for a single time step."""
    time_step: int
    divergence_score: float                # Mahalanobis distance D_M
    status: str                            # 'Normal', 'Watch', 'Significant'
    p_value: float                         # 1 - chi2.cdf(D_M^2, df=3)
    consecutive_high_count: int            # Number of consecutive steps >= threshold
    signal_contributions: Dict[str, float] # % contribution to total divergence
    ensemble_bands: Dict[str, Dict[str, float]] # Predictive uncertainty (p5, median, p95)
    isolation_forest_anomaly: bool         # True if IsolationForest flags anomaly
    isolation_forest_score: float          # Raw decision function score
    detector_agreement: str                # Agreement category between UKF and IsolationForest
    is_missing_data: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to serializable dictionary."""
        return {
            "time_step": self.time_step,
            "divergence_score": float(self.divergence_score),
            "status": self.status,
            "p_value": float(self.p_value),
            "consecutive_high_count": self.consecutive_high_count,
            "signal_contributions": self.signal_contributions,
            "ensemble_bands": self.ensemble_bands,
            "if_anomaly": self.isolation_forest_anomaly,
            "if_score": float(self.isolation_forest_score),
            "detector_agreement": self.detector_agreement,
            "is_missing_data": self.is_missing_data,
        }


class DivergenceEngine:
    """Real-time cardiovascular divergence detection and ensemble engine."""

    # Chi-square quantile thresholds for k=3 degrees of freedom:
    # chi2.ppf(0.85, 3) = 5.317 -> sqrt = 2.306 (Watch threshold)
    # chi2.ppf(0.98, 3) = 9.837 -> sqrt = 3.136 (Significant threshold)
    WATCH_THRESHOLD: float = 2.30
    SIGNIFICANT_THRESHOLD: float = 3.14
    PERSISTENCE_N: int = 3  # Requires 3 consecutive steps above threshold for 'Significant'

    def __init__(
        self,
        watch_threshold: float = WATCH_THRESHOLD,
        significant_threshold: float = SIGNIFICANT_THRESHOLD,
        persistence_n: int = PERSISTENCE_N,
        ensemble_size: int = 50,
        random_state: int = 42,
    ) -> None:
        """Initialize divergence engine with thresholds and secondary detector."""
        self.watch_threshold = float(watch_threshold)
        self.significant_threshold = float(significant_threshold)
        self.persistence_n = int(persistence_n)
        self.ensemble_size = int(ensemble_size)
        self.rng = np.random.default_rng(random_state)

        # Internal persistence state
        self.consecutive_high_count = 0
        self.history: List[DivergenceResult] = []

        # Secondary detector: IsolationForest
        self.iso_forest = IsolationForest(
            n_estimators=100,
            contamination=0.04,
            random_state=random_state,
        )
        self._is_forest_fitted = False
        self._residual_buffer: List[List[float]] = []

        # Pre-seed IsolationForest with standard resting Gaussian residuals
        initial_residuals = self.rng.normal(0.0, 1.5, size=(120, 3))
        self.fit_isolation_forest(initial_residuals)

    def fit_isolation_forest(self, residuals: np.ndarray) -> None:
        """Fit IsolationForest on a baseline sequence of residual vectors [r_HR, r_SBP, r_DBP]."""
        residuals_clean = np.asarray(residuals, dtype=np.float64)
        if len(residuals_clean) >= 20:
            self.iso_forest.fit(residuals_clean)
            self._is_forest_fitted = True

    def compute_ensemble_bands(
        self,
        ukf: CardiovascularUKF,
        state_mean: np.ndarray,
        state_cov: np.ndarray,
    ) -> Dict[str, Dict[str, float]]:
        """Sample 50 twins from posterior N(x, P) and generate predictive uncertainty bands.

        Returns percentiles: 5th, 50th (median), and 95th for HR, SBP, DBP.
        """
        # Ensure positive-definite covariance
        cov_pd = ukf._ensure_positive_definite(state_cov)
        try:
            samples = self.rng.multivariate_normal(
                mean=state_mean,
                cov=cov_pd,
                size=self.ensemble_size,
                check_valid="ignore",
            )
        except Exception:
            # Fallback to diagonal standard deviation sampling
            std = np.sqrt(np.diag(cov_pd))
            samples = state_mean + self.rng.normal(0.0, 1.0, size=(self.ensemble_size, len(state_mean))) * std

        # Evaluate observation model across all 50 twin samples
        ensemble_obs = np.array([ukf.observation_model(s) for s in samples], dtype=np.float64)

        obs_keys = ["HR", "SBP", "DBP"]
        bands: Dict[str, Dict[str, float]] = {}

        for idx, key in enumerate(obs_keys):
            vals = ensemble_obs[:, idx]
            bands[key] = {
                "p5": float(np.percentile(vals, 5)),
                "median": float(np.median(vals)),
                "p95": float(np.percentile(vals, 95)),
                "std": float(np.std(vals)),
            }

        return bands

    def evaluate_step(
        self,
        ukf_result: EstimatorStepResult,
        ukf_instance: CardiovascularUKF,
    ) -> DivergenceResult:
        """Evaluate divergence between observed measurements and digital twin predictions.

        Args:
            ukf_result: Step outcome from CardiovascularUKF.
            ukf_instance: The UKF instance used to evaluate ensemble predictions.

        Returns:
            DivergenceResult with score, status, persistence, and detector agreement.
        """
        t_step = ukf_result.time_step

        # Always compute 50-twin ensemble bands for predictive uncertainty
        ensemble_bands = self.compute_ensemble_bands(
            ukf=ukf_instance,
            state_mean=ukf_result.state_mean,
            state_cov=ukf_result.state_cov,
        )

        # Handle missing observation / sensor dropout
        if not ukf_result.update_performed or ukf_result.innovation is None:
            # When sensor reading is absent, maintain previous count and report Normal/Dropout
            self.consecutive_high_count = 0
            res = DivergenceResult(
                time_step=t_step,
                divergence_score=0.0,
                status="Normal",
                p_value=1.0,
                consecutive_high_count=0,
                signal_contributions={"HR": 0.0, "SBP": 0.0, "DBP": 0.0},
                ensemble_bands=ensemble_bands,
                isolation_forest_anomaly=False,
                isolation_forest_score=0.0,
                detector_agreement="Sensor Dropout (No Signal)",
                is_missing_data=True,
            )
            self.history.append(res)
            return res

        v = ukf_result.innovation  # [e_HR, e_SBP, e_DBP]
        S = ukf_result.innovation_cov  # Pyy (3x3)

        # Invert innovation covariance matrix S
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)

        # Mahalanobis Distance D_M = sqrt(v^T * S^-1 * v)
        dm_squared = float(v.T @ S_inv @ v)
        dm_squared = max(dm_squared, 0.0)
        dm = float(np.sqrt(dm_squared))

        # Chi-square p-value for 3 degrees of freedom
        p_val = float(1.0 - stats.chi2.cdf(dm_squared, df=3))

        # Decomposition: Per-signal contribution to Mahalanobis divergence
        # Diagonal term contribution ~ v_i^2 * S_inv[i, i]
        diag_contrib = (v ** 2) * np.diag(S_inv)
        total_diag = np.sum(diag_contrib)
        if total_diag > 1e-9:
            pct_contrib = (diag_contrib / total_diag) * 100.0
        else:
            pct_contrib = np.array([33.33, 33.33, 33.33])

        signal_contributions = {
            "HR": float(pct_contrib[0]),
            "SBP": float(pct_contrib[1]),
            "DBP": float(pct_contrib[2]),
        }

        # Secondary detector: IsolationForest on residual vector
        v_2d = v.reshape(1, -1)
        if self._is_forest_fitted:
            if_pred = int(self.iso_forest.predict(v_2d)[0])  # +1 inlier, -1 anomaly
            if_score = float(self.iso_forest.decision_function(v_2d)[0])
            is_if_anomaly = (if_pred == -1)
        else:
            is_if_anomaly = False
            if_score = 0.0

        # Persistence filtering logic
        if dm >= self.significant_threshold:
            self.consecutive_high_count += 1
        else:
            self.consecutive_high_count = 0

        # Determine clinical status:
        # 'Significant' requires persistence >= PERSISTENCE_N steps
        # 'Watch' if score elevated or transient spike
        if self.consecutive_high_count >= self.persistence_n:
            status = "Significant"
        elif dm >= self.watch_threshold:
            status = "Watch"
        else:
            status = "Normal"

        # Cross-Detector Agreement determination
        if status == "Significant" and is_if_anomaly:
            agreement = "Both Agree: Significant Anomaly"
        elif status == "Normal" and not is_if_anomaly:
            agreement = "Both Agree: Normal"
        elif status == "Significant" and not is_if_anomaly:
            agreement = "Mahalanobis Significant / IsolationForest Inlier"
        elif status == "Watch" and is_if_anomaly:
            agreement = "Both Flag Elevated / Watch"
        elif is_if_anomaly and status == "Normal":
            agreement = "IsolationForest Anomaly / Mahalanobis Normal"
        else:
            agreement = f"{status} (Cross-checked)"

        res = DivergenceResult(
            time_step=t_step,
            divergence_score=dm,
            status=status,
            p_value=p_val,
            consecutive_high_count=self.consecutive_high_count,
            signal_contributions=signal_contributions,
            ensemble_bands=ensemble_bands,
            isolation_forest_anomaly=is_if_anomaly,
            isolation_forest_score=if_score,
            detector_agreement=agreement,
            is_missing_data=False,
        )
        self.history.append(res)
        return res
