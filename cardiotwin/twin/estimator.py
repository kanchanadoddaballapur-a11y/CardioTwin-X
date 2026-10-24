"""Adaptive Hemodynamic State Estimator using Unscented Kalman Filtering (UKF).

Estimates unobserved internal cardiovascular parameters:
- State vector x: [Rp, C, SV, HR]
  - Rp: Peripheral resistance (mmHg * s / mL)
  - C: Systemic arterial compliance (mL / mmHg)
  - SV: Stroke volume (mL)
  - HR: Heart rate (bpm)
- Observation vector y: [HR, SBP, DBP]
  - HR: Observed heart rate (bpm)
  - SBP: Observed systolic blood pressure (mmHg)
  - DBP: Observed diastolic blood pressure (mmHg)

The UKF uses a non-linear observation model h(x) derived from the 3-element
Windkessel model. If an observation contains missing values (e.g. sensor dropout),
the measurement update step is skipped while retaining prior state uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from cardiotwin.twin.model import WindkesselParams, simulate_beat


@dataclass
class EstimatorStepResult:
    """Output bundle produced at each time step of the UKF estimator."""
    time_step: int
    state_mean: np.ndarray             # [Rp, C, SV, HR]
    state_std: np.ndarray              # 1-sigma uncertainty for each state
    state_cov: np.ndarray              # Full 4x4 state covariance matrix
    predicted_obs: np.ndarray          # Predicted [HR, SBP, DBP]
    predicted_obs_std: np.ndarray      # Predicted observation uncertainty
    innovation_cov: np.ndarray         # Pyy (3x3 innovation covariance matrix)
    observed: Optional[np.ndarray]      # Actual measurement [HR, SBP, DBP] (or None if missing)
    innovation: Optional[np.ndarray]    # y - y_pred (or None if missing)
    update_performed: bool             # True if measurement update was applied

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to user-friendly dictionary."""
        return {
            "time_step": self.time_step,
            "Rp": float(self.state_mean[0]),
            "Rp_std": float(self.state_std[0]),
            "C": float(self.state_mean[1]),
            "C_std": float(self.state_std[1]),
            "SV": float(self.state_mean[2]),
            "SV_std": float(self.state_std[2]),
            "HR": float(self.state_mean[3]),
            "HR_std": float(self.state_std[3]),
            "pred_HR": float(self.predicted_obs[0]),
            "pred_HR_std": float(self.predicted_obs_std[0]),
            "pred_SBP": float(self.predicted_obs[1]),
            "pred_SBP_std": float(self.predicted_obs_std[1]),
            "pred_DBP": float(self.predicted_obs[2]),
            "pred_DBP_std": float(self.predicted_obs_std[2]),
            "update_performed": self.update_performed,
        }


class CardiovascularUKF:
    """Unscented Kalman Filter for physiological twin parameter estimation."""

    STATE_NAMES = ["Rp", "C", "SV", "HR"]
    OBS_NAMES = ["HR", "SBP", "DBP"]

    # Physiological hard limits
    STATE_BOUNDS = np.array([
        [0.40, 3.00],   # Rp: mmHg * s / mL
        [0.30, 3.00],   # C:  mL / mmHg
        [30.0, 160.0],  # SV: mL
        [35.0, 200.0],  # HR: bpm
    ], dtype=np.float64)

    def __init__(
        self,
        initial_state: Optional[Union[np.ndarray, list]] = None,
        initial_cov: Optional[np.ndarray] = None,
        process_noise: Optional[np.ndarray] = None,
        measurement_noise: Optional[np.ndarray] = None,
        alpha: float = 0.5,
        beta: float = 2.0,
        kappa: float = 0.0,
    ) -> None:
        """Initialize UKF with state prior and noise statistics.

        Args:
            initial_state: Prior state vector [Rp, C, SV, HR].
            initial_cov: 4x4 initial error covariance matrix P0.
            process_noise: 4x4 process noise covariance Q.
            measurement_noise: 3x3 measurement noise covariance R.
            alpha: Primary spread parameter for sigma points.
            beta: Distribution parameter (2.0 is optimal for Gaussian).
            kappa: Secondary scaling parameter.
        """
        # Default healthy normotensive initial state
        if initial_state is None:
            self.x = np.array([1.253, 1.417, 62.8, 72.0], dtype=np.float64)
        else:
            self.x = np.asarray(initial_state, dtype=np.float64).copy()

        self.L = len(self.x)  # State dimension (4)
        self.M = 3            # Observation dimension (3)

        # Default initial covariance P0
        if initial_cov is None:
            self.P = np.diag([0.15**2, 0.20**2, 8.0**2, 4.0**2]).astype(np.float64)
        else:
            self.P = np.asarray(initial_cov, dtype=np.float64).copy()

        # Default process noise Q (allows twin parameters to track physiological shifts)
        if process_noise is None:
            self.Q = np.diag([1.5e-4, 1.5e-4, 1.2e-2, 1.0e-2]).astype(np.float64)
        else:
            self.Q = np.asarray(process_noise, dtype=np.float64).copy()

        # Default measurement noise R
        if measurement_noise is None:
            self.R = np.diag([1.5**2, 2.0**2, 1.5**2]).astype(np.float64)
        else:
            self.R = np.asarray(measurement_noise, dtype=np.float64).copy()

        # UKF weights calculation
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.kappa = float(kappa)
        self.lam = self.alpha ** 2 * (self.L + self.kappa) - self.L

        num_sigmas = 2 * self.L + 1
        self.Wm = np.full(num_sigmas, 1.0 / (2.0 * (self.L + self.lam)), dtype=np.float64)
        self.Wc = np.full(num_sigmas, 1.0 / (2.0 * (self.L + self.lam)), dtype=np.float64)

        self.Wm[0] = self.lam / (self.L + self.lam)
        self.Wc[0] = self.lam / (self.L + self.lam) + (1.0 - self.alpha ** 2 + self.beta)

        # Step counter and history
        self.time_step = 0
        self.history: List[EstimatorStepResult] = []

    def _ensure_positive_definite(self, mat: np.ndarray, min_eig: float = 1e-7) -> np.ndarray:
        """Symmetrize and clip negative eigenvalues to ensure positive definiteness."""
        sym = 0.5 * (mat + mat.T)
        eigvals, eigvecs = np.linalg.eigh(sym)
        clipped_eigvals = np.maximum(eigvals, min_eig)
        return (eigvecs @ np.diag(clipped_eigvals) @ eigvecs.T).astype(np.float64)

    def _generate_sigma_points(self, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
        """Generate 2L + 1 sigma points using Cholesky decomposition."""
        scale = self.L + self.lam
        cov_pd = self._ensure_positive_definite(cov)
        try:
            L_mat = np.linalg.cholesky(scale * cov_pd)
        except np.linalg.LinAlgError:
            # Fallback using eigendecomposition square root if Cholesky fails
            eigvals, eigvecs = np.linalg.eigh(cov_pd)
            L_mat = eigvecs @ np.diag(np.sqrt(np.maximum(scale * eigvals, 1e-7)))

        sigmas = [mean]
        for i in range(self.L):
            sigmas.append(mean + L_mat[:, i])
        for i in range(self.L):
            sigmas.append(mean - L_mat[:, i])
        return np.array(sigmas, dtype=np.float64)

    def observation_model(self, state_vec: np.ndarray) -> np.ndarray:
        """Nonlinear observation function mapping state [Rp, C, SV, HR] -> [HR, SBP, DBP].

        Uses analytical periodic Windkessel mapping with calibrated hemodynamic formulas
        for maximum computational speed and numerical stability.
        """
        # Enforce bounds
        rp = float(np.clip(state_vec[0], self.STATE_BOUNDS[0, 0], self.STATE_BOUNDS[0, 1]))
        c = float(np.clip(state_vec[1], self.STATE_BOUNDS[1, 0], self.STATE_BOUNDS[1, 1]))
        sv = float(np.clip(state_vec[2], self.STATE_BOUNDS[2, 0], self.STATE_BOUNDS[2, 1]))
        hr = float(np.clip(state_vec[3], self.STATE_BOUNDS[3, 0], self.STATE_BOUNDS[3, 1]))

        # Characteristic resistance constant
        rc = 0.050
        co = (sv * hr) / 60.0
        map_val = co * (rp + rc)

        t_cycle = 60.0 / hr
        t_dia = 0.67 * t_cycle
        decay = np.exp(-t_dia / (rp * c))
        pp_val = (sv / c) * (1.0 - 0.45 * decay)

        sbp = map_val + 0.60 * pp_val
        dbp = map_val - 0.40 * pp_val

        return np.array([hr, sbp, dbp], dtype=np.float64)

    def step(self, observation: Optional[Union[np.ndarray, list]]) -> EstimatorStepResult:
        """Process one time step with observation [HR, SBP, DBP].

        If observation is None or contains NaNs, skips the measurement update.

        Args:
            observation: [HR, SBP, DBP] reading or None.

        Returns:
            EstimatorStepResult containing posterior estimates, uncertainties, and innovation.
        """
        self.time_step += 1

        # 1. Time Update (Predict State and Prior Covariance)
        x_pred = self.x.copy()
        P_pred = self._ensure_positive_definite(self.P + self.Q)

        # 2. Sigma Points Generation from Prior
        sigmas = self._generate_sigma_points(x_pred, P_pred)

        # 3. Transform Sigma Points through Observation Model
        gamma = np.array([self.observation_model(s) for s in sigmas], dtype=np.float64)

        # Predicted observation mean y_hat
        y_pred = np.sum(self.Wm[:, None] * gamma, axis=0)

        # Predicted observation covariance Pyy
        y_diff = gamma - y_pred
        Pyy = np.sum(self.Wc[:, None, None] * (y_diff[:, :, None] * y_diff[:, None, :]), axis=0) + self.R
        Pyy = self._ensure_positive_definite(Pyy)

        # Cross-covariance Pxy
        x_diff = sigmas - x_pred
        Pxy = np.sum(self.Wc[:, None, None] * (x_diff[:, :, None] * y_diff[:, None, :]), axis=0)

        pred_obs_std = np.sqrt(np.diag(Pyy))

        # Check for missing observation / NaNs
        is_missing = False
        if observation is None:
            is_missing = True
        else:
            obs_arr = np.asarray(observation, dtype=np.float64)
            if np.any(np.isnan(obs_arr)) or len(obs_arr) < 3:
                is_missing = True

        if is_missing:
            # Skip measurement update; keep prediction
            self.x = x_pred
            self.P = P_pred
            step_result = EstimatorStepResult(
                time_step=self.time_step,
                state_mean=self.x.copy(),
                state_std=np.sqrt(np.diag(self.P)),
                state_cov=self.P.copy(),
                predicted_obs=y_pred,
                predicted_obs_std=pred_obs_std,
                innovation_cov=Pyy,
                observed=None,
                innovation=None,
                update_performed=False,
            )
            self.history.append(step_result)
            return step_result

        # 4. Measurement Update
        obs_clean = np.asarray(observation, dtype=np.float64)[:3]
        innovation = obs_clean - y_pred

        # Kalman Gain K = Pxy * inv(Pyy)
        try:
            K = Pxy @ np.linalg.inv(Pyy)
        except np.linalg.LinAlgError:
            K = Pxy @ np.linalg.pinv(Pyy)

        # Posterior state update
        x_post = x_pred + K @ innovation

        # Posterior covariance update
        P_post = P_pred - K @ Pyy @ K.T
        P_post = self._ensure_positive_definite(P_post)

        # Enforce physiological parameter limits on state vector
        for dim in range(self.L):
            x_post[dim] = np.clip(x_post[dim], self.STATE_BOUNDS[dim, 0], self.STATE_BOUNDS[dim, 1])

        self.x = x_post
        self.P = P_post

        step_result = EstimatorStepResult(
            time_step=self.time_step,
            state_mean=self.x.copy(),
            state_std=np.sqrt(np.diag(self.P)),
            state_cov=self.P.copy(),
            predicted_obs=y_pred,
            predicted_obs_std=pred_obs_std,
            innovation_cov=Pyy,
            observed=obs_clean,
            innovation=innovation,
            update_performed=True,
        )
        self.history.append(step_result)
        return step_result

    def get_history_dataframe(self) -> Any:
        """Convert estimator step history into a pandas DataFrame."""
        import pandas as pd
        records = [res.to_dict() for res in self.history]
        return pd.DataFrame(records)
