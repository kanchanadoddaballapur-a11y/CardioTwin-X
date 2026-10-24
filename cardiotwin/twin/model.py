"""3-Element Windkessel Model with Baroreflex Coupling.

Implements the classic 3-element Windkessel model (WK3) consisting of:
- Rp: Peripheral vascular resistance (mmHg * s / mL)
- C: Total systemic arterial compliance (mL / mmHg)
- Rc: Aortic characteristic resistance / impedance (mmHg * s / mL)
- SV: Left ventricular stroke volume (mL)
- HR: Heart rate (beats per minute)

The ejection flow waveform is modeled as a half-sine wave during systole.
Differential equations are solved using scipy.integrate.solve_ivp.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union
import numpy as np
from scipy.integrate import solve_ivp, trapezoid


@dataclass
class WindkesselParams:
    """Cardiovascular parameters for the 3-element Windkessel model.

    Default values are calibrated to yield canonical resting normotensive
    hemodynamics of approximately 120 / 80 mmHg (MAP ~ 93 mmHg) at 72 bpm.
    """
    Rp: float = 1.253      # Peripheral resistance (mmHg * s / mL)
    C: float = 1.417       # Arterial compliance (mL / mmHg)
    Rc: float = 0.050      # Aortic characteristic resistance (mmHg * s / mL)
    SV: float = 62.8       # Stroke volume (mL)
    HR: float = 72.0       # Heart rate (bpm)
    f_sys: float = 0.33    # Systolic ejection fraction of cardiac cycle

    # Physiological parameter bounds [lower, upper]
    BOUNDS: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "Rp": (0.4, 3.0),
            "C": (0.3, 3.0),
            "Rc": (0.01, 0.15),
            "SV": (30.0, 160.0),
            "HR": (35.0, 200.0),
        }
    )

    def to_vector(self) -> np.ndarray:
        """Return 5-element parameter vector [Rp, C, Rc, SV, HR]."""
        return np.array([self.Rp, self.C, self.Rc, self.SV, self.HR], dtype=np.float64)

    @classmethod
    def from_vector(cls, vec: Union[np.ndarray, list], f_sys: float = 0.33) -> WindkesselParams:
        """Create parameters instance from vector [Rp, C, Rc, SV, HR]."""
        v = np.asarray(vec, dtype=np.float64)
        if len(v) == 4:
            # If Rc omitted, use default 0.05
            return cls(Rp=float(v[0]), C=float(v[1]), Rc=0.05, SV=float(v[2]), HR=float(v[3]), f_sys=f_sys)
        return cls(
            Rp=float(v[0]),
            C=float(v[1]),
            Rc=float(v[2]),
            SV=float(v[3]),
            HR=float(v[4]),
            f_sys=f_sys,
        )

    def clipped(self) -> WindkesselParams:
        """Return a copy with values clipped within physiological bounds."""
        return WindkesselParams(
            Rp=float(np.clip(self.Rp, *self.BOUNDS["Rp"])),
            C=float(np.clip(self.C, *self.BOUNDS["C"])),
            Rc=float(np.clip(self.Rc, *self.BOUNDS["Rc"])),
            SV=float(np.clip(self.SV, *self.BOUNDS["SV"])),
            HR=float(np.clip(self.HR, *self.BOUNDS["HR"])),
            f_sys=self.f_sys,
        )


@dataclass
class WindkesselOutput:
    """Output metrics and full waveform from a simulated cardiac cycle."""
    sbp: float             # Systolic Blood Pressure (mmHg)
    dbp: float             # Diastolic Blood Pressure (mmHg)
    map: float             # Mean Arterial Pressure (mmHg)
    pp: float              # Pulse Pressure (mmHg)
    hr: float              # Heart Rate used in simulation (bpm)
    stroke_volume: float   # Stroke Volume (mL)
    time: np.ndarray       # Beat time vector (seconds, 0 -> T)
    pressure: np.ndarray   # Full aortic pressure waveform (mmHg)
    flow: np.ndarray       # Inflow waveform (mL / s)
    p_c: np.ndarray        # Compliance chamber pressure (mmHg)


def apply_baroreflex(
    mean_arterial_pressure: float,
    baseline_hr: float = 72.0,
    setpoint_map: float = 93.3,
    gain: float = 0.45,
    max_delta_hr: float = 16.0,
) -> float:
    """Compute baroreflex-mediated heart rate response.

    The arterial baroreflex acts as a negative-feedback mechanism:
    - If MAP rises above setpoint -> HR drops.
    - If MAP drops below setpoint -> HR increases.

    Args:
        mean_arterial_pressure: Current MAP in mmHg.
        baseline_hr: Baseline resting heart rate in bpm.
        setpoint_map: Target equilibrium MAP in mmHg (typically ~93.3 mmHg).
        gain: Sensitivity in bpm / mmHg (typical: 0.3 - 0.6).
        max_delta_hr: Max allowable heart rate deviation in bpm.

    Returns:
        Adjusted heart rate (bpm), clamped between 40 and 190.
    """
    error = mean_arterial_pressure - setpoint_map
    delta_hr = -gain * error
    delta_hr_clamped = float(np.clip(delta_hr, -max_delta_hr, max_delta_hr))
    adjusted_hr = baseline_hr + delta_hr_clamped
    return float(np.clip(adjusted_hr, 40.0, 190.0))


def simulate_beat(
    params: WindkesselParams,
    n_points: int = 200,
    enable_baroreflex: bool = False,
    baroreflex_setpoint: float = 93.3,
    baroreflex_gain: float = 0.45,
) -> WindkesselOutput:
    """Simulate a single periodic cardiac cycle of the 3-element Windkessel model.

    The model is governed by:
        d(P_c)/dt = (Q_in(t) - P_c / Rp) / C
        P_aortic(t) = P_c(t) + Rc * Q_in(t)

    Inflow Q_in(t) is a half-sine during systole:
        Q_in(t) = Q_peak * sin(pi * t / T_sys) for 0 <= t < T_sys
        Q_in(t) = 0                           for T_sys <= t < T

    The periodic boundary condition P_c(0) = P_c(T) is calculated analytically
    to seed scipy.integrate.solve_ivp for precision and stability.

    Args:
        params: WindkesselParams containing Rp, C, Rc, SV, HR, f_sys.
        n_points: Number of evaluation time steps across the cycle.
        enable_baroreflex: If True, adjust HR via baroreflex before simulation.
        baroreflex_setpoint: MAP setpoint for baroreflex.
        baroreflex_gain: Gain in bpm / mmHg.

    Returns:
        WindkesselOutput with SBP, DBP, MAP, PP, and waveform arrays.
    """
    p = params.clipped()
    hr = p.HR

    # Optional initial 1-step baroreflex adjustment if requested
    if enable_baroreflex:
        # Pre-estimate nominal MAP ~ CO * (Rp + Rc)
        nominal_co = (p.SV * hr) / 60.0
        est_map = nominal_co * (p.Rp + p.Rc)
        hr = apply_baroreflex(est_map, baseline_hr=p.HR, setpoint_map=baroreflex_setpoint, gain=baroreflex_gain)

    # Timing parameters
    T = 60.0 / hr
    f_sys = float(np.clip(p.f_sys, 0.20, 0.50))
    T_sys = f_sys * T
    omega = np.pi / T_sys
    Q_peak = (np.pi * p.SV) / (2.0 * T_sys)
    tau = p.Rp * p.C

    # Compute analytical periodic initial pressure P_c(0) where P_c(0) == P_c(T)
    # Systolic integral component:
    denom = (1.0 / tau) ** 2 + omega ** 2
    delta_P_sys = (Q_peak / p.C) * (omega / denom) * (1.0 + np.exp(-T_sys / tau))
    exp_diastole = np.exp(-(T - T_sys) / tau)
    exp_cycle = np.exp(-T / tau)

    if abs(1.0 - exp_cycle) > 1e-6:
        P0_periodic = (delta_P_sys * exp_diastole) / (1.0 - exp_cycle)
    else:
        P0_periodic = 80.0

    def flow_rate(t_val: float) -> float:
        """Inflow ejection waveform."""
        t_mod = t_val % T
        if t_mod < T_sys:
            return float(Q_peak * np.sin(omega * t_mod))
        return 0.0

    def ode(t: float, y: list[float]) -> list[float]:
        """WK3 differential equation: dPc/dt = (Q_in - Pc/Rp) / C."""
        p_c = y[0]
        q = flow_rate(t)
        dp_c_dt = (q - (p_c / p.Rp)) / p.C
        return [dp_c_dt]

    t_eval = np.linspace(0.0, T, n_points)
    sol = solve_ivp(
        ode,
        (0.0, T),
        [P0_periodic],
        t_eval=t_eval,
        method="RK45",
        rtol=1e-6,
        atol=1e-6,
    )

    p_c_arr = sol.y[0]
    flow_arr = np.array([flow_rate(t) for t in t_eval], dtype=np.float64)
    p_ao_arr = p_c_arr + p.Rc * flow_arr

    sbp = float(np.max(p_ao_arr))
    dbp = float(np.min(p_ao_arr))
    pp = sbp - dbp
    map_val = float(trapezoid(p_ao_arr, t_eval) / T)

    # Second pass for baroreflex if enabled to converge HR with exact simulated MAP
    if enable_baroreflex:
        refined_hr = apply_baroreflex(map_val, baseline_hr=p.HR, setpoint_map=baroreflex_setpoint, gain=baroreflex_gain)
        if abs(refined_hr - hr) > 0.5:
            # Re-simulate with final baroreflex adjusted HR
            p_refined = WindkesselParams(
                Rp=p.Rp, C=p.C, Rc=p.Rc, SV=p.SV, HR=refined_hr, f_sys=p.f_sys
            )
            return simulate_beat(p_refined, n_points=n_points, enable_baroreflex=False)

    return WindkesselOutput(
        sbp=sbp,
        dbp=dbp,
        map=map_val,
        pp=pp,
        hr=hr,
        stroke_volume=p.SV,
        time=t_eval,
        pressure=p_ao_arr,
        flow=flow_arr,
        p_c=p_c_arr,
    )
