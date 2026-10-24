"""CardioTwin X - Adaptive Simulation-Driven Cardiovascular Digital Twin.

A real-time physiological digital twin prototype incorporating:
- 3-Element Windkessel physiological modeling (WK3)
- Unscented Kalman Filter state estimation
- Innovation-covariance Mahalanobis divergence detection
- 50-twin Monte Carlo uncertainty ensemble
- Dual-detector cross-validation (IsolationForest)
- Physiological explainability engine
- Counterfactual what-if simulation lab
- 3D hemodynamic heart & aortic visualization
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Ensure package importability
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
for p in [current_dir, parent_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from cardiotwin.twin.data import PATIENT_PRESETS, PatientHistory
from cardiotwin.twin.divergence import DivergenceResult
from cardiotwin.twin.estimator import EstimatorStepResult
from cardiotwin.twin.explain import ExplanationReport
from cardiotwin.twin.heart3d import generate_3d_heart_html
from cardiotwin.twin.model import WindkesselParams, simulate_beat
from cardiotwin.twin.state import TwinSessionState
from cardiotwin.twin.validation import (
    VALIDATION_NOTE,
    check_public_dataset,
    run_cohort_validation,
)
from cardiotwin.twin.whatif import (
    MEDICAL_DISCLAIMER,
    WHATIF_SCENARIOS,
    ScenarioDelta,
    run_whatif_simulation,
)


# -----------------------------------------------------------------------------
# Streamlit Page Configuration & Rose Healthcare Theme Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CardioTwin X | Digital Twin",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS for soft rose/pink healthcare theme with rounded cards
st.markdown(
    """
    <style>
    /* Theme color variables */
    :root {
        --primary-rose: #e11d48;
        --rose-light: #fff1f2;
        --rose-border: #fecdd3;
        --card-bg: #ffffff;
        --text-slate: #1e293b;
        --text-muted: #64748b;
    }

    /* Top header bar */
    .header-card {
        background: linear-gradient(135deg, #fff1f2 0%, #ffffff 100%);
        border: 1px solid #fecdd3;
        border-radius: 16px;
        padding: 16px 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 14px rgba(225, 29, 72, 0.05);
    }

    /* Metric card */
    .vital-card {
        background-color: #ffffff;
        border: 1px solid #fecdd3;
        border-radius: 14px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(225, 29, 72, 0.04);
    }
    .vital-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .vital-value {
        font-size: 2.1rem;
        font-weight: 700;
        color: #e11d48;
        line-height: 1.2;
    }
    .vital-unit {
        font-size: 0.8rem;
        color: #94a3b8;
    }
    .vital-sub {
        font-size: 0.78rem;
        color: #059669;
        margin-top: 4px;
    }

    /* Status Badges */
    .badge-normal {
        background-color: #ecfdf5;
        color: #047857;
        border: 1px solid #a7f3d0;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-watch {
        background-color: #fffbeb;
        color: #b45309;
        border: 1px solid #fde68a;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-significant {
        background-color: #fef2f2;
        color: #b91c1c;
        border: 1px solid #fecaca;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-sync {
        background-color: #f1f5f9;
        color: #475569;
        border: 1px solid #cbd5e1;
        padding: 4px 10px;
        border-radius: 16px;
        font-size: 0.78rem;
        display: inline-block;
    }

    /* Banner for explainability */
    .explanation-banner {
        background-color: #fff1f2;
        border-left: 5px solid #e11d48;
        border-radius: 8px;
        padding: 12px 18px;
        margin-top: 14px;
        margin-bottom: 14px;
        color: #1e293b;
        font-size: 0.95rem;
    }

    /* Persistent Footer */
    .footer-text {
        text-align: center;
        color: #94a3b8;
        font-size: 0.82rem;
        padding: 24px 0 10px 0;
        border-top: 1px solid #f1f5f9;
        margin-top: 40px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Session Initialization
# -----------------------------------------------------------------------------
is_testing = (
    os.environ.get("STREAMLIT_TESTING") == "1"
    or os.environ.get("PYTEST_CURRENT_TEST") is not None
)

if "twin_state" not in st.session_state:
    st.session_state.twin_state = TwinSessionState(preset_key="healthy", seed=42)
if "selected_preset" not in st.session_state:
    st.session_state.selected_preset = "healthy"
if "auto_stream" not in st.session_state:
    st.session_state.auto_stream = False if is_testing else True


twin_mgr: TwinSessionState = st.session_state.twin_state
latest_step = twin_mgr.get_latest_step()

# Ensure we have active step data
if latest_step is None:
    twin_mgr.step_forward()
    latest_step = twin_mgr.get_latest_step()


# -----------------------------------------------------------------------------
# Top Header Bar
# -----------------------------------------------------------------------------
header_col1, header_col2, header_col3 = st.columns([2.6, 1.8, 1.6])

with header_col1:
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; gap: 12px;">
            <span style="font-size: 2.2rem;">🫀</span>
            <div>
                <h2 style="margin: 0; padding: 0; color: #e11d48; font-weight: 800; font-size: 1.6rem;">
                    CardioTwin X
                </h2>
                <div style="font-size: 0.86rem; color: #475569;">
                    <b>{twin_mgr.patient_history.name}</b>, {twin_mgr.patient_history.age} · 
                    <span style="color: #64748b;">Sample patient (synthetic data)</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with header_col2:
    sync_time_str = latest_step.timestamp.strftime("%H:%M:%S") if latest_step else "08:00:00"
    st.markdown(
        f"""
        <div style="padding-top: 10px;">
            <span class="badge-sync">🟢 Last synced: {sync_time_str} · Simulated stream</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with header_col3:
    col_t1, col_t2 = st.columns([1.3, 1.0])
    with col_t1:
        is_live_toggle = st.toggle(
            "Live",
            value=st.session_state.auto_stream,
            key="live_paused_toggle",
            help="Toggle between Live streaming and Paused inspection",
        )
        st.session_state.auto_stream = is_live_toggle
        if is_live_toggle:
            st.markdown('<span style="color: #059669; font-size: 0.82rem; font-weight: 700;">🟢 Live Mode</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span style="color: #64748b; font-size: 0.82rem; font-weight: 700;">⏸️ Paused</span>', unsafe_allow_html=True)
    with col_t2:
        if st.button("▶ Step (+1m)", help="Advance simulation 1 minute manually"):
            twin_mgr.step_forward()
            st.rerun()


# -----------------------------------------------------------------------------
# One-Time Baseline Setup Step (Skippable with Defaults)
# -----------------------------------------------------------------------------
if "baseline_setup_complete" not in st.session_state:
    st.session_state.baseline_setup_complete = False

if not st.session_state.baseline_setup_complete:
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #fff1f2 0%, #ffffff 100%); border: 1.5px solid #fecdd3; border-radius: 16px; padding: 20px 24px; margin-bottom: 22px; box-shadow: 0 4px 14px rgba(225, 29, 72, 0.05);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <div>
                    <h3 style="margin: 0; color: #e11d48; font-size: 1.25rem; font-weight: 700;">
                        📋 One-Time Baseline Setup
                    </h3>
                    <div style="font-size: 0.88rem; color: #475569; margin-top: 2px;">
                        Calibrate your digital twin with baseline cardiovascular and lifestyle metrics, or skip to use default values.
                    </div>
                </div>
                <span class="badge-watch" style="font-size: 0.8rem;">Skippable · Defaults Available</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container():
        b_col1, b_col2, b_col3 = st.columns([1.2, 1.2, 1.4])
        with b_col1:
            base_age = st.number_input("Patient Age", min_value=18, max_value=105, value=int(twin_mgr.patient_history.age), step=1)
            base_notes = st.text_input("Medical History / Clinical Notes", value=str(twin_mgr.patient_history.notes))
        with b_col2:
            base_smoker = st.checkbox("Smoker Flag", value=bool(twin_mgr.patient_history.smoker))
            base_alcohol = st.checkbox("Alcohol Consumption", value=bool(twin_mgr.patient_history.alcohol))
            base_activity = st.selectbox("Lifestyle Activity Level", ["Active", "Moderate", "Sedentary"], index=0)
        with b_col3:
            st.markdown("<b>Optional Baseline BP (mmHg):</b>", unsafe_allow_html=True)
            bp_sub1, bp_sub2 = st.columns(2)
            with bp_sub1:
                opt_sbp = st.number_input("Systolic BP", min_value=80.0, max_value=220.0, value=float(twin_mgr.patient_history.baseline_bp[0]), step=1.0)
            with bp_sub2:
                opt_dbp = st.number_input("Diastolic BP", min_value=50.0, max_value=140.0, value=float(twin_mgr.patient_history.baseline_bp[1]), step=1.0)

        btn_col1, btn_col2, _ = st.columns([1.8, 2.0, 2.2])
        with btn_col1:
            if st.button("💾 Complete Baseline Setup", type="primary", use_container_width=True):
                twin_mgr.patient_history.age = int(base_age)
                twin_mgr.patient_history.smoker = bool(base_smoker)
                twin_mgr.patient_history.alcohol = bool(base_alcohol)
                twin_mgr.patient_history.notes = str(base_notes)
                twin_mgr.patient_history.baseline_bp = (float(opt_sbp), float(opt_dbp))
                st.session_state.baseline_setup_complete = True
                st.success("Digital twin calibrated with customized baseline parameters!")
                st.rerun()
        with btn_col2:
            if st.button("⏩ Skip to Default Baseline", type="secondary", use_container_width=True):
                st.session_state.baseline_setup_complete = True
                st.info("Using default baseline (Sarah Jenkins, 42, normotensive 120/80 mmHg).")
                st.rerun()
    st.markdown("---")


# Small control row: Speed slider, Patient profile switcher, and Minor secondary "Add reading" action
with st.expander("⚙️ Stream Controls & Profile Settings", expanded=False):
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1.2, 1.4, 1.4])
    with ctrl_col1:
        speed_val = st.slider("Playback Speed (sec/reading)", min_value=0.5, max_value=3.0, value=1.0, step=0.5)
        twin_mgr.playback_speed = speed_val

    with ctrl_col2:
        preset_choice = st.selectbox(
            "Change Patient Profile Preset",
            options=["healthy", "hypertensive", "elderly"],
            index=["healthy", "hypertensive", "elderly"].index(st.session_state.selected_preset),
            format_func=lambda x: f"{x.capitalize()} Profile ({PATIENT_PRESETS[x]['history'].name})",
        )
        if preset_choice != st.session_state.selected_preset:
            st.session_state.selected_preset = preset_choice
            twin_mgr.reset(preset_key=preset_choice)
            st.rerun()

    with ctrl_col3:
        st.write("<b>Profile Summary:</b>", unsafe_allow_html=True)
        st.caption(
            f"Age: {twin_mgr.patient_history.age} | BMI: {twin_mgr.patient_history.bmi} | "
            f"Smoker: {'Yes' if twin_mgr.patient_history.smoker else 'No'} | "
            f"Alcohol: {'Yes' if twin_mgr.patient_history.alcohol else 'No'} | "
            f"Baseline BP: {twin_mgr.patient_history.baseline_bp[0]:.0f}/{twin_mgr.patient_history.baseline_bp[1]:.0f} mmHg"
        )
        if st.session_state.get("baseline_setup_complete", False):
            if st.button("🔄 Redo Baseline Setup", help="Re-open the baseline setup form"):
                st.session_state.baseline_setup_complete = False
                st.rerun()


    # Manual Reading Input
    st.markdown("---")
    st.markdown("<b>Add Reading Manually (Secondary Entry):</b>", unsafe_allow_html=True)
    m_col1, m_col2, m_col3, m_col4 = st.columns([1, 1, 1, 1])
    with m_col1:
        m_sbp = st.number_input("Systolic BP (mmHg)", min_value=70.0, max_value=240.0, value=120.0, step=1.0)
    with m_col2:
        m_dbp = st.number_input("Diastolic BP (mmHg)", min_value=40.0, max_value=150.0, value=80.0, step=1.0)
    with m_col3:
        m_hr = st.number_input("Heart Rate (bpm)", min_value=35.0, max_value=210.0, value=72.0, step=1.0)
    with m_col4:
        st.write("")
        st.write("")
        if st.button("📥 Submit Reading"):
            twin_mgr.step_forward(manual_obs=np.array([m_hr, m_sbp, m_dbp]))
            st.success("Manual reading assimilated into digital twin!")
            st.rerun()


# -----------------------------------------------------------------------------
# Cached Validation & Timeline Helpers
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner="Running 30-patient cohort validation benchmark...")
def get_cached_validation_summary():
    """Execute and cache empirical validation across 30 held-out synthetic patients."""
    return run_cohort_validation(patients_per_preset=10, steps_per_patient=40, seed_base=7000)


def build_patient_timeline_figure(twin_state: TwinSessionState) -> go.Figure:
    """Build interactive Plotly timeline showing readings, status changes, anomalies, what-if events, and data gaps."""
    steps = twin_state.steps
    if not steps:
        return go.Figure()

    times = [s.timestamp for s in steps]
    sbp_vals = [s.observed[1] if (s.observed is not None and not np.isnan(s.observed[1])) else None for s in steps]
    dbp_vals = [s.observed[2] if (s.observed is not None and not np.isnan(s.observed[2])) else None for s in steps]
    hr_vals = [s.observed[0] if (s.observed is not None and not np.isnan(s.observed[0])) else None for s in steps]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Hemodynamic Session Trajectory & Clinical Status Changes", "Heart Rate (bpm) & Data Gaps"),
        row_heights=[0.65, 0.35],
    )

    status_colors = []
    hover_texts = []
    rel_level, _, _ = twin_state.compute_data_reliability()

    for s in steps:
        st_val = s.divergence_result.status
        if st_val == "Significant":
            c = "#e11d48"
        elif st_val == "Watch":
            c = "#f59e0b"
        else:
            c = "#10b981"
        status_colors.append(c)

        obs_str = f"{s.observed[1]:.0f}/{s.observed[2]:.0f} mmHg" if s.observed is not None else "Data Gap (NaN)"
        hr_str = f"{s.observed[0]:.0f} bpm" if s.observed is not None else "N/A"
        ht = (
            f"<b>Time:</b> {s.timestamp.strftime('%H:%M:%S')}<br>"
            f"<b>Observed BP:</b> {obs_str}<br>"
            f"<b>Heart Rate:</b> {hr_str}<br>"
            f"<b>Clinical Status:</b> {s.divergence_result.status}<br>"
            f"<b>Divergence Score:</b> {s.divergence_result.divergence_score:.2f}<br>"
            f"<b>Reliability:</b> {rel_level}<br>"
            f"<b>Episode:</b> {s.episode}"
        )
        hover_texts.append(ht)

    # SBP and DBP traces
    fig.add_trace(
        go.Scatter(
            x=times, y=sbp_vals,
            mode="lines+markers",
            line=dict(color="#f43f5e", width=1.8),
            marker=dict(color=status_colors, size=6),
            text=hover_texts,
            hoverinfo="text",
            name="Systolic BP (mmHg)",
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=times, y=dbp_vals,
            mode="lines+markers",
            line=dict(color="#3b82f6", width=1.5),
            marker=dict(color=status_colors, size=5),
            text=hover_texts,
            hoverinfo="text",
            name="Diastolic BP (mmHg)",
        ),
        row=1, col=1,
    )

    # HR trace
    fig.add_trace(
        go.Scatter(
            x=times, y=hr_vals,
            mode="lines+markers",
            line=dict(color="#059669", width=1.5),
            marker=dict(color=status_colors, size=4),
            text=hover_texts,
            hoverinfo="text",
            name="Heart Rate (bpm)",
        ),
        row=2, col=1,
    )

    # Highlight Injected Anomalies (shaded vertical intervals)
    for s in steps:
        if s.is_anomaly:
            fig.add_vrect(
                x0=s.timestamp - timedelta(seconds=28),
                x1=s.timestamp + timedelta(seconds=28),
                fillcolor="rgba(225, 29, 72, 0.08)",
                line_width=0,
                row="all",
            )

    # Highlight Data Gaps (sensor dropouts)
    for s in steps:
        if s.observed is None or np.any(np.isnan(s.observed)):
            fig.add_trace(
                go.Scatter(
                    x=[s.timestamp],
                    y=[120.0],
                    mode="markers",
                    marker=dict(symbol="x", size=10, color="#64748b", line=dict(width=2)),
                    name="Data Gap (Dropout)",
                    hovertext=f"Data Gap: Telemetry missing at {s.timestamp.strftime('%H:%M:%S')}",
                    hoverinfo="text",
                    showlegend=False,
                ),
                row=1, col=1,
            )

    # Highlight What-If Experiments Run
    for w_ev in twin_state.whatif_events:
        fig.add_trace(
            go.Scatter(
                x=[w_ev["timestamp"]],
                y=[155.0],
                mode="markers+text",
                marker=dict(symbol="diamond", size=11, color="#d97706", line=dict(width=1.5)),
                text=["🧪 What-If"],
                textposition="top center",
                name=f"What-If: {w_ev['scenario']}",
                hovertext=f"<b>What-If Experiment</b><br><b>Scenario:</b> {w_ev['scenario']}<br><b>Time:</b> {w_ev['timestamp'].strftime('%H:%M:%S')}<br>{w_ev.get('details', '')}",
                hoverinfo="text",
                showlegend=False,
            ),
            row=1, col=1,
        )

    fig.update_layout(
        height=440,
        margin=dict(l=40, r=20, t=30, b=30),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
    )
    fig.update_yaxes(gridcolor="#f1f5f9", title_text="Pressure (mmHg)", row=1, col=1)
    fig.update_yaxes(gridcolor="#f1f5f9", title_text="Rate (bpm)", row=2, col=1)
    fig.update_xaxes(gridcolor="#f1f5f9")
    return fig


# -----------------------------------------------------------------------------
# Main Application Tabs
# -----------------------------------------------------------------------------
tabs = st.tabs([
    "🫀 Live Vitals & Trends",
    "⚙️ Twin State",
    "❤️ 3D Heart",
    "🧪 What-If Lab",
    "🔍 Explain & Alerts",
    "📊 Validation Dashboard",
    "👥 Care Circle",
    "ℹ️ About CardioTwin X",
])



# =============================================================================
# TAB 1: Live Vitals & Trends
# =============================================================================
with tabs[0]:
    df_hist = twin_mgr.get_telemetry_dataframe(max_points=90)
    div_res = latest_step.divergence_result
    ukf_res = latest_step.ukf_result
    exp_res = latest_step.explanation_report

    # Evaluate Data Reliability (High / Medium / Low)
    rel_level, rel_reason, rel_metrics = twin_mgr.compute_data_reliability()

    # Current readings (updating dynamically every second from synthetic stream)
    cur_obs = latest_step.observed
    cur_sbp = cur_obs[1] if (cur_obs is not None and not np.isnan(cur_obs[1])) else ukf_res.predicted_obs[1]
    cur_dbp = cur_obs[2] if (cur_obs is not None and not np.isnan(cur_obs[2])) else ukf_res.predicted_obs[2]
    cur_hr = cur_obs[0] if (cur_obs is not None and not np.isnan(cur_obs[0])) else ukf_res.predicted_obs[0]
    cur_sv = float(ukf_res.state_mean[2])

    # Prediction text or uncertainty warning when Low
    if rel_level == "Low":
        pred_sbp_sub = '<span style="color: #b91c1c; font-weight: 600; font-size: 0.76rem;">Insufficient or unreliable data. No prediction shown</span>'
        pred_dbp_sub = '<span style="color: #b91c1c; font-weight: 600; font-size: 0.76rem;">Insufficient or unreliable data. No prediction shown</span>'
        pred_hr_sub = '<span style="color: #b91c1c; font-weight: 600; font-size: 0.76rem;">Insufficient or unreliable data. No prediction shown</span>'
        pred_sv_sub = '<span style="color: #b91c1c; font-weight: 600; font-size: 0.76rem;">Insufficient or unreliable data. No prediction shown</span>'
    else:
        pred_sbp_sub = f"Twin Est: {ukf_res.predicted_obs[1]:.0f} mmHg"
        pred_dbp_sub = f"Twin Est: {ukf_res.predicted_obs[2]:.0f} mmHg"
        pred_hr_sub = f"Twin Est: {ukf_res.predicted_obs[0]:.0f} bpm"
        pred_sv_sub = f"Cardiac Output: {(cur_sv * cur_hr / 1000):.2f} L/min"

    # Prominent warning banner when data reliability is Low
    if rel_level == "Low":
        st.markdown(
            f"""
            <div style="background-color: #fef2f2; border: 1.5px solid #fecaca; border-radius: 12px; padding: 12px 16px; margin: 8px 0 16px 0; color: #b91c1c; font-size: 0.92rem;">
                <b>⚠️ Insufficient or unreliable data. No prediction shown</b> — {rel_reason}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Top Vitals Cards Row (Never fixed: varies continuously over time)
    v_col1, v_col2, v_col3, v_col4 = st.columns(4)

    with v_col1:
        st.markdown(
            f"""
            <div class="vital-card">
                <div class="vital-title">Systolic BP</div>
                <div class="vital-value">{cur_sbp:.0f}</div>
                <div class="vital-unit">mmHg · Real-time</div>
                <div class="vital-sub">{pred_sbp_sub}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with v_col2:
        st.markdown(
            f"""
            <div class="vital-card">
                <div class="vital-title">Diastolic BP</div>
                <div class="vital-value">{cur_dbp:.0f}</div>
                <div class="vital-unit">mmHg · Real-time</div>
                <div class="vital-sub">{pred_dbp_sub}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with v_col3:
        st.markdown(
            f"""
            <div class="vital-card">
                <div class="vital-title">Heart Rate</div>
                <div class="vital-value">{cur_hr:.0f}</div>
                <div class="vital-unit">beats / min · Pulse</div>
                <div class="vital-sub">{pred_hr_sub}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with v_col4:
        st.markdown(
            f"""
            <div class="vital-card">
                <div class="vital-title">Stroke Volume</div>
                <div class="vital-value">{cur_sv:.1f}</div>
                <div class="vital-unit">mL / beat · Inotropy</div>
                <div class="vital-sub">{pred_sv_sub}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Minor secondary action: Add reading button
    with st.expander("＋ Add reading (Manual Entry - Secondary Action)", expanded=False):
        st.caption("Manually enter an external cuff blood pressure or heart rate reading to assimilate into the digital twin.")
        m_c1, m_c2, m_c3, m_c4 = st.columns([1, 1, 1, 1.2])
        with m_c1:
            man_sbp = st.number_input("Systolic BP (mmHg)", min_value=60.0, max_value=260.0, value=float(np.clip(cur_sbp, 60.0, 260.0)), step=1.0, key="man_sbp_input")
        with m_c2:
            man_dbp = st.number_input("Diastolic BP (mmHg)", min_value=30.0, max_value=180.0, value=float(np.clip(cur_dbp, 30.0, 180.0)), step=1.0, key="man_dbp_input")
        with m_c3:
            man_hr = st.number_input("Heart Rate (bpm)", min_value=30.0, max_value=220.0, value=float(np.clip(cur_hr, 30.0, 220.0)), step=1.0, key="man_hr_input")
        with m_c4:
            st.write("")
            st.write("")
            if st.button("📥 Submit Reading", key="submit_man_reading_btn"):
                twin_mgr.step_forward(manual_obs=np.array([man_hr, man_sbp, man_dbp]))
                st.success("Manual measurement assimilated!")
                st.rerun()

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # DIVERGENCE PANEL (Core Innovation)
    # -------------------------------------------------------------------------
    badge_class = "badge-normal"
    if div_res.status == "Significant":
        badge_class = "badge-significant"
    elif div_res.status == "Watch":
        badge_class = "badge-watch"

    if rel_level == "High":
        rel_badge = '<span style="background-color: #d1fae5; color: #065f46; border: 1px solid #6ee7b7; padding: 5px 12px; border-radius: 16px; font-size: 0.82rem; font-weight: 700;">🟢 Data Reliability: High</span>'
    elif rel_level == "Medium":
        rel_badge = '<span style="background-color: #fef3c7; color: #92400e; border: 1px solid #fcd34d; padding: 5px 12px; border-radius: 16px; font-size: 0.82rem; font-weight: 700;">🟡 Data Reliability: Medium</span>'
    else:
        rel_badge = '<span style="background-color: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; padding: 5px 12px; border-radius: 16px; font-size: 0.82rem; font-weight: 700;">🔴 Data Reliability: Low</span>'

    st.markdown(
        f"""
        <div style="background-color: #ffffff; border: 1.5px solid #fecdd3; border-radius: 16px; padding: 20px 24px; box-shadow: 0 4px 14px rgba(225, 29, 72, 0.05); margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px; flex-wrap: wrap; gap: 10px;">
                <div>
                    <h3 style="margin: 0; color: #1e293b; font-size: 1.25rem; font-weight: 700;">
                        🎯 Divergence Panel
                    </h3>
                    <div style="font-size: 0.85rem; color: #64748b;">
                        Live comparison of observed telemetry vs twin-predicted hemodynamics (Mahalanobis Innovation Distance)
                    </div>
                </div>
                <div style="display: flex; align-items: center; gap: 10px; flex-wrap: wrap;">
                    {rel_badge}
                    <span style="font-size: 0.85rem; color: #64748b;">Clinical Status:</span>
                    <span class="{badge_class}" style="font-size: 0.95rem; padding: 6px 16px;">
                        {div_res.status}
                    </span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    chart_col, gauge_col = st.columns([3.2, 1.2])

    with chart_col:
        st.markdown("<b>Live Hemodynamic Trends: Observed vs Twin-Predicted with Uncertainty Bands</b>", unsafe_allow_html=True)

        if not df_hist.empty:
            fig_trends = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.08,
                subplot_titles=("Blood Pressure: Observed Points vs 90% Twin Predictive Band", "Heart Rate (bpm)"),
                row_heights=[0.62, 0.38],
            )

            x_time = df_hist["timestamp"]

            # Shaded uncertainty band for SBP [p5, p95]
            fig_trends.add_trace(
                go.Scatter(
                    x=pd.concat([x_time, x_time[::-1]]),
                    y=pd.concat([df_hist["SBP_p95"], df_hist["SBP_p5"][::-1]]),
                    fill="toself",
                    fillcolor="rgba(225, 29, 72, 0.12)",
                    line=dict(color="rgba(255,255,255,0)"),
                    hoverinfo="skip",
                    showlegend=True,
                    name="SBP 90% Twin Band",
                ),
                row=1, col=1,
            )

            # SBP Predicted & Observed
            fig_trends.add_trace(
                go.Scatter(x=x_time, y=df_hist["SBP_pred"], mode="lines", line=dict(color="#e11d48", width=2), name="Predicted SBP"),
                row=1, col=1,
            )
            fig_trends.add_trace(
                go.Scatter(x=x_time, y=df_hist["SBP_obs"], mode="markers", marker=dict(color="#9f1239", size=5), name="Observed SBP"),
                row=1, col=1,
            )

            # Shaded uncertainty band for DBP [p5, p95]
            fig_trends.add_trace(
                go.Scatter(
                    x=pd.concat([x_time, x_time[::-1]]),
                    y=pd.concat([df_hist["DBP_p95"], df_hist["DBP_p5"][::-1]]),
                    fill="toself",
                    fillcolor="rgba(59, 130, 246, 0.10)",
                    line=dict(color="rgba(255,255,255,0)"),
                    hoverinfo="skip",
                    showlegend=True,
                    name="DBP 90% Twin Band",
                ),
                row=1, col=1,
            )

            # DBP Predicted & Observed
            fig_trends.add_trace(
                go.Scatter(x=x_time, y=df_hist["DBP_pred"], mode="lines", line=dict(color="#2563eb", width=2), name="Predicted DBP"),
                row=1, col=1,
            )
            fig_trends.add_trace(
                go.Scatter(x=x_time, y=df_hist["DBP_obs"], mode="markers", marker=dict(color="#1d4ed8", size=5), name="Observed DBP"),
                row=1, col=1,
            )

            # HR Shaded Band & Lines (Row 2)
            fig_trends.add_trace(
                go.Scatter(
                    x=pd.concat([x_time, x_time[::-1]]),
                    y=pd.concat([df_hist["HR_p95"], df_hist["HR_p5"][::-1]]),
                    fill="toself",
                    fillcolor="rgba(16, 185, 129, 0.12)",
                    line=dict(color="rgba(255,255,255,0)"),
                    hoverinfo="skip",
                    showlegend=True,
                    name="HR 90% Twin Band",
                ),
                row=2, col=1,
            )
            fig_trends.add_trace(
                go.Scatter(x=x_time, y=df_hist["HR_pred"], mode="lines", line=dict(color="#059669", width=2), name="Predicted HR"),
                row=2, col=1,
            )
            fig_trends.add_trace(
                go.Scatter(x=x_time, y=df_hist["HR_obs"], mode="markers", marker=dict(color="#047857", size=5), name="Observed HR"),
                row=2, col=1,
            )

            fig_trends.update_layout(
                height=420,
                margin=dict(l=40, r=20, t=30, b=30),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
                plot_bgcolor="#ffffff",
                paper_bgcolor="#ffffff",
            )
            fig_trends.update_yaxes(gridcolor="#f1f5f9", title_text="Pressure (mmHg)", row=1, col=1)
            fig_trends.update_yaxes(gridcolor="#f1f5f9", title_text="Rate (bpm)", row=2, col=1)
            fig_trends.update_xaxes(gridcolor="#f1f5f9")

            st.plotly_chart(fig_trends, use_container_width=True)

    with gauge_col:
        st.markdown("<b>Divergence Score Gauge</b>", unsafe_allow_html=True)
        gauge_val = min(float(div_res.divergence_score), 6.0)

        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=gauge_val,
                title={"text": "Mahalanobis D<sub>M</sub>", "font": {"size": 14}},
                gauge={
                    "axis": {"range": [0, 6], "tickwidth": 1, "tickcolor": "#94a3b8"},
                    "bar": {"color": "#e11d48"},
                    "steps": [
                        {"range": [0, 2.30], "color": "#d1fae5"},
                        {"range": [2.30, 3.14], "color": "#fef3c7"},
                        {"range": [3.14, 6.0], "color": "#fee2e2"},
                    ],
                    "threshold": {
                        "line": {"color": "#b91c1c", "width": 3},
                        "thickness": 0.8,
                        "value": 3.14,
                    },
                },
            )
        )
        fig_gauge.update_layout(
            height=280,
            margin=dict(l=20, r=20, t=30, b=10),
            paper_bgcolor="#ffffff",
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.caption(
            f"<b>Current Episode:</b> {latest_step.episode}<br>"
            f"<b>Cross-Detector:</b> {div_res.detector_agreement}",
            unsafe_allow_html=True,
        )

    # One-line Plain-English Explanation of why the twin differs from observed data
    st.markdown(
        f"""
        <div class="explanation-banner">
            <b>💡 Why Twin Differs:</b> {exp_res.summary}
            <span style="float: right; font-size: 0.82rem; color: #64748b;">Confidence: {exp_res.confidence_label}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Clinical Target-Range Reference Table
    with st.expander("📊 Clinical Target Ranges (AHA Guidelines Reference)", expanded=False):
        ref_df = pd.DataFrame(
            [
                {"Category": "Normal", "Systolic (mmHg)": "< 120", "Diastolic (mmHg)": "< 80", "Status": "Target" if cur_sbp < 120 and cur_dbp < 80 else "-"},
                {"Category": "Elevated", "Systolic (mmHg)": "120 - 129", "Diastolic (mmHg)": "< 80", "Status": "Current" if (120 <= cur_sbp <= 129 and cur_dbp < 80) else "-"},
                {"Category": "Hypertension Stage 1", "Systolic (mmHg)": "130 - 139", "Diastolic (mmHg)": "80 - 89", "Status": "Current" if (130 <= cur_sbp <= 139 or 80 <= cur_dbp <= 89) else "-"},
                {"Category": "Hypertension Stage 2", "Systolic (mmHg)": "≥ 140", "Diastolic (mmHg)": "≥ 90", "Status": "Current" if (cur_sbp >= 140 or cur_dbp >= 90) else "-"},
            ]
        )
        st.dataframe(ref_df, hide_index=True, use_container_width=True)

    # -------------------------------------------------------------------------
    # PATIENT SESSION TIMELINE (Full Session Trajectory & Hover Details)
    # -------------------------------------------------------------------------
    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    st.markdown("### 📈 Patient Session Timeline")
    st.caption("Full session trajectory: continuous telemetry readings, clinical status changes (Normal/Watch/Significant), injected anomalies, what-if simulations, and sensor dropouts with hover inspection.")
    st.plotly_chart(build_patient_timeline_figure(twin_mgr), use_container_width=True)


# =============================================================================
# TAB 2: Twin State
# =============================================================================
with tabs[1]:
    st.markdown("### Estimated Physiological State & Continuous Waveform")
    st.caption("Internal parameters continuously adapted by the Unscented Kalman Filter from non-invasive telemetry.")

    p_est = ukf_res.state_mean
    p_std = ukf_res.state_std

    p_col1, p_col2, p_col3, p_col4 = st.columns(4)

    with p_col1:
        st.markdown(
            f"""
            <div class="vital-card">
                <div class="vital-title">Peripheral Resistance (R<sub>p</sub>)</div>
                <div class="vital-value">{p_est[0]:.2f}</div>
                <div class="vital-unit">mmHg · s / mL</div>
                <div class="vital-sub">± {p_std[0]:.3f} (1-σ uncertainty)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with p_col2:
        st.markdown(
            f"""
            <div class="vital-card">
                <div class="vital-title">Arterial Compliance (C)</div>
                <div class="vital-value">{p_est[1]:.2f}</div>
                <div class="vital-unit">mL / mmHg</div>
                <div class="vital-sub">± {p_std[1]:.3f} (1-σ uncertainty)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with p_col3:
        st.markdown(
            f"""
            <div class="vital-card">
                <div class="vital-title">Stroke Volume (SV)</div>
                <div class="vital-value">{p_est[2]:.1f}</div>
                <div class="vital-unit">mL</div>
                <div class="vital-sub">± {p_std[2]:.1f} mL uncertainty</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with p_col4:
        st.markdown(
            f"""
            <div class="vital-card">
                <div class="vital-title">Twin Heart Rate (HR)</div>
                <div class="vital-value">{p_est[3]:.1f}</div>
                <div class="vital-unit">beats / min</div>
                <div class="vital-sub">± {p_std[3]:.1f} bpm uncertainty</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    wave_col, trend_col = st.columns([1.8, 1.8])

    with wave_col:
        st.markdown("<b>Simulated Instantaneous Cardiac Cycle Pressure Waveform</b>", unsafe_allow_html=True)
        # Simulate beat using current estimated parameters
        cur_wp = WindkesselParams(
            Rp=p_est[0], C=p_est[1], Rc=0.05, SV=p_est[2], HR=p_est[3]
        )
        beat_out = simulate_beat(cur_wp, n_points=150)

        fig_wave = go.Figure()
        fig_wave.add_trace(
            go.Scatter(
                x=beat_out.time * 1000.0,
                y=beat_out.pressure,
                mode="lines",
                line=dict(color="#e11d48", width=3),
                name="Aortic Pressure (mmHg)",
            )
        )
        fig_wave.add_trace(
            go.Scatter(
                x=beat_out.time * 1000.0,
                y=beat_out.flow * 0.25 + 40.0,
                mode="lines",
                line=dict(color="#3b82f6", width=1.5, dash="dash"),
                name="Systolic Ejection Flow (scaled)",
            )
        )
        fig_wave.update_layout(
            height=320,
            margin=dict(l=40, r=20, t=20, b=30),
            xaxis_title="Time within Cardiac Cycle (ms)",
            yaxis_title="Pressure (mmHg)",
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        )
        fig_wave.update_yaxes(gridcolor="#f1f5f9")
        fig_wave.update_xaxes(gridcolor="#f1f5f9")
        st.plotly_chart(fig_wave, use_container_width=True)

    with trend_col:
        st.markdown("<b>Estimated Parameter Trajectory vs Baseline</b>", unsafe_allow_html=True)
        df_p = twin_mgr.get_telemetry_dataframe(max_points=60)
        if not df_p.empty:
            fig_pt = go.Figure()
            rp_base = twin_mgr.base_params.Rp
            c_base = twin_mgr.base_params.C

            fig_pt.add_trace(
                go.Scatter(
                    x=df_p["timestamp"],
                    y=df_p["Rp_est"] / rp_base,
                    mode="lines",
                    line=dict(color="#e11d48", width=2),
                    name="Resistance (Rp / baseline)",
                )
            )
            fig_pt.add_trace(
                go.Scatter(
                    x=df_p["timestamp"],
                    y=df_p["C_est"] / c_base,
                    mode="lines",
                    line=dict(color="#2563eb", width=2),
                    name="Compliance (C / baseline)",
                )
            )
            fig_pt.add_hline(y=1.0, line_dash="dash", line_color="#94a3b8")
            fig_pt.update_layout(
                height=320,
                margin=dict(l=40, r=20, t=20, b=30),
                xaxis_title="Time",
                yaxis_title="Ratio to Resting Baseline",
                plot_bgcolor="#ffffff",
                paper_bgcolor="#ffffff",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
            )
            fig_pt.update_yaxes(gridcolor="#f1f5f9")
            fig_pt.update_xaxes(gridcolor="#f1f5f9")
            st.plotly_chart(fig_pt, use_container_width=True)


# =============================================================================
# TAB 3: 3D Heart
# =============================================================================
with tabs[2]:
    st.markdown("### 🫀 3D Anatomical Heart & Hemodynamic Lub-Dub Biomechanics")
    st.caption(
        "Interactive 3D anatomical cardiovascular model featuring realistic ventricular apex, aortic arch with branching vessels, "
        "pulmonary trunk, superior vena cava, and coronary vasculature network. Physiologically driven in real-time by Blood Pressure (Lub-Dub)."
    )

    heart_html = generate_3d_heart_html(
        sbp=cur_sbp,
        dbp=cur_dbp,
        hr=cur_hr,
        sv=cur_sv,
        div_status=div_res.status,
        div_score=float(div_res.divergence_score),
    )
    st.components.v1.html(heart_html, height=640, scrolling=False)

    guide_col1, guide_col2 = st.columns([1.5, 1.5])

    with guide_col1:
        st.markdown(
            f"""
            <div style="background-color: #ffffff; border: 1px solid #fecdd3; border-radius: 14px; padding: 18px; box-shadow: 0 2px 10px rgba(225, 29, 72, 0.04); margin-top: 10px;">
                <h4 style="margin: 0 0 10px 0; color: #e11d48; font-size: 1.05rem;">
                    🫀 Anatomical Architecture (Visualized)
                </h4>
                <ul style="padding-left: 18px; font-size: 0.88rem; line-height: 1.6; color: #334155;">
                    <li><b>Conical Ventricles:</b> Muscular left and right ventricles tapering inferolaterally to the anatomical cardiac apex.</li>
                    <li><b>Aortic Arch (Red):</b> Sweeping ascending aorta with the <b>3 brachiocephalic, carotid, and subclavian branch arteries</b> emerging superiorly.</li>
                    <li><b>Pulmonary Trunk & Arteries (Blue):</b> Prominent cobalt blue vessel emerging anterior to aorta and dividing into left and right pulmonary branches.</li>
                    <li><b>Superior Vena Cava:</b> Vertical deoxygenated inflow vessel entering the right atrium.</li>
                    <li><b>Coronary Vasculature Network:</b> Branching anterior interventricular veins (blue) and arteries (red) traversing the sulcus with fine myocardial arborizations.</li>
                    <li><b>Right & Left Auricles:</b> Lobulated muscular pouches flanking the great vessels.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with guide_col2:
        st.markdown(
            f"""
            <div style="background-color: #ffffff; border: 1px solid #fecdd3; border-radius: 14px; padding: 18px; box-shadow: 0 2px 10px rgba(225, 29, 72, 0.04); margin-top: 10px;">
                <h4 style="margin: 0 0 10px 0; color: #e11d48; font-size: 1.05rem;">
                    💓 "Lub-Dub" Biomechanical BP Coupling
                </h4>
                <div style="font-size: 0.88rem; line-height: 1.6; color: #334155;">
                    <p style="margin-bottom: 6px;">
                        <b>1. "LUB" (S1 - Ventricular Systole):</b> Mitral/tricuspid valve closure snaps at onset of systole. 
                        Ventricles undergo forceful radial contraction and apical wringing twist. Contraction force is scaled directly by 
                        <b>Pulse Pressure (ΔP = {cur_sbp - cur_dbp:.0f} mmHg)</b> and Stroke Volume ({cur_sv:.1f} mL). 
                        Concurrently, the aortic arch dilates under systolic surge (<b>{cur_sbp:.0f} mmHg</b>).
                    </p>
                    <p style="margin-bottom: 6px;">
                        <b>2. "DUB" (S2 - Aortic Valve Closure):</b> Aortic semilunar valves snap shut at end-ejection, causing 
                        the characteristic dicrotic notch recoil vibration on the aortic arch.
                    </p>
                    <p style="margin-bottom: 0;">
                        <b>3. Diastolic Filling & Audio:</b> Ventricles relax smoothly to baseline end-diastolic volume at <b>{cur_dbp:.0f} mmHg</b>. 
                        Tempo is paced to exact heart rate (<b>{cur_hr:.0f} bpm</b>, T = {60.0 / max(cur_hr, 1.0):.2f}s). 
                        Click <b>Sound: ON</b> in the 3D viewer to hear the synthesized acoustic phonocardiogram.
                    </p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =============================================================================
# TAB 4: What-If Lab
# =============================================================================
with tabs[3]:
    st.markdown("### What-If Counterfactual Hemodynamic Simulator")
    st.caption("Perturb the digital twin's physiological state to simulate counterfactual clinical or lifestyle scenarios.")

    st.warning(f"⚠️ {MEDICAL_DISCLAIMER}")

    scen_col1, scen_col2 = st.columns([1.3, 2.3])

    with scen_col1:
        st.markdown("<b>Select Scenario:</b>", unsafe_allow_html=True)
        scenario_key = st.selectbox(
            "Scenario Preset",
            options=list(WHATIF_SCENARIOS.keys()) + ["custom"],
            format_func=lambda k: "Custom Sliders" if k == "custom" else WHATIF_SCENARIOS[k].name,
        )

        custom_delta = None
        if scenario_key == "custom":
            st.markdown("<b>Custom Perturbation Sliders:</b>", unsafe_allow_html=True)
            c_rp = st.slider("Peripheral Resistance Multiplier (Rp)", 0.50, 2.00, 1.00, 0.05)
            c_c = st.slider("Compliance Multiplier (C)", 0.30, 1.80, 1.00, 0.05)
            c_sv = st.slider("Stroke Volume Delta (mL)", -25.0, 35.0, 0.0, 1.0)
            c_hr = st.slider("Heart Rate Delta (bpm)", -25.0, 50.0, 0.0, 1.0)
            custom_delta = ScenarioDelta(
                name="Custom Scenario",
                description="User-defined hemodynamic parameter perturbation.",
                rp_multiplier=c_rp,
                c_multiplier=c_c,
                sv_delta=c_sv,
                hr_delta=c_hr,
            )
        else:
            preset_scen = WHATIF_SCENARIOS[scenario_key]
            st.info(f"ℹ️ {preset_scen.description}")

    # Run the what-if simulation
    cur_p = WindkesselParams(
        Rp=p_est[0], C=p_est[1], Rc=0.05, SV=p_est[2], HR=p_est[3]
    )
    whatif_res = run_whatif_simulation(
        current_params=cur_p,
        scenario_key=scenario_key,
        custom_delta=custom_delta,
        current_cov=ukf_res.state_cov,
    )

    with scen_col1:
        st.write("")
        if st.button("📌 Log Scenario to Timeline", key="log_whatif_scenario_btn", help="Register this counterfactual scenario onto the patient timeline"):
            twin_mgr.log_whatif_event(
                whatif_res.scenario_name,
                details=f"ΔSBP: {whatif_res.delta_sbp:+.1f} mmHg, ΔDBP: {whatif_res.delta_dbp:+.1f} mmHg, ΔHR: {whatif_res.delta_hr:+.0f} bpm",
            )
            st.success(f"Logged '{whatif_res.scenario_name}' onto Patient Session Timeline!")
            st.rerun()

    with scen_col2:
        st.markdown("<b>Simulated Outcomes (Before vs After):</b>", unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Systolic BP", f"{whatif_res.output_after.sbp:.0f} mmHg", delta=f"{whatif_res.delta_sbp:+.1f} mmHg", delta_color="inverse")
        m2.metric("Diastolic BP", f"{whatif_res.output_after.dbp:.0f} mmHg", delta=f"{whatif_res.delta_dbp:+.1f} mmHg", delta_color="inverse")
        m3.metric("Mean Pressure", f"{whatif_res.output_after.map:.0f} mmHg", delta=f"{whatif_res.delta_map:+.1f} mmHg", delta_color="inverse")
        m4.metric("Heart Rate", f"{whatif_res.output_after.hr:.0f} bpm", delta=f"{whatif_res.delta_hr:+.0f} bpm", delta_color="inverse")

        fig_whatif = go.Figure()

        t_ms_b = whatif_res.output_before.time * 1000.0
        fig_whatif.add_trace(
            go.Scatter(
                x=t_ms_b,
                y=whatif_res.output_before.pressure,
                mode="lines",
                line=dict(color="#64748b", width=2, dash="dash"),
                name="Current Baseline State",
            )
        )

        t_ms_a = whatif_res.output_after.time * 1000.0
        fig_whatif.add_trace(
            go.Scatter(
                x=t_ms_a,
                y=whatif_res.output_after.pressure,
                mode="lines",
                line=dict(color="#e11d48", width=3),
                name=f"Simulated: {whatif_res.scenario_name}",
            )
        )

        fig_whatif.update_layout(
            height=300,
            margin=dict(l=30, r=20, t=20, b=30),
            xaxis_title="Cardiac Cycle Time (ms)",
            yaxis_title="Pressure (mmHg)",
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        )
        fig_whatif.update_yaxes(gridcolor="#f1f5f9")
        fig_whatif.update_xaxes(gridcolor="#f1f5f9")
        st.plotly_chart(fig_whatif, use_container_width=True)


# =============================================================================
# TAB 5: Explain & Alerts
# =============================================================================
with tabs[4]:
    st.markdown("### Physiological Explainability & Divergence Diagnostics")
    st.caption("Dissects hemodynamic anomalies to isolate root physiological mechanisms and parameter deviations.")

    exp_col1, exp_col2 = st.columns([2.0, 1.6])

    with exp_col1:
        st.markdown("<b>Current Diagnostic Narrative</b>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="background-color: #ffffff; border: 1px solid #fecdd3; border-radius: 14px; padding: 18px; box-shadow: 0 2px 8px rgba(225, 29, 72, 0.04);">
                <div style="font-size: 1.1rem; font-weight: 700; color: #e11d48; margin-bottom: 8px;">
                    {exp_res.summary}
                </div>
                <p style="color: #334155; line-height: 1.6; font-size: 0.95rem;">
                    {exp_res.detailed_explanation}
                </p>
                <div style="margin-top: 14px; display: flex; gap: 10px; flex-wrap: wrap;">
                    <span style="background-color: #f1f5f9; padding: 4px 10px; border-radius: 8px; font-size: 0.8rem; font-weight: 600;">
                        Primary Shift: {exp_res.primary_driver_description}
                    </span>
                    <span style="background-color: #ecfdf5; color: #047857; padding: 4px 10px; border-radius: 8px; font-size: 0.8rem; font-weight: 600;">
                        {exp_res.confidence_label}
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
        st.markdown("<b>Dual-Detector Cross-Validation</b>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 14px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <b>UKF Innovation Mahalanobis:</b> {div_res.divergence_score:.2f} ({div_res.status})<br>
                        <b>IsolationForest Residual Detector:</b> {'Flagged Anomaly' if div_res.isolation_forest_anomaly else 'Nominal Inlier'} (Score: {div_res.isolation_forest_score:.3f})
                    </div>
                    <div>
                        <span class="badge-normal" style="font-size: 0.85rem;">{div_res.detector_agreement}</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with exp_col2:
        st.markdown("<b>Per-Signal Contribution to Divergence</b>", unsafe_allow_html=True)
        sig_data = exp_res.signal_contributions
        fig_contrib = go.Figure(
            go.Bar(
                x=[sig_data.get("HR", 0), sig_data.get("SBP", 0), sig_data.get("DBP", 0)],
                y=["Heart Rate", "Systolic BP", "Diastolic BP"],
                orientation="h",
                marker=dict(color=["#059669", "#e11d48", "#2563eb"]),
            )
        )
        fig_contrib.update_layout(
            height=260,
            margin=dict(l=20, r=20, t=10, b=30),
            xaxis_title="% Contribution to Normalized Innovation",
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
        )
        fig_contrib.update_xaxes(range=[0, 100], gridcolor="#f1f5f9")
        st.plotly_chart(fig_contrib, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🚨 Explainable Clinical Alerts Log")
    st.caption("Complete historical record of all Watch and Significant divergence alerts stored with top contributing signals, physiological parameter shift explanations, and data reliability status.")

    if twin_mgr.alerts_log:
        st.markdown(f"<b>Session Alerts ({len(twin_mgr.alerts_log)} recorded):</b>", unsafe_allow_html=True)
        for a in reversed(twin_mgr.alerts_log[-8:]):
            st_color = "#e11d48" if a["status"] == "Significant" else "#d97706"
            st_bg = "#fff1f2" if a["status"] == "Significant" else "#fffbeb"
            rel_icon = "🟢" if a["reliability_level"] == "High" else ("🟡" if a["reliability_level"] == "Medium" else "🔴")

            st.markdown(
                f"""
                <div style="background-color: #ffffff; border-left: 5px solid {st_color}; border-radius: 12px; padding: 14px 18px; margin-bottom: 12px; box-shadow: 0 2px 6px rgba(0,0,0,0.04); border-top: 1px solid #f1f5f9;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; flex-wrap: wrap; gap: 8px;">
                        <div>
                            <span style="font-weight: 800; color: {st_color}; font-size: 1.05rem;">Alert #{a['id']} · {a['status']}</span>
                            <span style="font-size: 0.85rem; color: #64748b; margin-left: 10px;">🕒 {a['timestamp']} · Episode: {a.get('episode', 'N/A')}</span>
                        </div>
                        <div style="display: flex; gap: 8px; align-items: center;">
                            <span style="font-size: 0.82rem; font-weight: 700; padding: 3px 10px; border-radius: 12px; background-color: #f8fafc; border: 1px solid #cbd5e1;">
                                {rel_icon} Reliability: {a['reliability_level']}
                            </span>
                            <span style="font-size: 0.82rem; font-weight: 700; padding: 3px 10px; border-radius: 12px; background-color: {st_bg}; color: {st_color};">
                                D<sub>M</sub>: {a['divergence_score']:.2f}
                            </span>
                        </div>
                    </div>
                    <div style="font-size: 0.94rem; color: #1e293b; margin-bottom: 8px;">
                        <b>Parameter Shift Explanation:</b> {a['parameter_shift']}
                    </div>
                    <div style="display: flex; gap: 24px; font-size: 0.85rem; color: #475569; flex-wrap: wrap; background-color: #f8fafc; padding: 8px 12px; border-radius: 8px;">
                        <div><b>Top Contributing Signals:</b> {a['top_signals']}</div>
                        <div><b>Cross Agreement:</b> {a.get('detector_agreement', 'N/A')}</div>
                        <div><b>Reliability Note:</b> {a.get('reliability_reason', 'Nominal')}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with st.expander("📋 View All Recorded Alerts as Table", expanded=False):
            alert_table_data = [
                {
                    "Alert #": a["id"],
                    "Time": a["timestamp"],
                    "Status": a["status"],
                    "Divergence Score": f"{a['divergence_score']:.2f}",
                    "Top Contributing Signals": a["top_signals"],
                    "Parameter Shift Explanation": a["parameter_shift"],
                    "Reliability Level": a["reliability_level"],
                    "Detector Agreement": a["detector_agreement"],
                }
                for a in reversed(twin_mgr.alerts_log)
            ]
            st.dataframe(pd.DataFrame(alert_table_data), hide_index=True, use_container_width=True)
    else:
        st.info("No Watch or Significant alerts recorded in this session yet. As the synthetic stream encounters anomalies or episodes, alerts with physiological explanations will be logged here.")


# =============================================================================
# TAB 6: Validation Dashboard
# =============================================================================
with tabs[5]:
    st.markdown("### 📊 Cohort Validation Dashboard")
    st.caption("Systematic empirical evaluation of CardioTwin X across 30 held-out synthetic patients.")

    st.info(f"ℹ️ **Validation Methodology Notice:** {VALIDATION_NOTE}")

    val_summary = get_cached_validation_summary()

    st.markdown("#### (a) One-Step-Ahead Predictive Accuracy (MAE & RMSE)")
    vm_c1, vm_c2, vm_c3, vm_c4 = st.columns(4)
    with vm_c1:
        st.metric(
            "SBP MAE / RMSE",
            f"{val_summary.overall_mae['SBP']:.2f} / {val_summary.overall_rmse['SBP']:.2f} mmHg",
            help="Mean Absolute Error and Root Mean Squared Error for Systolic Blood Pressure",
        )
    with vm_c2:
        st.metric(
            "DBP MAE / RMSE",
            f"{val_summary.overall_mae['DBP']:.2f} / {val_summary.overall_rmse['DBP']:.2f} mmHg",
            help="Mean Absolute Error and Root Mean Squared Error for Diastolic Blood Pressure",
        )
    with vm_c3:
        st.metric(
            "HR MAE / RMSE",
            f"{val_summary.overall_mae['HR']:.2f} / {val_summary.overall_rmse['HR']:.2f} bpm",
            help="Mean Absolute Error and Root Mean Squared Error for Heart Rate",
        )
    with vm_c4:
        st.metric(
            "90% Band Coverage",
            f"{val_summary.observed_coverage_90:.1f}%",
            delta=f"{val_summary.observed_coverage_90 - 90.0:+.1f}% vs nominal",
            help="Percentage of true observations falling within the twin's 90% predictive uncertainty envelope",
        )

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    v_col1, v_col2 = st.columns([1.8, 2.0])

    with v_col1:
        st.markdown("#### (b) Uncertainty Calibration (Expected vs Observed Coverage)")
        fig_cal = go.Figure()
        exp_cov = val_summary.calibration_curve["expected"]
        obs_cov = val_summary.calibration_curve["observed"]

        fig_cal.add_trace(
            go.Scatter(
                x=[0, 100], y=[0, 100],
                mode="lines",
                line=dict(color="#94a3b8", dash="dash"),
                name="Ideal Calibration (y=x)",
            )
        )
        fig_cal.add_trace(
            go.Scatter(
                x=exp_cov, y=obs_cov,
                mode="lines+markers",
                line=dict(color="#e11d48", width=2.5),
                marker=dict(size=8, color="#be123c"),
                name="CardioTwin X Ensemble Coverage",
            )
        )
        fig_cal.update_layout(
            height=340,
            margin=dict(l=30, r=20, t=20, b=30),
            xaxis_title="Nominal / Expected Confidence Level (%)",
            yaxis_title="Empirical Observed Coverage (%)",
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        )
        fig_cal.update_xaxes(range=[40, 100], gridcolor="#f1f5f9")
        fig_cal.update_yaxes(range=[40, 100], gridcolor="#f1f5f9")
        st.plotly_chart(fig_cal, use_container_width=True)
        st.caption(f"Uncertainty calibration: Nominal 90% predictive band achieved **{val_summary.observed_coverage_90:.1f}%** observed coverage across all 30 patients.")

    with v_col2:
        st.markdown("#### (c) Anomaly Detection Benchmark (Divergence Engine vs IsolationForest)")
        div_m = val_summary.divergence_metrics
        ifo_m = val_summary.isoforest_metrics

        fig_bench = go.Figure()
        fig_bench.add_trace(
            go.Bar(
                name="Divergence Engine (Innovation)",
                x=["Detection Rate (TPR %)", "False Alarm Rate (FPR %)", "Detection Delay (steps)"],
                y=[div_m["tpr_pct"], div_m["fpr_pct"], div_m["mean_delay_steps"]],
                marker_color="#e11d48",
            )
        )
        fig_bench.add_trace(
            go.Bar(
                name="IsolationForest (Residuals)",
                x=["Detection Rate (TPR %)", "False Alarm Rate (FPR %)", "Detection Delay (steps)"],
                y=[ifo_m["tpr_pct"], ifo_m["fpr_pct"], ifo_m["mean_delay_steps"]],
                marker_color="#3b82f6",
            )
        )
        fig_bench.update_layout(
            barmode="group",
            height=340,
            margin=dict(l=30, r=20, t=20, b=30),
            yaxis_title="Metric Value",
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        )
        fig_bench.update_yaxes(gridcolor="#f1f5f9")
        fig_bench.update_xaxes(gridcolor="#f1f5f9")
        st.plotly_chart(fig_bench, use_container_width=True)
        st.caption(f"Divergence Engine achieves {div_m['tpr_pct']:.1f}% TPR ({div_m['fpr_pct']:.1f}% FPR, delay: {div_m['mean_delay_steps']:.1f} steps) vs IsolationForest {ifo_m['tpr_pct']:.1f}% TPR ({ifo_m['fpr_pct']:.1f}% FPR, delay: {ifo_m['mean_delay_steps']:.1f} steps).")

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    st.markdown("#### (d) Validation Metrics Stratified by Patient Preset")
    st.dataframe(val_summary.preset_table, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 📂 Public Benchmark Dataset Loader (PTB-XL / BIDMC)")
    pub_info = val_summary.public_dataset_status
    if pub_info.get("loaded", False):
        st.success(f"Loaded public dataset: {pub_info.get('dataset_name')} ({pub_info.get('records_count')} records) from {pub_info.get('path')}")
    else:
        st.warning(f"📁 **public dataset not loaded** — {pub_info.get('message', 'No local benchmark files found.')}")
        st.caption("Upload or place `ptbxl_database.csv` or `bidmc_data.csv` in the project root to benchmark against real-world clinical recordings.")

    uploaded_file = st.file_uploader("Upload Public Dataset CSV (e.g. PTB-XL metadata or BIDMC waveform)", type=["csv"], key="public_dataset_uploader")
    if uploaded_file is not None:
        try:
            df_uploaded = pd.read_csv(uploaded_file, nrows=100)
            st.success(f"Uploaded public dataset preview: {uploaded_file.name} ({len(df_uploaded)} rows previewed, {len(df_uploaded.columns)} columns).")
            st.dataframe(df_uploaded.head(5), use_container_width=True)
        except Exception as e:
            st.error(f"Error reading uploaded dataset: {e}")


# =============================================================================
# TAB 7: Care Circle & User Reminders
# =============================================================================
with tabs[6]:
    st.markdown("### Care Circle & Reminders")
    st.caption("Coordinate cardiovascular telemetry oversight with designated clinicians and care contacts.")

    cc_col1, cc_col2 = st.columns([2.0, 1.6])

    with cc_col1:
        st.markdown("<b>Connected Care Contacts</b>", unsafe_allow_html=True)
        for idx, contact in enumerate(twin_mgr.care_circle):
            st.markdown(
                f"""
                <div style="background-color: #ffffff; border: 1px solid #fecdd3; border-radius: 12px; padding: 14px; margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <span style="font-size: 1.05rem; font-weight: 700; color: #1e293b;">{contact['name']}</span>
                            <span style="font-size: 0.8rem; background-color: #fff1f2; color: #e11d48; padding: 2px 8px; border-radius: 6px; margin-left: 8px;">{contact['role']}</span>
                            <div style="font-size: 0.85rem; color: #64748b; margin-top: 4px;">{contact['clinic']} · {contact['phone']}</div>
                        </div>
                        <div>
                            <span class="{'badge-normal' if contact['alert_enabled'] else 'badge-sync'}">
                                {'Alerts Active (Demo)' if contact['alert_enabled'] else 'Alerts Muted'}
                            </span>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with st.expander("➕ Add Family Member or Caregiver", expanded=False):
            with st.form("add_contact_form"):
                new_name = st.text_input("Full Name")
                new_role = st.text_input("Relationship / Role (e.g. Caregiver, Sibling)")
                new_phone = st.text_input("Phone Number")
                new_alert = st.checkbox("Enable Telemetry Divergence Alerts (Simulated)", value=True)
                submitted = st.form_submit_button("Add to Care Circle")
                if submitted and new_name:
                    twin_mgr.care_circle.append(
                        {
                            "name": new_name,
                            "role": new_role or "Care Contact",
                            "clinic": "Personal Contact",
                            "phone": new_phone or "N/A",
                            "alert_enabled": new_alert,
                        }
                    )
                    st.success(f"Added {new_name} to Care Circle!")
                    st.rerun()

    with cc_col2:
        st.markdown("<b>Personal Reminders</b>", unsafe_allow_html=True)
        st.caption("User-configured daily lifestyle reminders. No medical prescription or dosage advice.")

        for r_text in twin_mgr.reminders:
            st.markdown(
                f"""
                <div style="background-color: #f8fafc; border-left: 3px solid #3b82f6; padding: 8px 12px; border-radius: 6px; margin-bottom: 8px; font-size: 0.88rem;">
                    {r_text}
                </div>
                """,
                unsafe_allow_html=True,
            )

        with st.form("add_reminder_form"):
            new_reminder = st.text_input("Add New Daily Reminder", placeholder="e.g. Evening walk in park")
            r_submit = st.form_submit_button("Save Reminder")
            if r_submit and new_reminder:
                twin_mgr.reminders.append(f"Your reminder: {new_reminder}")
                st.success("Reminder added!")
                st.rerun()


# =============================================================================
# TAB 8: About CardioTwin X
# =============================================================================
with tabs[7]:
    st.markdown("### About CardioTwin X")
    st.markdown(
        """
        CardioTwin X is a practical, transparent, interactive implementation of a cardiac digital twin using synthetic data and a simplified physiological model. It does not claim to be a novel invention or clinical-grade product, but a research and educational prototype built to demonstrate how digital twin concepts could be applied in a real-time monitoring workflow.
        """
    )

    st.markdown("#### System Architecture")
    st.markdown(
        """
        ```
        +--------------------------------------------------------------------------+
        |                          PATIENT TELEMETRY                               |
        |  Wearable Sensor / Blood Pressure Monitor: HR, SBP, DBP, ECG features   |
        +--------------------------------------------------------------------------+
                                           |
                                           v
        +--------------------------------------------------------------------------+
        |               ADAPTIVE ESTIMATOR: UNSCENTED KALMAN FILTER                |
        |  State: [Rp, C, SV, HR]  <--->  Observation Model: h(x) WK3              |
        |  Covariance P Update with Missing Data Skip                              |
        +--------------------------------------------------------------------------+
                                           |
                                           v
        +--------------------------------------------------------------------------+
        |                     DIVERGENCE ENGINE (CORE INNOVATION)                  |
        |  • Mahalanobis Distance: D_M = sqrt(v^T * Pyy^-1 * v)                    |
        |  • Persistence Filter (N=3 consecutive steps above threshold)            |
        |  • 50-Twin Monte Carlo Ensemble: 90% Predictive Uncertainty Bands        |
        |  • Secondary Cross-Validator: IsolationForest on Residual Features       |
        +--------------------------------------------------------------------------+
                          |                                      |
                          v                                      v
        +-----------------------------------+  +-----------------------------------+
        |       PHYSIOLOGICAL EXPLAIN       |  |          WHAT-IF LAB LAB          |
        |  • Parameter Shift Sigma Analysis |  |  • Counterfactual Perturbations   |
        |  • Plain-English Diagnostic Text  |  |  • Before vs After Hemodynamics  |
        |  • Per-Signal Contribution Bars   |  |  • 3D Cardiac Mesh Deformation    |
        +-----------------------------------+  +-----------------------------------+
        ```
        """
    )

    st.markdown("#### Key Scientific Assumptions & Limitations")
    st.markdown(
        """
        1. **Synthetic Data**: Telemetry streams are produced by mathematical simulation with diurnal rhythms and clinical episodes. No actual human subject data is transmitted or stored.
        2. **3-Element Windkessel Approximations**: Arterial hemodynamics are modeled via lumped-parameter differential equations (Rp, C, Rc). Wave reflection phenomena and localized spatial arterial gradients are intentionally simplified.
        3. **Circular Validation Consideration**: Because both the synthetic generator and the digital twin utilize Windkessel foundations, validation within this environment illustrates model consistency more than true external generalization.
        4. **Future Roadmap**: Integration with benchmark datasets (e.g. PhysioNet PTB-XL, BIDMC PPG/ABP) and wearable sensor Bluetooth LE ingestion.
        """
    )


# -----------------------------------------------------------------------------
# Persistent Footer on Every Tab
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="footer-text">
        Research and demonstration prototype using synthetic data. Not medical advice.
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Smooth Live Streaming Loop
# -----------------------------------------------------------------------------
is_testing = (
    os.environ.get("STREAMLIT_TESTING") == "1"
    or os.environ.get("PYTEST_CURRENT_TEST") is not None
)

if not is_testing and st.session_state.get("auto_stream", False):
    time.sleep(float(twin_mgr.playback_speed))
    twin_mgr.step_forward()
    st.rerun()
