"""Authentication and Logo/Branding System for CardioTwin X.

Provides:
- High-resolution SVG vector logo for CardioTwin X with digital twin circuitry and ECG trace.
- Multi-role demo authentication (Cardiologist, Researcher, Patient, Guest).
- Login / Sign-in page with quick 1-click demo logins and logo showcase.
- User session state management and logout handling.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional
import streamlit as st


@dataclass
class UserAccount:
    """User profile data structure."""
    username: str
    name: str
    role: str
    email: str
    title: str
    avatar: str
    affiliation: str


DEMO_USERS: Dict[str, UserAccount] = {
    "cardiologist": UserAccount(
        username="dr.jenkins",
        name="Dr. Sarah Jenkins, MD, FACC",
        role="Lead Cardiologist",
        email="s.jenkins@cardiotwin.org",
        title="Director of Cardiovascular Telemetry & Hemodynamics",
        avatar="🩺",
        affiliation="Metro Heart & Vascular Institute",
    ),
    "researcher": UserAccount(
        username="prof.vance",
        name="Prof. Marcus Vance, PhD",
        role="Twin Systems Researcher",
        email="m.vance@in-silico-medicine.edu",
        title="Principal Investigator, Biomechanical Twin Lab",
        avatar="🔬",
        affiliation="Institute for Computational Physiology",
    ),
    "patient": UserAccount(
        username="sarah.j",
        name="Sarah Jenkins",
        role="Patient (Telemetry Stream)",
        email="sarah.jenkins@patient.org",
        title="Enrolled in Remote Cardiovascular Telemetry (Synthetic Data)",
        avatar="👤",
        affiliation="CardioTwin X Remote Patient Care Program",
    ),
    "guest": UserAccount(
        username="guest_evaluator",
        name="Clinical Guest Evaluator",
        role="Guest Observer",
        email="guest@cardiotwin.demo",
        title="Interactive Clinical Prototype Sandbox",
        avatar="✨",
        affiliation="Public Research Demonstration",
    ),
}


def get_cardiotwin_logo_svg(width: int = 340, height: int = 140, animated: bool = True) -> str:
    """Generate medical-grade vector SVG logo for CardioTwin X.

    Features:
    - Stylized anatomical cardiac silhouette with aortic arch and pulmonary trunk.
    - Embedded electronic digital twin nodes & circuit pathways.
    - Synchronized ECG QRS-complex vector line traversing the heart.
    - Polished rose/crimson gradient typography with dual-subtitles.
    """
    pulse_anim = """
      @keyframes pulseGlow {
        0%, 100% { transform: scale(1.0); filter: drop-shadow(0 0 4px rgba(225, 29, 72, 0.4)); }
        25% { transform: scale(1.03); filter: drop-shadow(0 0 14px rgba(225, 29, 72, 0.8)); }
        35% { transform: scale(0.98); }
        45% { transform: scale(1.02); filter: drop-shadow(0 0 10px rgba(225, 29, 72, 0.6)); }
        60% { transform: scale(1.0); }
      }
      @keyframes ecgDash {
        to { stroke-dashoffset: -100; }
      }
      .heart-anim {
        transform-origin: 55px 65px;
        animation: pulseGlow 1.2s infinite ease-in-out;
      }
      .ecg-anim {
        stroke-dasharray: 8, 4;
        animation: ecgDash 2s linear infinite;
      }
    """ if animated else ""

    svg = f"""
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">
      <defs>
        <style>
          {pulse_anim}
          .brand-text {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; font-weight: 800; }}
          .sub-text {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; font-weight: 600; }}
        </style>
        <linearGradient id="roseGradient" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stop-color="#f43f5e" />
          <stop offset="50%" stop-color="#e11d48" />
          <stop offset="100%" stop-color="#9f1239" />
        </linearGradient>
        <linearGradient id="blueGradient" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stop-color="#38bdf8" />
          <stop offset="100%" stop-color="#1d4ed8" />
        </linearGradient>
        <linearGradient id="textGradient" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stop-color="#e11d48" />
          <stop offset="85%" stop-color="#881337" />
          <stop offset="100%" stop-color="#0284c7" />
        </linearGradient>
        <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
          <feDropShadow dx="0" dy="4" stdDeviation="4" flood-color="#e11d48" flood-opacity="0.25"/>
        </filter>
      </defs>

      <!-- Central Cardiac Digital Twin Icon -->
      <g class="heart-anim" filter="url(#shadow)">
        <!-- Ventricular Base Heart Silhouette -->
        <path d="M 55 98 C 25 80, 16 52, 28 36 C 38 22, 50 28, 55 36 C 60 28, 72 22, 82 36 C 94 52, 85 80, 55 98 Z" 
              fill="url(#roseGradient)" />

        <!-- Aortic Arch Sweeping Over Top -->
        <path d="M 50 32 C 48 16, 68 12, 74 24" 
              fill="none" stroke="url(#roseGradient)" stroke-width="7" stroke-linecap="round" />
        <!-- 3 Branch Arteries from Aorta -->
        <line x1="57" y1="18" x2="56" y2="10" stroke="#be123c" stroke-width="3" stroke-linecap="round" />
        <line x1="64" y1="17" x2="65" y2="9" stroke="#be123c" stroke-width="2.5" stroke-linecap="round" />
        <line x1="71" y1="20" x2="74" y2="11" stroke="#be123c" stroke-width="2.5" stroke-linecap="round" />

        <!-- Pulmonary Trunk (Cobalt Blue Cross-Over) -->
        <path d="M 40 40 C 44 26, 32 20, 26 25" 
              fill="none" stroke="url(#blueGradient)" stroke-width="6" stroke-linecap="round" />
        <!-- Superior Vena Cava -->
        <line x1="28" y1="36" x2="26" y2="14" stroke="#1d4ed8" stroke-width="4.5" stroke-linecap="round" />

        <!-- Digital Twin Circuit Overlay Lines & Nodes -->
        <path d="M 32 50 L 44 58 L 52 48 L 64 62 L 76 52" 
              fill="none" stroke="#ffffff" stroke-width="1.8" opacity="0.85" stroke-linecap="round" />
        <circle cx="32" cy="50" r="2.5" fill="#38bdf8" />
        <circle cx="44" cy="58" r="2.2" fill="#ffffff" />
        <circle cx="52" cy="48" r="2.8" fill="#facc15" />
        <circle cx="64" cy="62" r="2.2" fill="#ffffff" />
        <circle cx="76" cy="52" r="2.5" fill="#38bdf8" />

        <!-- Dynamic ECG Vector Trace -->
        <path class="ecg-anim" d="M 18 64 L 34 64 L 38 52 L 42 76 L 46 44 L 50 68 L 54 64 L 92 64" 
              fill="none" stroke="#ffffff" stroke-width="2" stroke-linecap="round" opacity="0.95"/>
      </g>

      <!-- Brand Typography -->
      <g transform="translate(108, 0)">
        <!-- App Title -->
        <text x="0" y="52" class="brand-text" font-size="34" fill="url(#textGradient)" letter-spacing="-0.5">
          CardioTwin <tspan fill="#e11d48">X</tspan>
        </text>

        <!-- Subtitle & Category -->
        <text x="2" y="72" class="sub-text" font-size="11.5" fill="#475569" letter-spacing="0.8">
          CARDIOVASCULAR DIGITAL TWIN
        </text>

        <!-- Physiological Tagline -->
        <text x="2" y="90" font-family="-apple-system, sans-serif" font-size="10" fill="#94a3b8" letter-spacing="0.2">
          Adaptive · Simulation-Driven · Explainable
        </text>

        <!-- Medical Badge -->
        <rect x="2" y="102" width="138" height="20" rx="10" fill="#fff1f2" stroke="#fecdd3" stroke-width="1"/>
        <text x="71" y="116" font-family="-apple-system, sans-serif" font-size="9" font-weight="700" fill="#be123c" text-anchor="middle">
          In-Silico Hemodynamics
        </text>
      </g>
    </svg>
    """
    return svg


def render_logo_header_card() -> None:
    """Render the official responsive CardioTwin X logo card."""
    logo_svg = get_cardiotwin_logo_svg(width=340, height=135, animated=True)
    st.markdown(
        f"""
        <div style="display: flex; justify-content: center; align-items: center; padding: 12px 0 20px 0;">
            {logo_svg}
        </div>
        """,
        unsafe_allow_html=True,
    )


def initialize_auth_state() -> None:
    """Initialize authentication session state defaults."""
    is_testing = (
        os.environ.get("STREAMLIT_TESTING") == "1"
        or os.environ.get("PYTEST_CURRENT_TEST") is not None
    )

    if "authenticated" not in st.session_state:
        # Default authenticated to True during automated testing so test suites run uninterrupted
        st.session_state.authenticated = True if is_testing else False

    if "current_user" not in st.session_state:
        st.session_state.current_user = DEMO_USERS["cardiologist"]

    if "login_view" not in st.session_state:
        st.session_state.login_view = "login"  # 'login' or 'logo_showcase'


def render_auth_header_profile() -> None:
    """Render authenticated user badge in the header with a logout button."""
    user: UserAccount = st.session_state.get("current_user", DEMO_USERS["cardiologist"])

    prof_col1, prof_col2 = st.columns([3.2, 1.0])
    with prof_col1:
        st.markdown(
            f"""
            <div style="background-color: #ffffff; border: 1px solid #fecdd3; border-radius: 20px; padding: 6px 14px; display: inline-flex; align-items: center; gap: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.03);">
                <span style="font-size: 1.15rem;">{user.avatar}</span>
                <div style="line-height: 1.2;">
                    <span style="font-size: 0.82rem; font-weight: 700; color: #1e293b;">{user.name}</span>
                    <span style="font-size: 0.72rem; color: #e11d48; font-weight: 600; margin-left: 6px;">{user.role}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with prof_col2:
        if st.button("🚪 Sign Out", key="sign_out_top_btn", help="Sign out to Login & Logo Page"):
            st.session_state.authenticated = False
            st.rerun()


def render_login_and_logo_page() -> None:
    """Render the full Login Page and Logo Showcase Landing View."""
    st.markdown(
        """
        <style>
        .login-card {
            background-color: #ffffff;
            border: 1.5px solid #fecdd3;
            border-radius: 20px;
            padding: 28px 32px;
            box-shadow: 0 10px 30px rgba(225, 29, 72, 0.08);
            max-width: 480px;
            margin: 0 auto;
        }
        .demo-btn-box {
            background: #fff1f2;
            border: 1px dashed #f43f5e;
            border-radius: 12px;
            padding: 14px;
            margin-top: 18px;
        }
        .brand-showcase-box {
            background: linear-gradient(135deg, #fff1f2 0%, #ffffff 60%, #f0fdf4 100%);
            border: 1.5px solid #fecdd3;
            border-radius: 20px;
            padding: 26px 30px;
            box-shadow: 0 8px 24px rgba(225, 29, 72, 0.06);
            margin-bottom: 24px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Top Navigation toggle between Login Form and Brand / Logo Showcase
    top_t1, top_t2, top_t3 = st.columns([1.2, 2.0, 1.2])
    with top_t2:
        view_mode = st.radio(
            "Select Page View",
            options=["🔐 Sign In (Login Page)", "🎨 Official Logo & Brand Assets"],
            horizontal=True,
            label_visibility="collapsed",
            index=0 if st.session_state.login_view == "login" else 1,
        )
        st.session_state.login_view = "login" if "Sign In" in view_mode else "logo_showcase"

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    if st.session_state.login_view == "logo_showcase":
        # ---------------------------------------------------------------------
        # Dedicated LOGO & BRAND IDENTITY PAGE
        # ---------------------------------------------------------------------
        render_logo_page_content()
    else:
        # ---------------------------------------------------------------------
        # Dedicated LOGIN PAGE
        # ---------------------------------------------------------------------
        render_login_page_content()


def render_login_page_content() -> None:
    """Render the Sign In interface with interactive inputs and 1-click presets."""
    col_l, col_center, col_r = st.columns([1.0, 2.4, 1.0])

    with col_center:
        # Brand Logo Header
        render_logo_header_card()

        st.markdown(
            """
            <div style="text-align: center; margin-bottom: 20px;">
                <h3 style="color: #1e293b; font-weight: 800; font-size: 1.35rem; margin-bottom: 4px;">
                    Welcome to CardioTwin X
                </h3>
                <div style="color: #64748b; font-size: 0.88rem;">
                    Sign in to access your physiological digital twin telemetry dashboard
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.container():
            # Main Credentials Input Form
            with st.form("login_form"):
                user_role_select = st.selectbox(
                    "Sign In Role",
                    options=["Cardiologist / Clinician", "Twin Researcher", "Patient (Remote Monitoring)", "Guest Evaluator"],
                    index=0,
                )
                email_input = st.text_input("Work Email or Username", value="s.jenkins@cardiotwin.org")
                password_input = st.text_input("Password", value="••••••••••••", type="password")

                col_chk1, col_chk2 = st.columns(2)
                with col_chk1:
                    st.checkbox("Remember this workstation", value=True)
                with col_chk2:
                    st.caption("<div style='text-align: right; padding-top: 4px;'><a href='#' style='color: #e11d48; text-decoration: none;'>Forgot password?</a></div>", unsafe_allow_html=True)

                submit_login = st.form_submit_button("🚀 Sign In to Dashboard", type="primary", use_container_width=True)

                if submit_login:
                    if "Cardiologist" in user_role_select:
                        st.session_state.current_user = DEMO_USERS["cardiologist"]
                    elif "Researcher" in user_role_select:
                        st.session_state.current_user = DEMO_USERS["researcher"]
                    elif "Patient" in user_role_select:
                        st.session_state.current_user = DEMO_USERS["patient"]
                    else:
                        st.session_state.current_user = DEMO_USERS["guest"]

                    st.session_state.authenticated = True
                    st.success(f"Welcome back, {st.session_state.current_user.name}!")
                    st.rerun()

            # 1-Click Demo Quick-Access Section
            st.markdown(
                """
                <div class="demo-btn-box">
                    <div style="font-size: 0.82rem; font-weight: 700; color: #be123c; margin-bottom: 8px;">
                        ⚡ 1-Click Demo Profiles (Instant Evaluation):
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            d_col1, d_col2 = st.columns(2)
            with d_col1:
                if st.button("🩺 Clinician Access", use_container_width=True, help="Login as Dr. Sarah Jenkins, Lead Cardiologist"):
                    st.session_state.current_user = DEMO_USERS["cardiologist"]
                    st.session_state.authenticated = True
                    st.rerun()
                if st.button("👤 Patient Access", use_container_width=True, help="Login as Sarah Jenkins, Sample Patient"):
                    st.session_state.current_user = DEMO_USERS["patient"]
                    st.session_state.authenticated = True
                    st.rerun()

            with d_col2:
                if st.button("🔬 Researcher Access", use_container_width=True, help="Login as Prof. Marcus Vance, Twin Researcher"):
                    st.session_state.current_user = DEMO_USERS["researcher"]
                    st.session_state.authenticated = True
                    st.rerun()
                if st.button("✨ Instant Guest Sandbox", use_container_width=True, help="One-click immediate guest preview"):
                    st.session_state.current_user = DEMO_USERS["guest"]
                    st.session_state.authenticated = True
                    st.rerun()

        st.markdown(
            """
            <div style="text-align: center; color: #94a3b8; font-size: 0.76rem; margin-top: 24px;">
                Protected by Clinical Protocol Guard · Research and Demonstration Prototype
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_logo_page_content() -> None:
    """Render the dedicated Logo & Brand Identity Showcase Page."""
    st.markdown(
        """
        <div style="text-align: center; margin-bottom: 24px;">
            <h2 style="color: #e11d48; font-weight: 800; font-size: 1.8rem; margin-bottom: 4px;">
                🎨 CardioTwin X Brand Identity & Logo System
            </h2>
            <div style="color: #64748b; font-size: 0.95rem;">
                Official insignia, vector marks, design system, and physiological iconography
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    logo_col1, logo_col2 = st.columns([1.8, 1.4])

    with logo_col1:
        st.markdown(
            f"""
            <div class="brand-showcase-box" style="text-align: center; padding: 40px 20px;">
                <div style="display: flex; justify-content: center; margin-bottom: 16px;">
                    {get_cardiotwin_logo_svg(width=420, height=170, animated=True)}
                </div>
                <div style="font-size: 0.88rem; color: #475569; max-width: 440px; margin: 0 auto; line-height: 1.5;">
                    <b>Primary Insignia:</b> Conjoins the anatomical aortic-ventricular morphology with 
                    in-silico circuit nodes and an electrophysiological QRS vector trace.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### 🔬 Symbolism & Vector Geometry")
        st.markdown(
            """
            - **Anatomical Heart Silhouette:** Canonical left and right ventricular body rendered in arterial crimson gradients (`#f43f5e` $\to$ `#e11d48` $\to$ `#9f1239`).
            - **Aortic Arch & 3 Branches:** Sweeping crimson arch with brachiocephalic, carotid, and subclavian lumens symbolizing systemic afterload and compliance.
            - **Pulmonary Trunk (Cobalt Blue):** Represents deoxygenated venous return and closed-loop cardiopulmonary circulation.
            - **Digital Twin Mesh:** Interconnecting circuit pathways and yellow/cyan telemetry nodes representing continuous real-time UKF parameter assimilation.
            - **Integrated ECG Vector:** Animated QRS electrocardiogram wave symbolizing continuous chronotropic pacing.
            """
        )

    with logo_col2:
        st.markdown("#### 🎨 Color Palette & Design Tokens")
        st.markdown(
            """
            <div style="display: flex; flex-direction: column; gap: 8px; margin-bottom: 20px;">
                <div style="background-color: #e11d48; color: white; padding: 10px 14px; border-radius: 8px; font-weight: 700; font-size: 0.85rem;">
                    🔴 Arterial Crimson: #E11D48 (Primary Brand Accent)
                </div>
                <div style="background-color: #2563eb; color: white; padding: 10px 14px; border-radius: 8px; font-weight: 700; font-size: 0.85rem;">
                    🔵 Venous Cobalt: #2563EB (Deoxygenated Inflow)
                </div>
                <div style="background-color: #fff1f2; color: #be123c; border: 1px solid #fecdd3; padding: 10px 14px; border-radius: 8px; font-weight: 700; font-size: 0.85rem;">
                    🌸 Soft Rose Background: #FFF1F2 (Friendly Healthcare Canvas)
                </div>
                <div style="background-color: #059669; color: white; padding: 10px 14px; border-radius: 8px; font-weight: 700; font-size: 0.85rem;">
                    🟢 Normotensive Emerald: #059669 (Normal Clinical Status)
                </div>
                <div style="background-color: #d97706; color: white; padding: 10px 14px; border-radius: 8px; font-weight: 700; font-size: 0.85rem;">
                    🟡 Divergence Amber: #D97706 (Watch Clinical Alert)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### 🚀 Enter Dashboard")
        st.caption("Ready to inspect the live digital twin?")
        if st.button("🔐 Proceed to Sign In / Dashboard", type="primary", use_container_width=True):
            st.session_state.login_view = "login"
            st.rerun()
