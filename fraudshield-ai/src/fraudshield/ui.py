"""Shared Streamlit presentation helpers."""

from __future__ import annotations

import streamlit as st


def inject_styles() -> None:
    """Apply the compact FraudShield visual system."""
    st.markdown(
        """
        <style>
        :root {
            --fs-bg: #090b10;
            --fs-panel: #11151b;
            --fs-border: #27303a;
            --fs-text: #f4f7fa;
            --fs-muted: #9aa7b4;
            --fs-cyan: #22d3ee;
            --fs-red: #fb7185;
            --fs-green: #34d399;
        }
        .stApp { background: var(--fs-bg); color: var(--fs-text); }
        [data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid var(--fs-border); }
        [data-testid="stMetric"] {
            background: var(--fs-panel);
            border: 1px solid var(--fs-border);
            border-radius: 6px;
            padding: 14px 16px;
        }
        [data-testid="stMetricLabel"] { color: var(--fs-muted); }
        .fs-kicker {
            color: var(--fs-cyan);
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }
        .fs-title { font-size: 2rem; font-weight: 720; margin: 0; }
        .fs-subtitle { color: var(--fs-muted); margin: 0.3rem 0 1.5rem; }
        .fs-brand {
            align-items: center;
            display: flex;
            gap: 0.65rem;
            margin: 0.2rem 0 1.25rem;
        }
        .fs-mark {
            align-items: center;
            background: var(--fs-cyan);
            border-radius: 6px;
            color: #061017;
            display: inline-flex;
            font-weight: 900;
            height: 34px;
            justify-content: center;
            width: 34px;
        }
        .fs-brand-name { font-size: 1rem; font-weight: 750; }
        .fs-badge {
            border: 1px solid #1f6f7b;
            border-radius: 999px;
            color: var(--fs-cyan);
            display: inline-block;
            font-size: 0.72rem;
            padding: 0.2rem 0.55rem;
        }
        .fs-row {
            align-items: center;
            border-bottom: 1px solid var(--fs-border);
            display: flex;
            justify-content: space-between;
            padding: 0.8rem 0;
        }
        .fs-row:last-child { border-bottom: none; }
        .fs-row-label { color: var(--fs-muted); }
        .fs-ready { color: var(--fs-green); font-weight: 650; }
        .fs-pending { color: #fbbf24; font-weight: 650; }
        .fs-disclaimer {
            border-left: 3px solid var(--fs-red);
            color: var(--fs-muted);
            margin-top: 1.5rem;
            padding: 0.45rem 0 0.45rem 0.9rem;
        }
        div[data-testid="stStatusWidget"] { visibility: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_brand() -> None:
    """Render the compact sidebar brand lockup."""
    st.markdown(
        """
        <div class="fs-brand">
            <span class="fs-mark">FS</span>
            <div>
                <div class="fs-brand-name">FraudShield AI</div>
                <span class="fs-badge">DEMO MODE</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

