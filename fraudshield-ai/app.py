"""FraudShield AI Streamlit entry point."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIRECTORY = PROJECT_ROOT / "src"
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from fraudshield.config import ensure_runtime_directories, load_settings  # noqa: E402
from fraudshield.logging_config import configure_logging  # noqa: E402
from fraudshield.pages.data_center import render_data_center  # noqa: E402
from fraudshield.pages.fraud_analysis import render_fraud_analysis  # noqa: E402
from fraudshield.pages.model_lab import render_model_lab  # noqa: E402
from fraudshield.pages.report_center import render_report_center  # noqa: E402
from fraudshield.pages.risk_review import render_risk_review  # noqa: E402
from fraudshield.state import ACTIVE_DATA_KEY, MODEL_RESULT_KEY  # noqa: E402
from fraudshield.ui import inject_styles, render_brand  # noqa: E402


load_dotenv(PROJECT_ROOT / ".env")
settings = load_settings()
ensure_runtime_directories(settings)
logger = configure_logging(settings.path("logs"))

ui_settings = settings.section("ui")
st.set_page_config(
    page_title=ui_settings["page_title"],
    page_icon=":shield:",
    layout=ui_settings["layout"],
    initial_sidebar_state=ui_settings["sidebar_state"],
)
inject_styles()


def render_overview() -> None:
    """Render the current operations overview."""
    app_settings = settings.section("app")
    dataset_status = "Loaded" if ACTIVE_DATA_KEY in st.session_state else "Not loaded"
    model_status = "Trained" if MODEL_RESULT_KEY in st.session_state else "Not trained"
    st.markdown('<div class="fs-kicker">Risk Operations Console</div>', unsafe_allow_html=True)
    st.markdown(f'<h1 class="fs-title">{app_settings["name"]}</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="fs-subtitle">'
        f'{app_settings["tagline"]}. Foundation status and build progress.'
        "</p>",
        unsafe_allow_html=True,
    )

    metric_columns = st.columns(4)
    metric_columns[0].metric("Runtime", "Healthy")
    metric_columns[1].metric("Dataset", dataset_status)
    metric_columns[2].metric("Model", model_status)
    metric_columns[3].metric("Version", app_settings["version"])

    st.subheader("Build status")
    left, right = st.columns([1.4, 1])
    with left:
        phases = pd.DataFrame(
            [
                {
                    "Phase": "1. Foundation",
                    "Status": "Ready",
                    "Output": "App, config, logging, tests",
                },
                {
                    "Phase": "2. Data Center",
                    "Status": "Ready",
                    "Output": "Upload, validation, cleaning",
                },
                {
                    "Phase": "3. Analysis",
                    "Status": "Ready",
                    "Output": "EDA and fraud patterns",
                },
                {
                    "Phase": "4. Model Lab",
                    "Status": "Ready",
                    "Output": "Training and comparison",
                },
                {
                    "Phase": "5. Risk Review",
                    "Status": "Ready",
                    "Output": "Scores and explanations",
                },
                {"Phase": "6. Reports", "Status": "Ready", "Output": "PDF and deployment"},
            ]
        )
        st.dataframe(phases, hide_index=True, use_container_width=True)

    with right:
        st.markdown("#### Workspace")
        st.markdown(
            """
            <div class="fs-row"><span class="fs-row-label">Configuration</span>
            <span class="fs-ready">Ready</span></div>
            <div class="fs-row"><span class="fs-row-label">Runtime folders</span>
            <span class="fs-ready">Ready</span></div>
            <div class="fs-row"><span class="fs-row-label">Application logging</span>
            <span class="fs-ready">Ready</span></div>
            <div class="fs-row"><span class="fs-row-label">Data pipeline</span>
            <span class="fs-ready">Ready</span></div>
            <div class="fs-row"><span class="fs-row-label">Fraud analysis</span>
            <span class="fs-ready">Ready</span></div>
            <div class="fs-row"><span class="fs-row-label">Model training</span>
            <span class="fs-ready">Ready</span></div>
            <div class="fs-row"><span class="fs-row-label">Risk review</span>
            <span class="fs-ready">Ready</span></div>
            <div class="fs-row"><span class="fs-row-label">Executive reporting</span>
            <span class="fs-ready">Ready</span></div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="fs-disclaimer">
            Decision-support prototype. A fraud score must be reviewed with business rules
            and human oversight before blocking a real transaction.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_system_check() -> None:
    """Render configuration and path checks without exposing secrets."""
    st.markdown('<div class="fs-kicker">Diagnostics</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="fs-title">System check</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="fs-subtitle">Resolved runtime settings for the current installation.</p>',
        unsafe_allow_html=True,
    )

    checks = []
    for label, path_name in (
        ("Raw data", "raw_data"),
        ("Processed data", "processed_data"),
        ("Models", "models"),
        ("Reports", "reports"),
        ("Logs", "logs"),
    ):
        path = settings.path(path_name)
        checks.append(
            {
                "Component": label,
                "Status": "Ready" if path.exists() else "Missing",
                "Path": str(path),
            }
        )

    st.dataframe(pd.DataFrame(checks), hide_index=True, use_container_width=True)
    st.success("FraudShield AI v1.0 configuration and runtime paths loaded successfully.")


with st.sidebar:
    render_brand()
    page = st.radio(
        "Navigation",
        (
            "Overview",
            "Data Center",
            "Fraud Analysis",
            "Model Lab",
            "Risk Review",
            "Report Center",
            "System check",
        ),
        label_visibility="collapsed",
    )
    st.caption("FraudShield AI v1.0")

logger.info("Rendering page: %s", page)
if page == "Overview":
    render_overview()
elif page == "Data Center":
    render_data_center(settings, logger)
elif page == "Fraud Analysis":
    render_fraud_analysis(logger)
elif page == "Model Lab":
    render_model_lab(settings, logger)
elif page == "Risk Review":
    render_risk_review(settings, logger)
elif page == "Report Center":
    render_report_center(settings, logger)
else:
    render_system_check()
