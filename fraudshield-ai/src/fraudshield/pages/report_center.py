"""Executive PDF report generation and export page."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import pandas as pd
import streamlit as st

from fraudshield.config import Settings
from fraudshield.data.quality import profile_dataset
from fraudshield.reporting.pdf_report import build_executive_report
from fraudshield.state import (
    ACTIVE_DATA_KEY,
    METADATA_KEY,
    MODEL_RESULT_KEY,
    REPORT_CONTEXT_KEY,
    REPORT_PDF_KEY,
    SCORING_RESULT_KEY,
)


def _report_context(frame: pd.DataFrame) -> tuple[object, ...]:
    metadata = st.session_state.get(METADATA_KEY)
    training = st.session_state.get(MODEL_RESULT_KEY)
    scoring = st.session_state.get(SCORING_RESULT_KEY)
    return (
        getattr(metadata, "fingerprint", None),
        getattr(training, "data_signature", None),
        getattr(scoring, "model_name", None),
        getattr(scoring, "decision_threshold", None),
        len(frame),
        len(frame.columns),
    )


def _section_status() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Section": "Dataset quality", "Status": "Included"},
            {"Section": "Labeled fraud profile", "Status": "Included when target is binary"},
            {
                "Section": "Model evaluation",
                "Status": (
                    "Included" if MODEL_RESULT_KEY in st.session_state else "Train a model first"
                ),
            },
            {
                "Section": "Investigation queue",
                "Status": (
                    "Included" if SCORING_RESULT_KEY in st.session_state else "Score data first"
                ),
            },
            {"Section": "Governance and limitations", "Status": "Included"},
            {"Section": "Production readiness", "Status": "Included"},
        ]
    )


def render_report_center(settings: Settings, logger: logging.Logger) -> None:
    """Render report readiness, PDF generation, download, and local save."""
    st.markdown('<div class="fs-kicker">Executive Reporting</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="fs-title">Report Center</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="fs-subtitle">Generate a privacy-conscious decision-support summary.</p>',
        unsafe_allow_html=True,
    )

    frame = st.session_state.get(ACTIVE_DATA_KEY)
    if frame is None:
        st.info("Load a dataset in Data Center before generating a report.")
        return

    quality = profile_dataset(frame)
    metadata = st.session_state.get(METADATA_KEY)
    training = st.session_state.get(MODEL_RESULT_KEY)
    scoring = st.session_state.get(SCORING_RESULT_KEY)
    columns = st.columns(4)
    columns[0].metric("Dataset", "Ready")
    columns[1].metric("Quality", f"{quality.quality_score:.1f}/100")
    columns[2].metric("Model section", "Ready" if training is not None else "Not available")
    columns[3].metric("Risk section", "Ready" if scoring is not None else "Not available")

    st.markdown("#### Report contents")
    st.dataframe(_section_status(), hide_index=True, use_container_width=True)
    st.info(
        "The PDF contains summary metrics and queue identifiers only. "
        "Raw transaction fields are not embedded."
    )

    current_context = _report_context(frame)
    if st.button("Generate executive PDF", type="primary", use_container_width=True):
        try:
            with st.spinner("Building executive report..."):
                pdf_bytes = build_executive_report(
                    frame,
                    app_version=str(settings.section("app")["version"]),
                    dataset_name=getattr(metadata, "filename", "In-session transaction dataset"),
                    training_result=training,
                    scoring_result=scoring,
                )
        except (ValueError, KeyError) as error:
            logger.warning("Report generation rejected: %s", error)
            st.error(str(error))
        else:
            st.session_state[REPORT_PDF_KEY] = pdf_bytes
            st.session_state[REPORT_CONTEXT_KEY] = current_context
            logger.info("Generated executive PDF with %s bytes", len(pdf_bytes))
            st.success("Executive report generated.")

    pdf_bytes = st.session_state.get(REPORT_PDF_KEY)
    report_context = st.session_state.get(REPORT_CONTEXT_KEY)
    if pdf_bytes is None:
        return
    if report_context != current_context:
        st.warning("Dataset, model, or scores changed. Generate the report again.")
        return

    date_stamp = datetime.now(UTC).strftime("%Y%m%d")
    filename = f"fraudshield_executive_report_{date_stamp}.pdf"
    left, right = st.columns(2)
    with left:
        st.download_button(
            "Download executive PDF",
            data=pdf_bytes,
            file_name=filename,
            mime="application/pdf",
            use_container_width=True,
        )
    with right:
        if st.button("Save to project reports folder", use_container_width=True):
            destination = settings.path("reports") / filename
            destination.write_bytes(pdf_bytes)
            logger.info("Saved executive report to %s", destination)
            st.success(f"Saved: {destination.name}")
