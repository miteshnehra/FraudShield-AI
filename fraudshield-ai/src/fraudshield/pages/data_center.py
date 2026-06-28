"""Interactive Data Center page."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import asdict

import pandas as pd
import plotly.express as px
import streamlit as st

from fraudshield.config import Settings
from fraudshield.data.cleaning import CleaningOptions, CleaningResult, clean_dataset
from fraudshield.data.loader import DatasetLoadError, DatasetMetadata, load_csv_bytes
from fraudshield.data.quality import (
    column_profile_table,
    missing_value_table,
    profile_dataset,
)
from fraudshield.state import (
    ACTIVE_DATA_KEY,
    CLEANING_RESULT_KEY,
    LOCAL_EXPLANATION_CONTEXT_KEY,
    LOCAL_EXPLANATION_KEY,
    METADATA_KEY,
    MODEL_CONFIG_KEY,
    MODEL_RESULT_KEY,
    ORIGINAL_DATA_KEY,
    REPORT_CONTEXT_KEY,
    REPORT_PDF_KEY,
    SCORING_RESULT_KEY,
)


def _clear_model_state() -> None:
    st.session_state.pop(MODEL_RESULT_KEY, None)
    st.session_state.pop(MODEL_CONFIG_KEY, None)
    st.session_state.pop(SCORING_RESULT_KEY, None)
    st.session_state.pop(LOCAL_EXPLANATION_KEY, None)
    st.session_state.pop(LOCAL_EXPLANATION_CONTEXT_KEY, None)
    st.session_state.pop(REPORT_PDF_KEY, None)
    st.session_state.pop(REPORT_CONTEXT_KEY, None)


def _store_loaded_dataset(frame: pd.DataFrame, metadata: DatasetMetadata) -> None:
    st.session_state[ORIGINAL_DATA_KEY] = frame.copy(deep=True)
    st.session_state[ACTIVE_DATA_KEY] = frame.copy(deep=True)
    st.session_state[METADATA_KEY] = metadata
    st.session_state.pop(CLEANING_RESULT_KEY, None)
    _clear_model_state()


def _format_bytes(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.1f} MB"


def _render_upload(settings: Settings, logger: logging.Logger) -> None:
    data_settings = settings.section("data")
    accepted = tuple(str(item).lower() for item in data_settings["accepted_extensions"])
    uploaded_file = st.file_uploader(
        "Transaction dataset",
        type=list(accepted),
        help=f"CSV only. Maximum size: {data_settings['max_upload_mb']} MB.",
    )
    st.caption(
        "The uploaded file stays in the current app session unless you download a cleaned copy."
    )

    if uploaded_file is None:
        return

    content = uploaded_file.getvalue()
    fingerprint = hashlib.sha256(content).hexdigest()
    previous = st.session_state.get(METADATA_KEY)
    if previous is not None and previous.fingerprint == fingerprint:
        return

    try:
        loaded = load_csv_bytes(
            content,
            uploaded_file.name,
            max_upload_mb=int(data_settings["max_upload_mb"]),
            accepted_extensions=accepted,
        )
    except DatasetLoadError as error:
        logger.warning("Dataset rejected: %s", error)
        st.error(str(error))
        return

    _store_loaded_dataset(loaded.frame, loaded.metadata)
    logger.info(
        "Loaded dataset %s with %s rows and %s columns",
        loaded.metadata.filename,
        loaded.metadata.row_count,
        loaded.metadata.column_count,
    )
    st.success("Dataset loaded and validated.")


def _render_summary(frame: pd.DataFrame, metadata: DatasetMetadata) -> None:
    report = profile_dataset(frame)
    columns = st.columns(5)
    columns[0].metric("Rows", f"{report.row_count:,}")
    columns[1].metric("Columns", f"{report.column_count:,}")
    columns[2].metric("Missing", f"{report.missing_percentage:.2f}%")
    columns[3].metric("Duplicates", f"{report.duplicate_rows:,}")
    columns[4].metric("Quality", f"{report.quality_score:.1f}/100", report.quality_band)

    st.caption(
        f"{metadata.filename} | {_format_bytes(metadata.size_bytes)} | "
        f"{metadata.encoding} | delimiter: {repr(metadata.delimiter)}"
    )


def _render_preview(frame: pd.DataFrame) -> None:
    preview_rows = st.slider("Preview rows", min_value=5, max_value=100, value=20, step=5)
    st.dataframe(frame.head(preview_rows), hide_index=True, use_container_width=True)

    st.markdown("#### Column schema")
    st.dataframe(column_profile_table(frame), hide_index=True, use_container_width=True)


def _render_quality(frame: pd.DataFrame) -> None:
    report = profile_dataset(frame)
    missing_table = missing_value_table(frame)

    left, right = st.columns([1.35, 1])
    with left:
        st.markdown("#### Missing values")
        missing_only = missing_table[missing_table["Missing"] > 0]
        if missing_only.empty:
            st.success("No missing values detected.")
        else:
            chart = px.bar(
                missing_only.head(25),
                x="Missing %",
                y="Column",
                orientation="h",
                color_discrete_sequence=["#fb7185"],
            )
            chart.update_layout(
                height=max(280, min(700, 34 * len(missing_only))),
                margin=dict(l=0, r=10, t=10, b=0),
                yaxis={"categoryorder": "total ascending"},
            )
            st.plotly_chart(chart, use_container_width=True)
    with right:
        st.markdown("#### Quality flags")
        flags = pd.DataFrame(
            [
                {"Check": "Missing cells", "Count": report.missing_cells},
                {"Check": "Duplicate rows", "Count": report.duplicate_rows},
                {"Check": "Empty columns", "Count": len(report.empty_columns)},
                {"Check": "Constant columns", "Count": len(report.constant_columns)},
                {"Check": "Formula-like cells", "Count": report.formula_like_cells},
            ]
        )
        st.dataframe(flags, hide_index=True, use_container_width=True)
        if report.formula_like_cells:
            st.warning(
                "Formula-like text was detected. Neutralize it before opening exported CSV files "
                "in spreadsheet software unless those values are expected."
            )

    st.markdown("#### Column quality report")
    st.dataframe(missing_table, hide_index=True, use_container_width=True)


def _options_form() -> CleaningOptions | None:
    with st.form("cleaning_options"):
        st.markdown("#### Cleaning rules")
        first, second = st.columns(2)
        with first:
            convert_empty = st.checkbox("Treat blank text as missing", value=True)
            trim_text = st.checkbox("Trim text whitespace", value=True)
            drop_rows = st.checkbox("Remove fully empty rows", value=True)
            drop_columns = st.checkbox("Remove fully empty columns", value=True)
        with second:
            remove_duplicates = st.checkbox("Remove duplicate rows", value=True)
            normalize_columns = st.checkbox("Normalize column names", value=False)
            neutralize_formulas = st.checkbox("Neutralize spreadsheet formulas", value=False)

        numeric_label = st.selectbox(
            "Numeric missing values",
            ("Keep missing", "Median", "Mean", "Zero"),
        )
        categorical_label = st.selectbox(
            "Text missing values",
            ("Keep missing", "Most frequent", "Unknown"),
        )
        submitted = st.form_submit_button("Run cleaning", type="primary")

    if not submitted:
        return None

    numeric_map = {"Keep missing": "none", "Median": "median", "Mean": "mean", "Zero": "zero"}
    categorical_map = {"Keep missing": "none", "Most frequent": "mode", "Unknown": "unknown"}
    return CleaningOptions(
        convert_empty_strings=convert_empty,
        trim_text=trim_text,
        drop_empty_rows=drop_rows,
        drop_empty_columns=drop_columns,
        remove_duplicates=remove_duplicates,
        normalize_column_names=normalize_columns,
        numeric_missing=numeric_map[numeric_label],
        categorical_missing=categorical_map[categorical_label],
        neutralize_formulas=neutralize_formulas,
    )


def _render_cleaning_result(result: CleaningResult) -> None:
    before = profile_dataset(st.session_state[ORIGINAL_DATA_KEY])
    after = profile_dataset(result.frame)

    st.markdown("#### Cleaning result")
    comparison = pd.DataFrame(
        [
            {"Metric": "Rows", "Before": before.row_count, "After": after.row_count},
            {"Metric": "Columns", "Before": before.column_count, "After": after.column_count},
            {
                "Metric": "Missing cells",
                "Before": before.missing_cells,
                "After": after.missing_cells,
            },
            {
                "Metric": "Duplicate rows",
                "Before": before.duplicate_rows,
                "After": after.duplicate_rows,
            },
            {
                "Metric": "Quality score",
                "Before": before.quality_score,
                "After": after.quality_score,
            },
        ]
    )
    st.dataframe(comparison, hide_index=True, use_container_width=True)

    action_table = pd.DataFrame(asdict(action) for action in result.actions)
    if not action_table.empty:
        action_table.columns = ["Action", "Affected", "Detail"]
        st.dataframe(action_table, hide_index=True, use_container_width=True)


def _render_cleaning(logger: logging.Logger) -> None:
    original = st.session_state[ORIGINAL_DATA_KEY]
    options = _options_form()
    if options is not None:
        result = clean_dataset(original, options)
        st.session_state[ACTIVE_DATA_KEY] = result.frame
        st.session_state[CLEANING_RESULT_KEY] = result
        _clear_model_state()
        logger.info(
            "Cleaned dataset from shape %s to %s",
            result.original_shape,
            result.cleaned_shape,
        )
        st.success("Cleaning completed. The original dataset is still available.")

    result = st.session_state.get(CLEANING_RESULT_KEY)
    if result is not None:
        _render_cleaning_result(result)

    active = st.session_state[ACTIVE_DATA_KEY]
    left, right = st.columns([1, 1])
    with left:
        if st.button("Restore original dataset", use_container_width=True):
            st.session_state[ACTIVE_DATA_KEY] = original.copy(deep=True)
            st.session_state.pop(CLEANING_RESULT_KEY, None)
            _clear_model_state()
            st.rerun()
    with right:
        st.download_button(
            "Download active CSV",
            data=active.to_csv(index=False).encode("utf-8"),
            file_name="fraudshield_cleaned.csv",
            mime="text/csv",
            use_container_width=True,
        )


def render_data_center(settings: Settings, logger: logging.Logger) -> None:
    """Render CSV intake, profiling, and cleaning workflows."""
    st.markdown('<div class="fs-kicker">Dataset Operations</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="fs-title">Data Center</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="fs-subtitle">Validate transaction data before analysis or model training.</p>',
        unsafe_allow_html=True,
    )

    _render_upload(settings, logger)
    frame = st.session_state.get(ACTIVE_DATA_KEY)
    metadata = st.session_state.get(METADATA_KEY)
    if frame is None or metadata is None:
        st.info("Upload a CSV transaction dataset to begin quality analysis.")
        return

    _render_summary(frame, metadata)
    preview_tab, quality_tab, cleaning_tab = st.tabs(("Preview", "Data quality", "Cleaning"))
    with preview_tab:
        _render_preview(frame)
    with quality_tab:
        _render_quality(frame)
    with cleaning_tab:
        _render_cleaning(logger)
