"""Operational transaction risk queue and local explanation dashboard."""

from __future__ import annotations

import logging

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from fraudshield.config import Settings
from fraudshield.risk.explanations import (
    ExplanationUnavailableError,
    explain_transaction,
    human_readable_reasons,
)
from fraudshield.risk.scoring import (
    RiskThresholds,
    risk_band_counts,
    risk_summary,
    score_transactions,
)
from fraudshield.state import (
    ACTIVE_DATA_KEY,
    LOCAL_EXPLANATION_CONTEXT_KEY,
    LOCAL_EXPLANATION_KEY,
    MODEL_CONFIG_KEY,
    MODEL_RESULT_KEY,
    REPORT_CONTEXT_KEY,
    REPORT_PDF_KEY,
    SCORING_RESULT_KEY,
)


def _plot(figure: go.Figure) -> None:
    figure.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#f4f7fa"},
        margin=dict(l=10, r=10, t=35, b=10),
        hoverlabel={"bgcolor": "#11151b", "font_color": "#f4f7fa"},
    )
    st.plotly_chart(
        figure,
        use_container_width=True,
        config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
    )


def _risk_thresholds(settings: Settings) -> RiskThresholds:
    return RiskThresholds.from_mapping(settings.section("risk"))


def _score_controls(
    frame: pd.DataFrame,
    settings: Settings,
    logger: logging.Logger,
) -> tuple[object, str, float] | None:
    result = st.session_state.get(MODEL_RESULT_KEY)
    if result is None:
        st.info("Train at least one model in Model Lab before scoring transactions.")
        return None

    model_names = list(result.runs)
    selected_model = st.selectbox(
        "Scoring model",
        model_names,
        index=model_names.index(result.best_model_name),
        format_func=lambda name: result.runs[name].display_name,
    )
    training_config = st.session_state.get(MODEL_CONFIG_KEY)
    default_threshold = (
        float(training_config.threshold)
        if training_config is not None
        else float(settings.section("model")["default_threshold"])
    )
    threshold = st.slider(
        "Manual-review threshold",
        min_value=0.05,
        max_value=0.95,
        value=default_threshold,
        step=0.05,
        help="A score at or above this probability enters the manual-review queue.",
    )
    thresholds = _risk_thresholds(settings)
    st.caption(
        f"Risk bands: Low 0-{thresholds.low_max} | "
        f"Medium {thresholds.low_max + 1}-{thresholds.medium_max} | "
        f"High {thresholds.medium_max + 1}-{thresholds.high_max} | "
        f"Critical {thresholds.high_max + 1}-100"
    )

    if st.button("Score active dataset", type="primary", use_container_width=True):
        try:
            with st.spinner("Calculating transaction probabilities and risk bands..."):
                scoring = score_transactions(
                    frame,
                    result,
                    selected_model,
                    threshold,
                    thresholds,
                )
        except (ValueError, KeyError) as error:
            logger.warning("Risk scoring rejected: %s", error)
            st.error(str(error))
            return None
        st.session_state[SCORING_RESULT_KEY] = scoring
        st.session_state.pop(LOCAL_EXPLANATION_KEY, None)
        st.session_state.pop(LOCAL_EXPLANATION_CONTEXT_KEY, None)
        st.session_state.pop(REPORT_PDF_KEY, None)
        st.session_state.pop(REPORT_CONTEXT_KEY, None)
        logger.info(
            "Scored %s transactions using model=%s threshold=%.2f",
            len(frame),
            selected_model,
            threshold,
        )
        st.success("Risk queue updated.")

    scoring = st.session_state.get(SCORING_RESULT_KEY)
    if scoring is None:
        st.info("Choose a model and score the active dataset to create the risk queue.")
        return None
    if scoring.model_name != selected_model or scoring.decision_threshold != threshold:
        st.warning("The model or threshold changed. Score the dataset again to refresh the queue.")
        return None
    return scoring, selected_model, threshold


def _render_queue_summary(scored: pd.DataFrame) -> None:
    review_count = int((scored["FS_Decision"] == "Manual review").sum())
    critical_count = int((scored["FS_Risk_Band"] == "Critical").sum())
    elevated_count = int(scored["FS_Risk_Band"].isin(["High", "Critical"]).sum())
    columns = st.columns(5)
    columns[0].metric("Scored", f"{len(scored):,}")
    columns[1].metric("Average risk", f"{scored['FS_Risk_Score'].mean():.1f}/100")
    columns[2].metric("Manual review", f"{review_count:,}")
    columns[3].metric("High + Critical", f"{elevated_count:,}")
    columns[4].metric("Critical", f"{critical_count:,}")

    counts = risk_band_counts(scored)
    figure = px.bar(
        counts,
        x="Risk band",
        y="Transactions",
        color="Risk band",
        color_discrete_map={
            "Low": "#34d399",
            "Medium": "#fbbf24",
            "High": "#fb923c",
            "Critical": "#fb7185",
        },
        text="Transactions",
        category_orders={"Risk band": ["Low", "Medium", "High", "Critical"]},
    )
    figure.update_layout(showlegend=False, title="Risk queue distribution", height=330)
    _plot(figure)


def _filtered_queue(scored: pd.DataFrame, *, key_prefix: str) -> pd.DataFrame:
    first, second = st.columns([1.3, 1])
    with first:
        bands = st.multiselect(
            "Risk bands",
            ["Low", "Medium", "High", "Critical"],
            default=["High", "Critical"],
            key=f"{key_prefix}_risk_bands",
        )
    with second:
        minimum_score = st.slider(
            "Minimum risk score",
            0,
            100,
            0,
            key=f"{key_prefix}_minimum_score",
        )

    if not bands:
        return scored.iloc[0:0].copy()
    filtered = scored[
        scored["FS_Risk_Band"].isin(bands) & (scored["FS_Risk_Score"] >= minimum_score)
    ]
    return filtered.sort_values("FS_Risk_Score", ascending=False)


def _render_queue(scored: pd.DataFrame) -> pd.DataFrame:
    st.markdown("#### Investigation queue")
    filtered = _filtered_queue(scored, key_prefix="queue")
    preferred = [
        "FS_Row_Number",
        "FS_Source_Index",
        "FS_Risk_Score",
        "FS_Risk_Band",
        "FS_Fraud_Probability",
        "FS_Decision",
    ]
    remaining = [column for column in scored.columns if column not in preferred + ["FS_Model"]]
    display_columns = preferred + remaining[:8]
    st.dataframe(
        filtered.loc[:, display_columns].head(500),
        hide_index=True,
        use_container_width=True,
    )
    st.caption(f"Showing up to 500 of {len(filtered):,} filtered transactions.")
    st.download_button(
        "Download complete scored CSV",
        data=scored.to_csv(index=False).encode("utf-8"),
        file_name="fraudshield_scored_transactions.csv",
        mime="text/csv",
    )
    return filtered


def _row_label(scored: pd.DataFrame, row_number: int) -> str:
    row = scored.loc[scored["FS_Row_Number"] == row_number].iloc[0]
    identifier_columns = [
        column
        for column in scored.columns
        if column.lower().endswith("_id") and not column.startswith("FS_")
    ]
    identifier = str(row[identifier_columns[0]]) if identifier_columns else row["FS_Source_Index"]
    return f"{identifier} | {row['FS_Risk_Band']} | {row['FS_Risk_Score']:.2f}/100"


def _render_explanation(
    source_frame: pd.DataFrame,
    scored: pd.DataFrame,
    filtered: pd.DataFrame,
    result: object,
    model_name: str,
    logger: logging.Logger,
) -> None:
    st.markdown("#### Transaction explanation")
    choices = filtered["FS_Row_Number"].astype(int).tolist()
    if not choices:
        st.info("No transactions match the current queue filters.")
        return
    row_number = st.selectbox(
        "Transaction",
        choices,
        format_func=lambda value: _row_label(scored, value),
    )
    row = scored.loc[scored["FS_Row_Number"] == row_number].iloc[0]
    columns = st.columns(4)
    columns[0].metric("Risk score", f"{row['FS_Risk_Score']:.2f}/100")
    columns[1].metric("Risk band", row["FS_Risk_Band"])
    columns[2].metric("Probability", f"{row['FS_Fraud_Probability']:.2%}")
    columns[3].metric("Decision", row["FS_Decision"])
    st.info(risk_summary(str(row["FS_Risk_Band"])))

    context = (model_name, int(row_number), result.data_signature)
    if st.button("Generate local explanation", use_container_width=True):
        try:
            with st.spinner("Calculating signed local feature contributions..."):
                explanation = explain_transaction(
                    source_frame,
                    result,
                    model_name,
                    int(row_number),
                )
        except (ExplanationUnavailableError, ValueError, KeyError, IndexError) as error:
            logger.warning("Local explanation unavailable: %s", error)
            st.error(str(error))
        else:
            st.session_state[LOCAL_EXPLANATION_KEY] = explanation
            st.session_state[LOCAL_EXPLANATION_CONTEXT_KEY] = context

    explanation = st.session_state.get(LOCAL_EXPLANATION_KEY)
    explanation_context = st.session_state.get(LOCAL_EXPLANATION_CONTEXT_KEY)
    if explanation is None or explanation_context != context:
        return

    st.caption(f"Explanation method: {explanation.method}")
    for reason in human_readable_reasons(explanation):
        st.markdown(f"- {reason}")

    contributions = explanation.contributions.head(15).sort_values("Contribution")
    colors = ["#22d3ee" if value < 0 else "#fb7185" for value in contributions["Contribution"]]
    figure = go.Figure(
        go.Bar(
            x=contributions["Contribution"],
            y=contributions["Readable feature"],
            orientation="h",
            marker_color=colors,
            customdata=contributions[["Direction"]],
            hovertemplate=(
                "%{y}<br>Contribution=%{x:.4f}<br>%{customdata[0]}<extra></extra>"
            ),
        )
    )
    figure.add_vline(x=0, line_color="#637080")
    figure.update_layout(title="Local signed feature contributions", height=470)
    _plot(figure)

    with st.expander("Source transaction values"):
        source_row = source_frame.iloc[int(row_number)]
        source_table = pd.DataFrame(
            {"Field": source_row.index.astype(str), "Value": source_row.astype(str).to_numpy()}
        )
        st.dataframe(source_table, hide_index=True, use_container_width=True)


def render_risk_review(settings: Settings, logger: logging.Logger) -> None:
    """Render batch scoring, investigation queue, and local explanations."""
    st.markdown('<div class="fs-kicker">Decision Support</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="fs-title">Risk Review</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="fs-subtitle">Prioritize transactions for human investigation.</p>',
        unsafe_allow_html=True,
    )

    frame = st.session_state.get(ACTIVE_DATA_KEY)
    if frame is None:
        st.info("Load a dataset in Data Center before opening Risk Review.")
        return

    controls = _score_controls(frame, settings, logger)
    if controls is None:
        return
    scoring, model_name, _ = controls
    scored = scoring.frame
    result = st.session_state[MODEL_RESULT_KEY]

    st.warning(
        "These are operational scores for the active dataset, not validation metrics. "
        "Do not report them as model accuracy or automatically block transactions."
    )
    _render_queue_summary(scored)
    queue_tab, explanation_tab = st.tabs(("Risk queue", "Transaction detail"))
    with queue_tab:
        _render_queue(scored)
    with explanation_tab:
        filtered = _filtered_queue(scored, key_prefix="detail")
        _render_explanation(frame, scored, filtered, result, model_name, logger)
