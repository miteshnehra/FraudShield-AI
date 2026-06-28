"""Interactive fraud-focused exploratory analysis dashboard."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from fraudshield.analysis.eda import (
    amount_distribution_frame,
    categorical_fraud_rates,
    class_distribution,
    correlation_matrix,
    hourly_fraud_pattern,
    infer_amount_columns,
    infer_binary_target_columns,
    infer_categorical_columns,
    infer_datetime_columns,
    infer_fraud_value,
    profile_target,
    strongest_fraud_correlations,
    target_values,
    time_fraud_trend,
)
from fraudshield.state import ACTIVE_DATA_KEY


@dataclass(frozen=True)
class AnalysisSelection:
    target_column: str
    fraud_value: Any
    amount_column: str | None
    datetime_column: str | None
    category_columns: tuple[str, ...]


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


def _optional_choice(label: str, columns: list[str], *, key: str) -> str | None:
    options: list[str | None] = [None, *columns]
    return st.selectbox(
        label,
        options,
        index=1 if columns else 0,
        key=key,
        format_func=lambda value: "Not selected" if value is None else value,
    )


def _analysis_selection(frame: pd.DataFrame) -> AnalysisSelection | None:
    targets = infer_binary_target_columns(frame)
    if not targets:
        st.error(
            "No binary target column was found. Fraud analysis needs a label column with "
            "exactly two values, such as 0/1 or safe/fraud."
        )
        return None

    with st.expander("Analysis configuration", expanded=True):
        first, second, third = st.columns(3)
        with first:
            target_column = st.selectbox("Fraud target", targets)
            values = target_values(frame, target_column)
            likely_fraud = infer_fraud_value(values)
            fraud_index = values.index(likely_fraud)
            fraud_value = st.selectbox(
                "Fraud label",
                values,
                index=fraud_index,
                format_func=str,
            )
        with second:
            amount_columns = infer_amount_columns(frame, exclude=(target_column,))
            amount_column = _optional_choice(
                "Transaction amount",
                amount_columns,
                key="eda_amount_column",
            )
        with third:
            datetime_column = _optional_choice(
                "Transaction time",
                infer_datetime_columns(frame),
                key="eda_datetime_column",
            )

        categories = infer_categorical_columns(
            frame,
            exclude=tuple(
                item for item in (target_column, datetime_column) if item is not None
            ),
        )
        st.caption(
            "Selections affect charts only. The active dataset is never modified by analysis."
        )

    return AnalysisSelection(
        target_column=target_column,
        fraud_value=fraud_value,
        amount_column=amount_column,
        datetime_column=datetime_column,
        category_columns=tuple(categories),
    )


def _render_metrics(frame: pd.DataFrame, selection: AnalysisSelection) -> None:
    profile = profile_target(
        frame,
        selection.target_column,
        selection.fraud_value,
        selection.amount_column,
    )
    columns = st.columns(5)
    columns[0].metric("Valid transactions", f"{profile.valid_transactions:,}")
    columns[1].metric("Fraud transactions", f"{profile.fraud_transactions:,}")
    columns[2].metric("Fraud rate", f"{profile.fraud_rate:.2f}%")
    if profile.fraud_amount is None:
        columns[3].metric("Fraud amount", "Not selected")
        columns[4].metric("Amount exposure", "Not selected")
    else:
        columns[3].metric("Fraud amount", f"{profile.fraud_amount:,.2f}")
        exposure = profile.fraud_amount_rate
        columns[4].metric(
            "Amount exposure",
            f"{exposure:.2f}%" if exposure is not None else "Unavailable",
        )

    if profile.missing_target:
        st.warning(f"{profile.missing_target:,} rows have no target label and are excluded.")
    if 0 < profile.fraud_rate < 5:
        st.warning(
            "Strong class imbalance detected. Accuracy will be misleading in model evaluation; "
            "Phase 4 will prioritize precision, recall, F1, PR-AUC, and ROC-AUC."
        )
    elif profile.fraud_rate > 50:
        st.warning(
            "Fraud exceeds 50% of labeled rows. Confirm that the fraud label is selected correctly."
        )
    if profile.fraud_transactions < 10:
        st.info(
            "Fewer than 10 fraud rows are available. Treat pattern percentages as exploratory only."
        )


def _render_overview(frame: pd.DataFrame, selection: AnalysisSelection) -> None:
    left, right = st.columns([0.85, 1.25])
    with left:
        st.markdown("#### Class balance")
        distribution = class_distribution(
            frame,
            selection.target_column,
            selection.fraud_value,
        )
        figure = px.pie(
            distribution,
            values="Transactions",
            names="Class",
            hole=0.58,
            color="Class",
            color_discrete_map={"Legitimate": "#34d399", "Fraud": "#fb7185"},
        )
        figure.update_traces(textinfo="label+percent", sort=False)
        _plot(figure)

    with right:
        st.markdown("#### Category risk")
        if not selection.category_columns:
            st.info("No low-cardinality category columns were detected.")
        else:
            category = st.selectbox(
                "Compare by",
                selection.category_columns,
                key="eda_category_column",
            )
            minimum = st.number_input(
                "Minimum transactions per category",
                min_value=1,
                max_value=max(len(frame), 1),
                value=1,
                step=1,
            )
            rates = categorical_fraud_rates(
                frame,
                selection.target_column,
                selection.fraud_value,
                category,
                min_transactions=int(minimum),
            )
            if rates.empty:
                st.info("No categories meet the selected transaction threshold.")
            else:
                figure = px.bar(
                    rates.sort_values("Fraud rate %"),
                    x="Fraud rate %",
                    y="Category",
                    orientation="h",
                    color="Fraud rate %",
                    color_continuous_scale=["#34d399", "#fbbf24", "#fb7185"],
                    hover_data=["Transactions", "Fraud", "Legitimate"],
                )
                figure.update_layout(coloraxis_showscale=False, height=max(320, len(rates) * 32))
                _plot(figure)

    fraud_mask = frame[selection.target_column].eq(selection.fraud_value).fillna(False)
    with st.expander("Fraud-labeled transaction sample"):
        fraud_rows = frame.loc[fraud_mask]
        if fraud_rows.empty:
            st.info("No rows match the selected fraud label.")
        else:
            st.dataframe(fraud_rows.head(100), hide_index=True, use_container_width=True)
            if len(fraud_rows) > 100:
                st.caption(f"Showing 100 of {len(fraud_rows):,} fraud-labeled rows.")


def _render_amount(frame: pd.DataFrame, selection: AnalysisSelection) -> None:
    if selection.amount_column is None:
        st.info("Select a transaction amount column in Analysis configuration.")
        return

    distribution = amount_distribution_frame(
        frame,
        selection.target_column,
        selection.fraud_value,
        selection.amount_column,
    )
    if distribution.empty:
        st.warning("The selected amount column contains no usable numeric values.")
        return

    summary = (
        distribution.groupby("Class", observed=True)["Amount"]
        .agg(Transactions="count", Mean="mean", Median="median", Minimum="min", Maximum="max")
        .round(2)
        .reset_index()
    )
    st.dataframe(summary, hide_index=True, use_container_width=True)

    bins = st.slider("Histogram bins", min_value=10, max_value=100, value=40, step=5)
    left, right = st.columns(2)
    with left:
        figure = px.histogram(
            distribution,
            x="Amount",
            color="Class",
            nbins=bins,
            barmode="overlay",
            opacity=0.68,
            color_discrete_map={"Legitimate": "#34d399", "Fraud": "#fb7185"},
        )
        figure.update_layout(title="Transaction amount distribution")
        _plot(figure)
    with right:
        figure = px.box(
            distribution,
            x="Class",
            y="Amount",
            color="Class",
            points=False,
            color_discrete_map={"Legitimate": "#34d399", "Fraud": "#fb7185"},
        )
        figure.update_layout(title="Amount spread by class", showlegend=False)
        _plot(figure)

    st.caption(
        "Large datasets are stratified to at most 50,000 rows for chart performance; "
        "summary metrics use the plotted sample."
    )


def _render_time(frame: pd.DataFrame, selection: AnalysisSelection) -> None:
    if selection.datetime_column is None:
        st.info("Select a transaction time column in Analysis configuration.")
        return

    period = st.selectbox("Timeline granularity", ("hour", "day", "week", "month"), index=1)
    trend = time_fraud_trend(
        frame,
        selection.target_column,
        selection.fraud_value,
        selection.datetime_column,
        period=period,
    )
    if trend.empty:
        st.warning("The selected time column contains no usable timestamps.")
        return

    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(
        go.Bar(
            x=trend["Period"],
            y=trend["Transactions"],
            name="Transactions",
            marker_color="#27303a",
        ),
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=trend["Period"],
            y=trend["Fraud rate %"],
            name="Fraud rate %",
            line={"color": "#fb7185", "width": 2.5},
            mode="lines+markers",
        ),
        secondary_y=True,
    )
    figure.update_yaxes(title_text="Transactions", secondary_y=False)
    figure.update_yaxes(title_text="Fraud rate %", secondary_y=True)
    figure.update_layout(title="Fraud rate and transaction volume over time")
    _plot(figure)

    st.markdown("#### Hour-of-day pattern")
    hourly = hourly_fraud_pattern(
        frame,
        selection.target_column,
        selection.fraud_value,
        selection.datetime_column,
    )
    if hourly.empty:
        st.info("No usable hour-of-day pattern is available.")
        return
    figure = px.bar(
        hourly,
        x="Hour",
        y="Fraud rate %",
        color="Fraud rate %",
        color_continuous_scale=["#34d399", "#fbbf24", "#fb7185"],
        hover_data=["Transactions", "Fraud"],
    )
    figure.update_layout(coloraxis_showscale=False)
    _plot(figure)


def _render_correlations(frame: pd.DataFrame, selection: AnalysisSelection) -> None:
    matrix = correlation_matrix(frame, selection.target_column, selection.fraud_value)
    if matrix.empty:
        st.info("At least one usable numeric feature is required for correlation analysis.")
        return

    st.caption(
        "Spearman correlation measures monotonic association, not causation. "
        "Fraud indicator is always oriented so 1 means the selected fraud label."
    )
    left, right = st.columns([1.45, 0.85])
    with left:
        figure = go.Figure(
            data=go.Heatmap(
                z=matrix.to_numpy(),
                x=matrix.columns,
                y=matrix.index,
                zmin=-1,
                zmax=1,
                colorscale=[
                    [0.0, "#22d3ee"],
                    [0.5, "#11151b"],
                    [1.0, "#fb7185"],
                ],
                colorbar={"title": "rho"},
                hovertemplate="%{y} vs %{x}<br>rho=%{z:.3f}<extra></extra>",
            )
        )
        figure.update_layout(title="Feature correlation matrix", height=max(430, len(matrix) * 28))
        _plot(figure)
    with right:
        strongest = strongest_fraud_correlations(matrix)
        st.markdown("#### Strongest fraud associations")
        st.dataframe(
            strongest[["Feature", "Correlation"]].round(4),
            hide_index=True,
            use_container_width=True,
        )
        if not strongest.empty:
            figure = px.bar(
                strongest.sort_values("Correlation"),
                x="Correlation",
                y="Feature",
                orientation="h",
                color="Correlation",
                color_continuous_scale=["#22d3ee", "#27303a", "#fb7185"],
                range_color=(-1, 1),
            )
            figure.update_layout(coloraxis_showscale=False)
            _plot(figure)


def render_fraud_analysis(logger: logging.Logger) -> None:
    """Render interactive EDA for the active transaction dataset."""
    st.markdown('<div class="fs-kicker">Pattern Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="fs-title">Fraud Analysis</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="fs-subtitle">Explore labeled fraud behavior before building a model.</p>',
        unsafe_allow_html=True,
    )

    frame = st.session_state.get(ACTIVE_DATA_KEY)
    if frame is None:
        st.info("Load a dataset in Data Center before opening fraud analysis.")
        return

    selection = _analysis_selection(frame)
    if selection is None:
        return

    logger.info(
        "Rendering fraud analysis using target=%s fraud_value=%s",
        selection.target_column,
        selection.fraud_value,
    )
    _render_metrics(frame, selection)
    overview, amount, time, correlations = st.tabs(
        ("Overview", "Amount", "Time", "Correlations")
    )
    with overview:
        _render_overview(frame, selection)
    with amount:
        _render_amount(frame, selection)
    with time:
        _render_time(frame, selection)
    with correlations:
        _render_correlations(frame, selection)
