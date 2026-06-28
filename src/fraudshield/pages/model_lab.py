"""Interactive leakage-safe model training and comparison page."""

from __future__ import annotations

import logging

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fraudshield.analysis.eda import (
    infer_binary_target_columns,
    infer_datetime_columns,
    infer_fraud_value,
    target_values,
)
from fraudshield.config import Settings
from fraudshield.modeling.schema import feature_schema_table, suggest_feature_plan
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


try:
    from fraudshield.modeling.training import (
        MODEL_DISPLAY_NAMES,
        TrainingConfig,
        TrainingError,
        TrainingResult,
        evaluate_probabilities,
        feature_importance_table,
        model_curve_data,
        save_model_bundle,
        serialize_model_bundle,
        threshold_curve,
        train_models,
    )

    MODEL_DEPENDENCY_ERROR: ModuleNotFoundError | None = None
except ModuleNotFoundError as error:
    MODEL_DEPENDENCY_ERROR = error


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


def _render_training_form(frame: pd.DataFrame, settings: Settings, logger: logging.Logger) -> None:
    targets = infer_binary_target_columns(frame)
    if not targets:
        st.error("Model training requires a target column with exactly two non-null values.")
        return

    model_settings = settings.section("model")
    target_column = st.selectbox("Fraud target", targets, key="model_target")
    values = target_values(frame, target_column)
    likely_fraud = infer_fraud_value(values)
    fraud_value = st.selectbox(
        "Fraud label",
        values,
        index=values.index(likely_fraud),
        format_func=str,
        key="model_fraud_value",
    )

    plan = suggest_feature_plan(
        frame,
        target_column,
        max_categorical_values=int(model_settings["max_categorical_values"]),
    )
    available_features = [
        str(column) for column in frame.columns if str(column) != target_column
    ]
    selected_features = st.multiselect(
        "Input features",
        available_features,
        default=list(plan.recommended),
        help=(
            "The target is always excluded. Review identifier and high-cardinality "
            "warnings below."
        ),
    )

    with st.expander("Feature safety review"):
        st.dataframe(
            feature_schema_table(frame, plan),
            hide_index=True,
            use_container_width=True,
        )
        st.caption(
            "Detected timestamps are converted to hour, weekday, day, month, and weekend features "
            "inside every training and validation fold."
        )

    display_to_key = {display: key for key, display in MODEL_DISPLAY_NAMES.items()}
    selected_displays = st.multiselect(
        "Candidate models",
        list(display_to_key),
        default=list(display_to_key),
    )
    first, second, third = st.columns(3)
    with first:
        test_percentage = st.slider(
            "Holdout size",
            min_value=15,
            max_value=40,
            value=int(float(model_settings["test_size"]) * 100),
            step=5,
            format="%d%%",
        )
    with second:
        threshold = st.slider(
            "Initial threshold",
            min_value=0.10,
            max_value=0.90,
            value=float(model_settings["default_threshold"]),
            step=0.05,
        )
    with third:
        run_cv = st.checkbox("Cross-validation", value=True)

    class_counts = frame.loc[frame[target_column].notna(), target_column].value_counts()
    if not class_counts.empty:
        st.caption(
            "Labeled class counts: "
            + " | ".join(f"{label}: {count:,}" for label, count in class_counts.items())
        )
        if int(class_counts.sum()) < 100 or int(class_counts.min()) < 10:
            st.warning(
                "This dataset is too small for trustworthy performance claims. "
                "Training is available for workflow testing only."
            )

    if not st.button("Train and compare models", type="primary", use_container_width=True):
        return
    if not selected_features:
        st.error("Select at least one input feature.")
        return
    if not selected_displays:
        st.error("Select at least one candidate model.")
        return

    datetime_columns = tuple(
        column for column in infer_datetime_columns(frame) if column in selected_features
    )
    config = TrainingConfig(
        test_size=test_percentage / 100,
        random_state=int(model_settings["random_state"]),
        threshold=threshold,
        model_names=tuple(display_to_key[name] for name in selected_displays),
        run_cross_validation=run_cv,
        cv_folds=int(model_settings["cv_folds"]),
    )
    try:
        with st.spinner("Fitting leakage-safe pipelines and evaluating the holdout set..."):
            result = train_models(
                frame,
                target_column,
                fraud_value,
                tuple(selected_features),
                datetime_columns,
                config,
            )
    except (TrainingError, ValueError, KeyError) as error:
        logger.warning("Model training rejected: %s", error)
        st.error(str(error))
        return

    st.session_state[MODEL_RESULT_KEY] = result
    st.session_state[MODEL_CONFIG_KEY] = config
    st.session_state.pop(SCORING_RESULT_KEY, None)
    st.session_state.pop(LOCAL_EXPLANATION_KEY, None)
    st.session_state.pop(LOCAL_EXPLANATION_CONTEXT_KEY, None)
    st.session_state.pop(REPORT_PDF_KEY, None)
    st.session_state.pop(REPORT_CONTEXT_KEY, None)
    logger.info(
        "Trained %s models; selected best=%s",
        len(result.runs),
        result.best_model_name,
    )
    st.success("Training complete. Holdout data was not used for preprocessing or fitting.")


def _render_leaderboard(result: TrainingResult) -> None:
    st.markdown("#### Model leaderboard")
    table = result.leaderboard.drop(columns=["Model key"]).copy()
    numeric_columns = table.select_dtypes(include="number").columns
    table[numeric_columns] = table[numeric_columns].round(4)
    st.dataframe(table, hide_index=True, use_container_width=True)
    best = result.runs[result.best_model_name]
    st.success(
        f"Recommended candidate: {best.display_name}. "
        "Ranking uses holdout PR-AUC, then F1 and recall."
    )
    st.caption(
        f"Train rows: {result.train_rows:,} ({result.train_fraud:,} fraud) | "
        f"Holdout rows: {result.test_rows:,} ({result.test_fraud:,} fraud)"
    )


def _render_operating_point(result: TrainingResult) -> tuple[str, float]:
    model_names = list(result.runs)
    default_index = model_names.index(result.best_model_name)
    selected_name = st.selectbox(
        "Review model",
        model_names,
        index=default_index,
        format_func=lambda name: result.runs[name].display_name,
    )
    threshold = st.slider(
        "Decision threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.50,
        step=0.05,
        help="Lower thresholds catch more fraud but usually create more false alerts.",
    )
    run = result.runs[selected_name]
    metrics, _ = evaluate_probabilities(run.y_test, run.probabilities, threshold)

    columns = st.columns(6)
    columns[0].metric("Precision", f"{metrics.precision:.3f}")
    columns[1].metric("Recall", f"{metrics.recall:.3f}")
    columns[2].metric("F1", f"{metrics.f1:.3f}")
    columns[3].metric("PR-AUC", f"{metrics.average_precision:.3f}")
    columns[4].metric("False alerts", f"{metrics.false_positives:,}")
    columns[5].metric("Missed fraud", f"{metrics.false_negatives:,}")

    left, right = st.columns([0.8, 1.2])
    with left:
        matrix = [
            [metrics.true_negatives, metrics.false_positives],
            [metrics.false_negatives, metrics.true_positives],
        ]
        figure = go.Figure(
            go.Heatmap(
                z=matrix,
                x=["Predicted legitimate", "Predicted fraud"],
                y=["Actual legitimate", "Actual fraud"],
                colorscale=[[0, "#11151b"], [1, "#22d3ee"]],
                text=matrix,
                texttemplate="%{text}",
                showscale=False,
            )
        )
        figure.update_layout(title="Holdout confusion matrix", height=360)
        _plot(figure)
    with right:
        curve = threshold_curve(run)
        figure = go.Figure()
        for metric, color in (
            ("Precision", "#22d3ee"),
            ("Recall", "#fb7185"),
            ("F1", "#fbbf24"),
            ("Specificity", "#34d399"),
        ):
            figure.add_trace(
                go.Scatter(
                    x=curve["Threshold"],
                    y=curve[metric],
                    name=metric,
                    line={"color": color, "width": 2},
                )
            )
        figure.add_vline(x=threshold, line_dash="dash", line_color="#f4f7fa")
        figure.update_layout(title="Threshold trade-offs", yaxis_range=[0, 1.02], height=360)
        _plot(figure)
    return selected_name, threshold


def _render_model_curves(result: TrainingResult) -> None:
    left, right = st.columns(2)
    roc_figure = go.Figure()
    pr_figure = go.Figure()
    colors = ("#22d3ee", "#fb7185", "#fbbf24")
    for index, run in enumerate(result.runs.values()):
        roc_data, pr_data = model_curve_data(run)
        color = colors[index % len(colors)]
        if not roc_data.empty:
            roc_figure.add_trace(
                go.Scatter(
                    x=roc_data["False positive rate"],
                    y=roc_data["True positive rate"],
                    name=run.display_name,
                    line={"color": color},
                )
            )
        if not pr_data.empty:
            pr_figure.add_trace(
                go.Scatter(
                    x=pr_data["Recall"],
                    y=pr_data["Precision"],
                    name=run.display_name,
                    line={"color": color},
                )
            )
    roc_figure.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            name="Random",
            line={"color": "#637080", "dash": "dash"},
        )
    )
    roc_figure.update_layout(title="ROC curve", xaxis_title="False positive rate")
    pr_figure.update_layout(title="Precision-recall curve", xaxis_title="Recall")
    with left:
        _plot(roc_figure)
    with right:
        _plot(pr_figure)


def _render_importance(result: TrainingResult, selected_name: str) -> None:
    importance = feature_importance_table(result.runs[selected_name], limit=25)
    st.markdown("#### Model feature influence")
    if importance.empty:
        st.info("This estimator does not expose coefficients or feature importances.")
        return
    figure = go.Figure(
        go.Bar(
            x=importance["Importance"].iloc[::-1],
            y=importance["Feature"].iloc[::-1],
            orientation="h",
            marker_color="#22d3ee",
        )
    )
    figure.update_layout(height=max(360, len(importance) * 27))
    _plot(figure)
    st.caption(
        "Tree importances and logistic coefficient magnitudes are model-specific associations; "
        "they are not causal explanations. SHAP is added in Phase 5."
    )


def _render_artifact_actions(
    result: TrainingResult,
    model_name: str,
    threshold: float,
    settings: Settings,
    logger: logging.Logger,
) -> None:
    st.markdown("#### Model artifact")
    st.warning(
        "Joblib model files can execute code when loaded. "
        "Only load artifacts you created and trust."
    )
    artifact = serialize_model_bundle(result, model_name, threshold)
    left, right = st.columns(2)
    with left:
        st.download_button(
            "Download model bundle",
            data=artifact,
            file_name=f"fraudshield_{model_name}.joblib",
            mime="application/octet-stream",
            use_container_width=True,
        )
    with right:
        if st.button("Save to project models folder", use_container_width=True):
            destination = settings.path("models") / f"fraudshield_{model_name}.joblib"
            saved = save_model_bundle(result, model_name, threshold, destination)
            logger.info("Saved model bundle to %s", saved)
            st.success(f"Saved: {saved.name}")


def _render_results(settings: Settings, logger: logging.Logger) -> None:
    result = st.session_state.get(MODEL_RESULT_KEY)
    if result is None:
        st.info("Train at least one model to open evaluation results.")
        return
    _render_leaderboard(result)
    selected_name, threshold = _render_operating_point(result)
    _render_model_curves(result)
    _render_importance(result, selected_name)
    _render_artifact_actions(result, selected_name, threshold, settings, logger)


def render_model_lab(settings: Settings, logger: logging.Logger) -> None:
    """Render model training, comparison, thresholding, and export."""
    st.markdown('<div class="fs-kicker">Detection Engineering</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="fs-title">Model Lab</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="fs-subtitle">'
        "Train reproducible fraud classifiers without preprocessing leakage."
        "</p>",
        unsafe_allow_html=True,
    )

    if MODEL_DEPENDENCY_ERROR is not None:
        st.error(
            "Model dependencies are not installed. Run `pip install -r requirements.txt`, "
            "then restart Streamlit."
        )
        st.code(str(MODEL_DEPENDENCY_ERROR))
        return

    frame = st.session_state.get(ACTIVE_DATA_KEY)
    if frame is None:
        st.info("Load and review a dataset in Data Center before training models.")
        return

    training_tab, results_tab = st.tabs(("Training setup", "Evaluation"))
    with training_tab:
        _render_training_form(frame, settings, logger)
    with results_tab:
        _render_results(settings, logger)
