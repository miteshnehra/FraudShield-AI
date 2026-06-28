"""Local model explanations and concise human-readable reasons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd


if TYPE_CHECKING:
    from fraudshield.modeling.training import TrainingResult


class ExplanationUnavailableError(RuntimeError):
    """Raised when the selected model cannot be explained in the current runtime."""


@dataclass(frozen=True)
class LocalExplanation:
    """Signed local feature contributions for one transaction."""

    method: str
    base_value: float | None
    contributions: pd.DataFrame


def _dense_row(values: Any) -> np.ndarray:
    if hasattr(values, "toarray"):
        return np.asarray(values.toarray(), dtype=float)
    return np.asarray(values, dtype=float)


def _friendly_feature_name(feature: str) -> str:
    return feature.replace("__", " ").replace("_", " ").strip().title()


def _contribution_table(feature_names: np.ndarray, values: np.ndarray) -> pd.DataFrame:
    if len(feature_names) != len(values):
        raise ExplanationUnavailableError(
            "Explanation feature names and contribution values are inconsistent."
        )
    table = pd.DataFrame(
        {
            "Feature": feature_names.astype(str),
            "Readable feature": [_friendly_feature_name(name) for name in feature_names],
            "Contribution": values.astype(float),
            "Impact": np.abs(values.astype(float)),
        }
    )
    table["Direction"] = np.where(
        table["Contribution"] > 0,
        "Raises risk",
        np.where(table["Contribution"] < 0, "Lowers risk", "Neutral"),
    )
    return table.sort_values("Impact", ascending=False, ignore_index=True)


def _logistic_explanation(pipeline: Any, source_row: pd.DataFrame) -> LocalExplanation:
    transformed = _dense_row(pipeline[:-1].transform(source_row))
    estimator = pipeline.named_steps["model"]
    coefficients = np.asarray(estimator.coef_[0], dtype=float)
    contributions = transformed[0] * coefficients
    names = np.asarray(
        pipeline.named_steps["preprocessor"].get_feature_names_out(),
        dtype=str,
    )
    intercept = float(np.asarray(estimator.intercept_).ravel()[0])
    return LocalExplanation(
        method="Exact logistic log-odds contribution",
        base_value=intercept,
        contributions=_contribution_table(names, contributions),
    )


def _tree_shap_explanation(pipeline: Any, source_row: pd.DataFrame) -> LocalExplanation:
    try:
        import shap
    except ModuleNotFoundError as error:
        raise ExplanationUnavailableError(
            "SHAP is required for local tree-model explanations. Install requirements and restart."
        ) from error

    transformed = _dense_row(pipeline[:-1].transform(source_row))
    estimator = pipeline.named_steps["model"]
    names = np.asarray(
        pipeline.named_steps["preprocessor"].get_feature_names_out(),
        dtype=str,
    )
    explanation = shap.TreeExplainer(estimator)(transformed)
    values = np.asarray(explanation.values)
    base_values = np.asarray(explanation.base_values)

    if values.ndim == 3:
        local_values = values[0, :, 1]
        base_value = float(base_values[0, 1])
    elif values.ndim == 2:
        local_values = values[0]
        base_value = float(base_values.reshape(-1)[0])
    else:
        raise ExplanationUnavailableError("Unexpected SHAP value shape for this tree model.")
    return LocalExplanation(
        method="Tree SHAP contribution",
        base_value=base_value,
        contributions=_contribution_table(names, local_values),
    )


def explain_transaction(
    frame: pd.DataFrame,
    result: TrainingResult,
    model_name: str,
    row_number: int,
) -> LocalExplanation:
    """Explain one scored row with exact logistic or SHAP tree contributions."""
    if model_name not in result.runs:
        raise KeyError(f"Trained model not found: {model_name}")
    if row_number < 0 or row_number >= len(frame):
        raise IndexError("Transaction row number is outside the active dataset.")

    run = result.runs[model_name]
    source_row = frame.iloc[[row_number]].loc[:, list(result.feature_columns)]
    estimator = run.pipeline.named_steps["model"]
    if hasattr(estimator, "coef_"):
        return _logistic_explanation(run.pipeline, source_row)
    if hasattr(estimator, "feature_importances_"):
        return _tree_shap_explanation(run.pipeline, source_row)
    raise ExplanationUnavailableError("The selected model has no supported local explainer.")


def human_readable_reasons(
    explanation: LocalExplanation,
    *,
    positive_limit: int = 3,
    negative_limit: int = 2,
) -> list[str]:
    """Turn top signed contributions into cautious review reasons."""
    table = explanation.contributions
    reasons: list[str] = []
    positive = table[table["Contribution"] > 0].nlargest(positive_limit, "Impact")
    negative = table[table["Contribution"] < 0].nlargest(negative_limit, "Impact")
    for _, row in positive.iterrows():
        reasons.append(f"{row['Readable feature']} increased the model's fraud signal.")
    for _, row in negative.iterrows():
        reasons.append(f"{row['Readable feature']} reduced the model's fraud signal.")
    if not reasons:
        reasons.append("No feature made a material signed contribution for this transaction.")
    return reasons
