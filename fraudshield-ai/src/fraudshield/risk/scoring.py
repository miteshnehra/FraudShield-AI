"""Transparent probability-to-risk scoring for trained model runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd


if TYPE_CHECKING:
    from fraudshield.modeling.training import TrainingResult


SCORE_COLUMNS = (
    "FS_Row_Number",
    "FS_Source_Index",
    "FS_Fraud_Probability",
    "FS_Risk_Score",
    "FS_Risk_Band",
    "FS_Decision",
    "FS_Model",
)


@dataclass(frozen=True)
class RiskThresholds:
    """Inclusive upper bounds for four operational risk bands."""

    low_max: int = 29
    medium_max: int = 59
    high_max: int = 79
    critical_max: int = 100

    def __post_init__(self) -> None:
        values = (self.low_max, self.medium_max, self.high_max, self.critical_max)
        if not 0 <= self.low_max < self.medium_max < self.high_max < self.critical_max:
            raise ValueError("Risk thresholds must be strictly increasing from 0 to 100.")
        if self.critical_max != 100:
            raise ValueError("Critical risk must end at 100.")
        if any(not isinstance(value, int) for value in values):
            raise TypeError("Risk thresholds must be integers.")

    @classmethod
    def from_mapping(cls, values: dict[str, object]) -> RiskThresholds:
        return cls(
            low_max=int(values["low_max"]),
            medium_max=int(values["medium_max"]),
            high_max=int(values["high_max"]),
            critical_max=int(values["critical_max"]),
        )


@dataclass(frozen=True)
class ScoringResult:
    """Batch scores and the policy settings used to create them."""

    frame: pd.DataFrame
    model_name: str
    model_display_name: str
    decision_threshold: float
    thresholds: RiskThresholds


def risk_band(score: float, thresholds: RiskThresholds) -> str:
    """Convert a finite 0-100 score into an operational band."""
    if not np.isfinite(score) or score < 0 or score > 100:
        raise ValueError("Risk score must be a finite value between 0 and 100.")
    if score <= thresholds.low_max:
        return "Low"
    if score <= thresholds.medium_max:
        return "Medium"
    if score <= thresholds.high_max:
        return "High"
    return "Critical"


def risk_summary(band: str) -> str:
    """Return a cautious human-review message for a risk band."""
    messages = {
        "Low": "Low model risk; continue normal monitoring.",
        "Medium": "Moderate model risk; monitor and apply business rules.",
        "High": "High model risk; prioritize manual review.",
        "Critical": "Critical model risk; investigate promptly before taking action.",
    }
    if band not in messages:
        raise ValueError(f"Unknown risk band: {band}")
    return messages[band]


def score_transactions(
    frame: pd.DataFrame,
    result: TrainingResult,
    model_name: str,
    decision_threshold: float,
    thresholds: RiskThresholds,
) -> ScoringResult:
    """Score every row with a fitted pipeline without changing source data."""
    if model_name not in result.runs:
        raise KeyError(f"Trained model not found: {model_name}")
    if not 0.0 < decision_threshold < 1.0:
        raise ValueError("Decision threshold must be between 0 and 1.")
    if frame.empty:
        raise ValueError("At least one transaction is required for scoring.")
    missing_features = [
        column for column in result.feature_columns if column not in frame.columns
    ]
    if missing_features:
        raise KeyError(f"Required model features are missing: {', '.join(missing_features)}")
    collisions = sorted(set(SCORE_COLUMNS).intersection(frame.columns))
    if collisions:
        raise ValueError(
            f"Dataset already contains reserved score columns: {', '.join(collisions)}"
        )

    run = result.runs[model_name]
    source_features = frame.loc[:, list(result.feature_columns)]
    probabilities = np.asarray(run.pipeline.predict_proba(source_features)[:, 1], dtype=float)
    if len(probabilities) != len(frame):
        raise ValueError("Model returned an unexpected number of probability scores.")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Model returned invalid fraud probabilities.")

    scores = np.clip(probabilities * 100, 0, 100)
    scored = frame.copy(deep=True)
    scored.insert(0, "FS_Source_Index", frame.index.map(str))
    scored.insert(0, "FS_Row_Number", np.arange(len(frame), dtype=int))
    scored["FS_Fraud_Probability"] = np.round(probabilities, 6)
    scored["FS_Risk_Score"] = np.round(scores, 2)
    scored["FS_Risk_Band"] = [risk_band(float(score), thresholds) for score in scores]
    scored["FS_Decision"] = np.where(
        probabilities >= decision_threshold,
        "Manual review",
        "No automatic hold",
    )
    scored["FS_Model"] = run.display_name

    return ScoringResult(
        frame=scored,
        model_name=model_name,
        model_display_name=run.display_name,
        decision_threshold=decision_threshold,
        thresholds=thresholds,
    )


def risk_band_counts(scored_frame: pd.DataFrame) -> pd.DataFrame:
    """Return ordered counts and shares for scored risk bands."""
    if "FS_Risk_Band" not in scored_frame.columns:
        raise KeyError("Scored frame does not contain FS_Risk_Band.")
    order = ["Low", "Medium", "High", "Critical"]
    counts = scored_frame["FS_Risk_Band"].value_counts().reindex(order, fill_value=0)
    total = max(len(scored_frame), 1)
    return pd.DataFrame(
        {
            "Risk band": order,
            "Transactions": counts.to_numpy(dtype=int),
            "Share %": (counts.to_numpy(dtype=float) / total * 100).round(2),
        }
    )
