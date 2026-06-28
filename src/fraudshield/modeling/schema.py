"""Feature selection and deterministic transaction feature engineering."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fraudshield.analysis.eda import infer_datetime_columns


@dataclass(frozen=True)
class FeaturePlan:
    """Recommended model inputs and columns excluded with reasons."""

    recommended: tuple[str, ...]
    datetime_columns: tuple[str, ...]
    excluded: tuple[tuple[str, str], ...]


def _normalized_name(column: str) -> str:
    return str(column).strip().lower().replace(" ", "_").replace("-", "_")


def _looks_like_identifier(column: str) -> bool:
    name = _normalized_name(column)
    return name in {"id", "uuid", "guid"} or name.endswith(("_id", "_uuid", "_guid"))


def suggest_feature_plan(
    frame: pd.DataFrame,
    target_column: str,
    *,
    max_categorical_values: int = 200,
) -> FeaturePlan:
    """Suggest useful source features without using the target values."""
    if target_column not in frame.columns:
        raise KeyError(f"Target column not found: {target_column}")

    datetime_columns = set(infer_datetime_columns(frame))
    recommended: list[str] = []
    selected_datetimes: list[str] = []
    excluded: list[tuple[str, str]] = []
    row_count = max(len(frame), 1)

    for raw_column in frame.columns:
        column = str(raw_column)
        if column == target_column:
            excluded.append((column, "Fraud target"))
            continue

        series = frame[raw_column]
        non_null = int(series.notna().sum())
        unique = int(series.nunique(dropna=True))
        unique_ratio = unique / row_count
        if non_null == 0:
            excluded.append((column, "All values missing"))
            continue
        if unique <= 1:
            excluded.append((column, "Constant column"))
            continue
        if _looks_like_identifier(column) and unique_ratio > 0.5:
            excluded.append((column, "High-cardinality identifier"))
            continue
        is_text = pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)
        if is_text and unique > max_categorical_values and unique_ratio > 0.5:
            excluded.append((column, "High-cardinality text"))
            continue

        recommended.append(column)
        if column in datetime_columns:
            selected_datetimes.append(column)

    return FeaturePlan(
        recommended=tuple(recommended),
        datetime_columns=tuple(selected_datetimes),
        excluded=tuple(excluded),
    )


def validate_feature_columns(
    frame: pd.DataFrame,
    target_column: str,
    feature_columns: tuple[str, ...] | list[str],
) -> tuple[str, ...]:
    """Validate user-selected source columns and preserve their order."""
    selected = tuple(dict.fromkeys(str(column) for column in feature_columns))
    if not selected:
        raise ValueError("Select at least one feature column.")
    if target_column in selected:
        raise ValueError("The fraud target cannot be used as an input feature.")
    missing = [column for column in selected if column not in frame.columns]
    if missing:
        raise KeyError(f"Feature columns not found: {', '.join(missing)}")
    return selected


def prepare_feature_frame(
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...] | list[str],
    datetime_columns: tuple[str, ...] | list[str] = (),
) -> pd.DataFrame:
    """Select source features and derive stable numeric parts from timestamps."""
    selected = tuple(feature_columns)
    datetime_set = set(datetime_columns).intersection(selected)
    prepared = pd.DataFrame(index=frame.index)

    for column in selected:
        if column not in frame.columns:
            raise KeyError(f"Feature column not found: {column}")
        if column not in datetime_set:
            prepared[column] = frame[column]
            continue

        parsed = pd.to_datetime(frame[column], errors="coerce", utc=True)
        prepared[f"{column}__hour"] = parsed.dt.hour.astype("float64")
        prepared[f"{column}__day_of_week"] = parsed.dt.dayofweek.astype("float64")
        prepared[f"{column}__day_of_month"] = parsed.dt.day.astype("float64")
        prepared[f"{column}__month"] = parsed.dt.month.astype("float64")
        prepared[f"{column}__is_weekend"] = parsed.dt.dayofweek.isin([5, 6]).astype("float64")
        prepared.loc[parsed.isna(), f"{column}__is_weekend"] = np.nan

    prepared = prepared.replace([np.inf, -np.inf], np.nan)
    for column in prepared.columns:
        if not pd.api.types.is_numeric_dtype(prepared[column]):
            prepared[column] = prepared[column].astype("object")
            prepared[column] = prepared[column].where(prepared[column].notna(), np.nan)
    return prepared


def feature_schema_table(frame: pd.DataFrame, plan: FeaturePlan) -> pd.DataFrame:
    """Return an explainable source-column recommendation table."""
    datetime_set = set(plan.datetime_columns)
    excluded = dict(plan.excluded)
    rows = []
    for raw_column in frame.columns:
        column = str(raw_column)
        if column in excluded:
            recommendation = "Excluded"
            reason = excluded[column]
        elif column in datetime_set:
            recommendation = "Use derived time parts"
            reason = "Timestamp detected"
        else:
            recommendation = "Recommended"
            reason = "Usable feature"
        rows.append(
            {
                "Column": column,
                "Data type": str(frame[raw_column].dtype),
                "Unique": int(frame[raw_column].nunique(dropna=True)),
                "Recommendation": recommendation,
                "Reason": reason,
            }
        )
    return pd.DataFrame(rows)

