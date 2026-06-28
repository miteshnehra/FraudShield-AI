"""Reusable fraud-focused exploratory data analysis calculations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


TARGET_NAME_PRIORITY = (
    "class",
    "is_fraud",
    "fraud",
    "fraud_flag",
    "target",
    "label",
    "is_fraudulent",
)
AMOUNT_NAME_HINTS = ("amount", "amt", "value", "price", "transaction_value")
DATETIME_NAME_HINTS = ("timestamp", "datetime", "date", "time", "created_at", "transaction_time")
FRAUD_VALUE_HINTS = {"1", "true", "fraud", "fraudulent", "yes", "positive"}


@dataclass(frozen=True)
class TargetProfile:
    """Transaction and optional amount metrics for a selected fraud target."""

    total_rows: int
    valid_transactions: int
    missing_target: int
    fraud_transactions: int
    legitimate_transactions: int
    fraud_rate: float
    total_amount: float | None
    fraud_amount: float | None
    fraud_amount_rate: float | None


def _normalized_name(value: Any) -> str:
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


def _name_matches_hint(name: str, hints: tuple[str, ...]) -> bool:
    normalized = _normalized_name(name)
    return any(
        normalized == hint
        or normalized.startswith(f"{hint}_")
        or normalized.endswith(f"_{hint}")
        for hint in hints
    )


def target_values(frame: pd.DataFrame, target_column: str) -> list[Any]:
    """Return distinct non-null target values in stable appearance order."""
    if target_column not in frame.columns:
        raise KeyError(f"Target column not found: {target_column}")
    return frame[target_column].dropna().drop_duplicates().tolist()


def infer_binary_target_columns(frame: pd.DataFrame) -> list[str]:
    """Find binary columns and prioritize common fraud-target names."""
    candidates = [
        str(column)
        for column in frame.columns
        if frame[column].dropna().nunique() == 2
    ]
    priority = {name: index for index, name in enumerate(TARGET_NAME_PRIORITY)}
    return sorted(
        candidates,
        key=lambda column: (priority.get(_normalized_name(column), len(priority)), column.lower()),
    )


def infer_fraud_value(values: list[Any]) -> Any:
    """Choose the most likely positive/fraud label from target values."""
    if not values:
        raise ValueError("The selected target has no non-null values.")
    for value in values:
        if _normalized_name(value) in FRAUD_VALUE_HINTS:
            return value
    numeric_values = [value for value in values if isinstance(value, (int, float))]
    if len(numeric_values) == len(values):
        return max(numeric_values)
    return values[-1]


def infer_amount_columns(frame: pd.DataFrame, *, exclude: tuple[str, ...] = ()) -> list[str]:
    """Return numeric amount-like columns before other numeric choices."""
    numeric = [
        str(column)
        for column in frame.select_dtypes(include=["number"]).columns
        if str(column) not in exclude
    ]
    return sorted(
        numeric,
        key=lambda column: (
            0 if _name_matches_hint(column, AMOUNT_NAME_HINTS) else 1,
            column.lower(),
        ),
    )


def infer_datetime_columns(frame: pd.DataFrame) -> list[str]:
    """Find native datetimes and date-named columns that parse reliably."""
    candidates: list[str] = []
    for column in frame.columns:
        name = str(column)
        series = frame[column]
        if isinstance(series.dtype, pd.DatetimeTZDtype) or pd.api.types.is_datetime64_any_dtype(
            series
        ):
            candidates.append(name)
            continue
        if not _name_matches_hint(name, DATETIME_NAME_HINTS):
            continue
        non_null = series.dropna().head(500)
        if non_null.empty:
            continue
        parsed = pd.to_datetime(non_null, errors="coerce", utc=True)
        if float(parsed.notna().mean()) >= 0.8:
            candidates.append(name)
    return candidates


def infer_categorical_columns(
    frame: pd.DataFrame,
    *,
    exclude: tuple[str, ...] = (),
    max_unique: int = 50,
) -> list[str]:
    """Find text, boolean, categorical, and low-cardinality integer columns."""
    candidates = []
    for column in frame.columns:
        name = str(column)
        if name in exclude:
            continue
        series = frame[column]
        unique = int(series.nunique(dropna=True))
        unique_ratio = unique / max(len(series), 1)
        normalized_name = _normalized_name(name)
        looks_like_identifier = normalized_name == "id" or normalized_name.endswith("_id")
        if looks_like_identifier and unique_ratio > 0.5:
            continue
        is_textual = (
            isinstance(series.dtype, pd.CategoricalDtype)
            or pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
            or pd.api.types.is_bool_dtype(series)
        )
        is_small_integer = pd.api.types.is_integer_dtype(series) and unique <= max_unique
        if 1 < unique <= max_unique and (is_textual or is_small_integer):
            candidates.append(name)
    return candidates


def profile_target(
    frame: pd.DataFrame,
    target_column: str,
    fraud_value: Any,
    amount_column: str | None = None,
) -> TargetProfile:
    """Calculate fraud prevalence and optional fraud-value exposure."""
    if target_column not in frame.columns:
        raise KeyError(f"Target column not found: {target_column}")

    valid_mask = frame[target_column].notna()
    fraud_mask = valid_mask & frame[target_column].eq(fraud_value).fillna(False)
    valid_transactions = int(valid_mask.sum())
    fraud_transactions = int(fraud_mask.sum())
    legitimate_transactions = valid_transactions - fraud_transactions
    fraud_rate = fraud_transactions / valid_transactions * 100 if valid_transactions else 0.0

    total_amount: float | None = None
    fraud_amount: float | None = None
    fraud_amount_rate: float | None = None
    if amount_column is not None:
        if amount_column not in frame.columns:
            raise KeyError(f"Amount column not found: {amount_column}")
        amount = pd.to_numeric(frame[amount_column], errors="coerce")
        valid_amount = amount[valid_mask].dropna()
        fraud_values = amount[fraud_mask].dropna()
        if not valid_amount.empty:
            total_amount = float(valid_amount.sum())
            fraud_amount = float(fraud_values.sum()) if not fraud_values.empty else 0.0
            if total_amount != 0:
                fraud_amount_rate = fraud_amount / total_amount * 100

    return TargetProfile(
        total_rows=len(frame),
        valid_transactions=valid_transactions,
        missing_target=len(frame) - valid_transactions,
        fraud_transactions=fraud_transactions,
        legitimate_transactions=legitimate_transactions,
        fraud_rate=round(fraud_rate, 4),
        total_amount=total_amount,
        fraud_amount=fraud_amount,
        fraud_amount_rate=round(fraud_amount_rate, 4) if fraud_amount_rate is not None else None,
    )


def class_distribution(
    frame: pd.DataFrame,
    target_column: str,
    fraud_value: Any,
) -> pd.DataFrame:
    """Return legitimate/fraud transaction counts and shares."""
    profile = profile_target(frame, target_column, fraud_value)
    denominator = max(profile.valid_transactions, 1)
    return pd.DataFrame(
        [
            {
                "Class": "Legitimate",
                "Transactions": profile.legitimate_transactions,
                "Share %": profile.legitimate_transactions / denominator * 100,
            },
            {
                "Class": "Fraud",
                "Transactions": profile.fraud_transactions,
                "Share %": profile.fraud_transactions / denominator * 100,
            },
        ]
    )


def categorical_fraud_rates(
    frame: pd.DataFrame,
    target_column: str,
    fraud_value: Any,
    category_column: str,
    *,
    min_transactions: int = 1,
    max_categories: int = 20,
) -> pd.DataFrame:
    """Calculate transaction volume and fraud rate for each category."""
    if category_column == target_column:
        raise ValueError("Category column must be different from the fraud target.")
    required = {target_column, category_column}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Columns not found: {', '.join(sorted(missing))}")

    working = frame.loc[frame[target_column].notna(), [category_column, target_column]].copy()
    working["Category"] = working[category_column].astype("string").fillna("(Missing)")
    working["Is fraud"] = working[target_column].eq(fraud_value).fillna(False).astype(int)
    grouped = (
        working.groupby("Category", dropna=False, observed=True)
        .agg(Transactions=("Is fraud", "size"), Fraud=("Is fraud", "sum"))
        .reset_index()
    )
    grouped = grouped[grouped["Transactions"] >= max(1, min_transactions)]
    grouped["Legitimate"] = grouped["Transactions"] - grouped["Fraud"]
    grouped["Fraud rate %"] = grouped["Fraud"] / grouped["Transactions"] * 100
    grouped = grouped.nlargest(max_categories, "Transactions")
    return grouped.sort_values(
        ["Fraud rate %", "Transactions"], ascending=[False, False], ignore_index=True
    )


def amount_distribution_frame(
    frame: pd.DataFrame,
    target_column: str,
    fraud_value: Any,
    amount_column: str,
    *,
    max_rows: int = 50000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Prepare a numeric, stratified amount sample suitable for plotting."""
    working = frame.loc[frame[target_column].notna(), [amount_column, target_column]].copy()
    working["Amount"] = pd.to_numeric(working[amount_column], errors="coerce")
    working["Class"] = "Legitimate"
    fraud_mask = working[target_column].eq(fraud_value).fillna(False)
    working.loc[fraud_mask, "Class"] = "Fraud"
    working = working.dropna(subset=["Amount"])[["Amount", "Class"]]
    if len(working) <= max_rows:
        return working.reset_index(drop=True)

    class_count = max(int(working["Class"].nunique()), 1)
    per_class = max(max_rows // class_count, 1)
    samples = []
    for _, group in working.groupby("Class", observed=True):
        samples.append(group.sample(n=min(len(group), per_class), random_state=random_state))
    return pd.concat(samples, ignore_index=True)


def time_fraud_trend(
    frame: pd.DataFrame,
    target_column: str,
    fraud_value: Any,
    datetime_column: str,
    *,
    period: str = "day",
) -> pd.DataFrame:
    """Aggregate transaction volume and fraud rate over calendar periods."""
    if period not in {"hour", "day", "week", "month"}:
        raise ValueError(f"Unsupported time period: {period}")

    timestamps = pd.to_datetime(frame[datetime_column], errors="coerce", utc=True)
    working = pd.DataFrame(
        {
            "Timestamp": timestamps.dt.tz_convert(None),
            "Is fraud": frame[target_column].eq(fraud_value).fillna(False).astype(int),
            "Target valid": frame[target_column].notna(),
        }
    )
    working = working[working["Target valid"] & working["Timestamp"].notna()].copy()
    if working.empty:
        return pd.DataFrame(columns=["Period", "Transactions", "Fraud", "Fraud rate %"])

    if period == "hour":
        working["Period"] = working["Timestamp"].dt.floor("h")
    elif period == "day":
        working["Period"] = working["Timestamp"].dt.floor("d")
    elif period == "week":
        working["Period"] = working["Timestamp"].dt.to_period("W").dt.start_time
    else:
        working["Period"] = working["Timestamp"].dt.to_period("M").dt.start_time

    trend = (
        working.groupby("Period", observed=True)
        .agg(Transactions=("Is fraud", "size"), Fraud=("Is fraud", "sum"))
        .reset_index()
    )
    trend["Fraud rate %"] = trend["Fraud"] / trend["Transactions"] * 100
    return trend


def hourly_fraud_pattern(
    frame: pd.DataFrame,
    target_column: str,
    fraud_value: Any,
    datetime_column: str,
) -> pd.DataFrame:
    """Aggregate fraud volume and rate by hour of day."""
    timestamps = pd.to_datetime(frame[datetime_column], errors="coerce", utc=True)
    working = pd.DataFrame(
        {
            "Hour": timestamps.dt.hour,
            "Is fraud": frame[target_column].eq(fraud_value).fillna(False).astype(int),
            "Target valid": frame[target_column].notna(),
        }
    )
    working = working[working["Target valid"] & working["Hour"].notna()]
    pattern = (
        working.groupby("Hour", observed=True)
        .agg(Transactions=("Is fraud", "size"), Fraud=("Is fraud", "sum"))
        .reset_index()
    )
    pattern["Fraud rate %"] = pattern["Fraud"] / pattern["Transactions"] * 100
    return pattern


def correlation_matrix(
    frame: pd.DataFrame,
    target_column: str,
    fraud_value: Any,
    *,
    max_features: int = 25,
) -> pd.DataFrame:
    """Return a Spearman matrix with a consistently oriented fraud indicator."""
    numeric = frame.select_dtypes(include=["number", "bool"]).copy()
    numeric = numeric.drop(columns=[target_column], errors="ignore")
    for column in numeric.select_dtypes(include=["bool"]).columns:
        numeric[column] = numeric[column].astype(int)
    numeric["Fraud indicator"] = frame[target_column].eq(fraud_value).fillna(False).astype(int)
    numeric = numeric.replace([float("inf"), float("-inf")], pd.NA)
    useful_columns = [
        column
        for column in numeric.columns
        if numeric[column].notna().sum() >= 2 and numeric[column].nunique(dropna=True) > 1
    ]
    numeric = numeric[useful_columns]
    if "Fraud indicator" not in numeric.columns or len(numeric.columns) < 2:
        return pd.DataFrame()

    if len(numeric.columns) > max_features:
        initial = numeric.corr(method="spearman")["Fraud indicator"].abs()
        selected = initial.drop("Fraud indicator").nlargest(max_features - 1).index.tolist()
        numeric = numeric[selected + ["Fraud indicator"]]
    return numeric.corr(method="spearman")


def strongest_fraud_correlations(matrix: pd.DataFrame, *, limit: int = 10) -> pd.DataFrame:
    """Rank feature correlations with the fraud indicator by absolute strength."""
    if matrix.empty or "Fraud indicator" not in matrix.columns:
        return pd.DataFrame(columns=["Feature", "Correlation", "Absolute correlation"])
    values = matrix["Fraud indicator"].drop(labels=["Fraud indicator"], errors="ignore")
    result = pd.DataFrame(
        {
            "Feature": values.index.astype(str),
            "Correlation": values.to_numpy(),
            "Absolute correlation": values.abs().to_numpy(),
        }
    )
    return result.nlargest(limit, "Absolute correlation").reset_index(drop=True)
