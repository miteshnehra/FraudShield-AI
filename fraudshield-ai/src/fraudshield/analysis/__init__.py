"""Fraud-focused exploratory analysis services."""

from fraudshield.analysis.eda import (
    TargetProfile,
    categorical_fraud_rates,
    correlation_matrix,
    infer_amount_columns,
    infer_binary_target_columns,
    infer_categorical_columns,
    infer_datetime_columns,
    infer_fraud_value,
    profile_target,
    time_fraud_trend,
)

__all__ = [
    "TargetProfile",
    "categorical_fraud_rates",
    "correlation_matrix",
    "infer_amount_columns",
    "infer_binary_target_columns",
    "infer_categorical_columns",
    "infer_datetime_columns",
    "infer_fraud_value",
    "profile_target",
    "time_fraud_trend",
]

