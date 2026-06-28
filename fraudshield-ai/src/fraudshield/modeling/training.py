"""Leakage-safe model training, evaluation, and artifact persistence."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fraudshield.modeling.schema import prepare_feature_frame, validate_feature_columns


MODEL_DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "extra_trees": "Extra Trees",
}


class TrainingError(ValueError):
    """Raised when a dataset or training configuration is unsafe or invalid."""


@dataclass(frozen=True)
class TrainingConfig:
    """Reproducible model-training controls."""

    test_size: float = 0.20
    random_state: int = 42
    threshold: float = 0.50
    model_names: tuple[str, ...] = tuple(MODEL_DISPLAY_NAMES)
    run_cross_validation: bool = True
    cv_folds: int = 5


@dataclass(frozen=True)
class MetricSet:
    """Holdout metrics at one decision threshold."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    average_precision: float | None
    specificity: float
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int


@dataclass(frozen=True)
class ModelRun:
    """One fitted pipeline and its holdout outputs."""

    model_name: str
    display_name: str
    pipeline: Pipeline
    metrics: MetricSet
    y_test: np.ndarray
    probabilities: np.ndarray
    predictions: np.ndarray
    test_indices: tuple[Any, ...]
    cv_f1_mean: float | None
    cv_average_precision_mean: float | None


@dataclass(frozen=True)
class TrainingResult:
    """All fitted candidates and reproducibility metadata."""

    runs: dict[str, ModelRun]
    leaderboard: pd.DataFrame
    best_model_name: str
    target_column: str
    fraud_value: Any
    feature_columns: tuple[str, ...]
    datetime_columns: tuple[str, ...]
    train_rows: int
    test_rows: int
    train_fraud: int
    test_fraud: int
    dropped_target_rows: int
    created_at: str
    data_signature: str


class TransactionFeatureBuilder(BaseEstimator, TransformerMixin):
    """Sklearn-compatible deterministic feature preparation."""

    def __init__(self, feature_columns: tuple[str, ...], datetime_columns: tuple[str, ...]):
        self.feature_columns = feature_columns
        self.datetime_columns = datetime_columns

    def fit(self, frame: pd.DataFrame, target: Any = None) -> TransactionFeatureBuilder:
        prepare_feature_frame(frame, self.feature_columns, self.datetime_columns)
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return prepare_feature_frame(frame, self.feature_columns, self.datetime_columns)


def _validate_config(config: TrainingConfig) -> None:
    if not 0.05 <= config.test_size <= 0.50:
        raise TrainingError("Test size must be between 5% and 50%.")
    if not 0.0 < config.threshold < 1.0:
        raise TrainingError("Decision threshold must be between 0 and 1.")
    if config.cv_folds < 2:
        raise TrainingError("Cross-validation folds must be at least 2.")
    if not config.model_names:
        raise TrainingError("Select at least one model.")
    unknown = sorted(set(config.model_names).difference(MODEL_DISPLAY_NAMES))
    if unknown:
        raise TrainingError(f"Unknown models: {', '.join(unknown)}")


def _build_preprocessor(prepared_training_frame: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = list(
        prepared_training_frame.select_dtypes(include=["number", "bool"]).columns
    )
    categorical_columns = [
        column for column in prepared_training_frame.columns if column not in numeric_columns
    ]
    transformers = []
    if numeric_columns:
        numeric_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_columns))
    if categorical_columns:
        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="infrequent_if_exist",
                        min_frequency=2,
                        sparse_output=True,
                    ),
                ),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_columns))
    if not transformers:
        raise TrainingError("No usable numeric or categorical features remain after preparation.")
    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.3,
        verbose_feature_names_out=False,
    )


def _model_estimator(name: str, config: TrainingConfig) -> BaseEstimator:
    if name == "logistic_regression":
        return LogisticRegression(
            class_weight="balanced",
            max_iter=1500,
            random_state=config.random_state,
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced_subsample",
            random_state=config.random_state,
            n_jobs=-1,
        )
    if name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=config.random_state,
            n_jobs=-1,
        )
    raise TrainingError(f"Unknown model: {name}")


def evaluate_probabilities(
    y_true: np.ndarray | pd.Series,
    probabilities: np.ndarray,
    threshold: float,
) -> tuple[MetricSet, np.ndarray]:
    """Evaluate fraud probabilities at a user-selected operating threshold."""
    if not 0.0 < threshold < 1.0:
        raise ValueError("Threshold must be between 0 and 1.")
    actual = np.asarray(y_true, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    if len(actual) != len(scores):
        raise ValueError("Labels and probabilities must have the same length.")
    if len(actual) == 0:
        raise ValueError("At least one holdout prediction is required.")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Probabilities must be finite values between 0 and 1.")

    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(actual, predictions, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    has_both_classes = len(np.unique(actual)) == 2
    metrics = MetricSet(
        accuracy=float(accuracy_score(actual, predictions)),
        precision=float(precision_score(actual, predictions, zero_division=0)),
        recall=float(recall_score(actual, predictions, zero_division=0)),
        f1=float(f1_score(actual, predictions, zero_division=0)),
        roc_auc=float(roc_auc_score(actual, scores)) if has_both_classes else None,
        average_precision=(
            float(average_precision_score(actual, scores)) if has_both_classes else None
        ),
        specificity=float(specificity),
        true_negatives=int(tn),
        false_positives=int(fp),
        false_negatives=int(fn),
        true_positives=int(tp),
    )
    return metrics, predictions


def _cross_validation_scores(
    pipeline: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    config: TrainingConfig,
) -> tuple[float | None, float | None]:
    if not config.run_cross_validation:
        return None, None
    smallest_class = int(target.value_counts().min())
    folds = min(config.cv_folds, smallest_class)
    if folds < 2:
        return None, None
    splitter = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=config.random_state,
    )
    scores = cross_validate(
        clone(pipeline),
        features,
        target,
        cv=splitter,
        scoring={"f1": "f1", "average_precision": "average_precision"},
        n_jobs=None,
        error_score="raise",
    )
    return float(scores["test_f1"].mean()), float(scores["test_average_precision"].mean())


def _leaderboard(runs: dict[str, ModelRun]) -> pd.DataFrame:
    rows = []
    for run in runs.values():
        rows.append(
            {
                "Model": run.display_name,
                "Model key": run.model_name,
                "Precision": run.metrics.precision,
                "Recall": run.metrics.recall,
                "F1": run.metrics.f1,
                "PR-AUC": run.metrics.average_precision,
                "ROC-AUC": run.metrics.roc_auc,
                "Accuracy": run.metrics.accuracy,
                "CV F1": run.cv_f1_mean,
                "CV PR-AUC": run.cv_average_precision_mean,
            }
        )
    table = pd.DataFrame(rows)
    ranking = table.assign(
        _pr=table["PR-AUC"].fillna(-1),
        _f1=table["F1"].fillna(-1),
        _recall=table["Recall"].fillna(-1),
    ).sort_values(["_pr", "_f1", "_recall"], ascending=False)
    return ranking.drop(columns=["_pr", "_f1", "_recall"]).reset_index(drop=True)


def train_models(
    frame: pd.DataFrame,
    target_column: str,
    fraud_value: Any,
    feature_columns: tuple[str, ...] | list[str],
    datetime_columns: tuple[str, ...] | list[str],
    config: TrainingConfig,
) -> TrainingResult:
    """Fit candidate pipelines with a stratified, untouched holdout set."""
    _validate_config(config)
    selected = validate_feature_columns(frame, target_column, feature_columns)
    datetimes = tuple(column for column in datetime_columns if column in selected)

    labeled_mask = frame[target_column].notna()
    labeled = frame.loc[labeled_mask]
    features = labeled.loc[:, list(selected)].copy()
    target = labeled[target_column].eq(fraud_value).fillna(False).astype(int)
    class_counts = target.value_counts()
    if len(class_counts) != 2:
        raise TrainingError("The selected target and fraud label must produce two classes.")
    if int(class_counts.min()) < 2:
        raise TrainingError("Each class needs at least two labeled rows for a stratified split.")

    desired_test_rows = max(2, math.ceil(len(target) * config.test_size))
    maximum_test_rows = len(target) - 2
    if desired_test_rows > maximum_test_rows:
        raise TrainingError("Not enough labeled rows to keep both classes in train and test sets.")

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=desired_test_rows,
        random_state=config.random_state,
        stratify=target,
    )

    builder = TransactionFeatureBuilder(selected, datetimes)
    prepared_train = builder.fit_transform(x_train)
    preprocessor = _build_preprocessor(prepared_train)
    runs: dict[str, ModelRun] = {}

    for model_name in config.model_names:
        pipeline = Pipeline(
            [
                ("feature_builder", clone(builder)),
                ("preprocessor", clone(preprocessor)),
                ("model", _model_estimator(model_name, config)),
            ]
        )
        pipeline.fit(x_train, y_train)
        probabilities = pipeline.predict_proba(x_test)[:, 1]
        metrics, predictions = evaluate_probabilities(y_test, probabilities, config.threshold)
        cv_f1, cv_pr_auc = _cross_validation_scores(pipeline, x_train, y_train, config)
        runs[model_name] = ModelRun(
            model_name=model_name,
            display_name=MODEL_DISPLAY_NAMES[model_name],
            pipeline=pipeline,
            metrics=metrics,
            y_test=y_test.to_numpy(dtype=int),
            probabilities=np.asarray(probabilities, dtype=float),
            predictions=predictions,
            test_indices=tuple(x_test.index.tolist()),
            cv_f1_mean=cv_f1,
            cv_average_precision_mean=cv_pr_auc,
        )

    leaderboard = _leaderboard(runs)
    best_display_name = str(leaderboard.iloc[0]["Model"])
    best_model_name = next(
        name
        for name, display_name in MODEL_DISPLAY_NAMES.items()
        if display_name == best_display_name
    )
    signature_frame = frame.loc[labeled_mask, [*selected, target_column]]
    signature_bytes = pd.util.hash_pandas_object(signature_frame, index=True).values.tobytes()

    return TrainingResult(
        runs=runs,
        leaderboard=leaderboard,
        best_model_name=best_model_name,
        target_column=target_column,
        fraud_value=fraud_value,
        feature_columns=selected,
        datetime_columns=datetimes,
        train_rows=len(x_train),
        test_rows=len(x_test),
        train_fraud=int(y_train.sum()),
        test_fraud=int(y_test.sum()),
        dropped_target_rows=len(frame) - len(labeled),
        created_at=datetime.now(UTC).isoformat(),
        data_signature=hashlib.sha256(signature_bytes).hexdigest(),
    )


def threshold_curve(run: ModelRun) -> pd.DataFrame:
    """Evaluate operating metrics across decision thresholds."""
    rows = []
    for threshold in np.linspace(0.05, 0.95, 19):
        metrics, _ = evaluate_probabilities(run.y_test, run.probabilities, float(threshold))
        rows.append(
            {
                "Threshold": float(threshold),
                "Precision": metrics.precision,
                "Recall": metrics.recall,
                "F1": metrics.f1,
                "Specificity": metrics.specificity,
            }
        )
    return pd.DataFrame(rows)


def model_curve_data(run: ModelRun) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ROC and precision-recall curve points for one run."""
    if len(np.unique(run.y_test)) != 2:
        return pd.DataFrame(), pd.DataFrame()
    false_positive_rate, true_positive_rate, _ = roc_curve(run.y_test, run.probabilities)
    precision, recall, _ = precision_recall_curve(run.y_test, run.probabilities)
    roc_data = pd.DataFrame(
        {"False positive rate": false_positive_rate, "True positive rate": true_positive_rate}
    )
    pr_data = pd.DataFrame({"Recall": recall, "Precision": precision})
    return roc_data, pr_data


def feature_importance_table(run: ModelRun, *, limit: int = 30) -> pd.DataFrame:
    """Return coefficients or impurity importances from a fitted pipeline."""
    preprocessor = run.pipeline.named_steps["preprocessor"]
    estimator = run.pipeline.named_steps["model"]
    names = np.asarray(preprocessor.get_feature_names_out(), dtype=str)
    if hasattr(estimator, "coef_"):
        signed = np.asarray(estimator.coef_[0], dtype=float)
        importance = np.abs(signed)
    elif hasattr(estimator, "feature_importances_"):
        importance = np.asarray(estimator.feature_importances_, dtype=float)
        signed = importance
    else:
        return pd.DataFrame(columns=["Feature", "Importance", "Direction"])

    if len(names) != len(importance):
        raise TrainingError("Model feature names and importance values are inconsistent.")
    table = pd.DataFrame(
        {
            "Feature": names,
            "Importance": importance,
            "Direction": signed,
        }
    )
    return table.nlargest(limit, "Importance").reset_index(drop=True)


def serialize_model_bundle(
    result: TrainingResult,
    model_name: str,
    threshold: float,
) -> bytes:
    """Serialize a trusted local model pipeline with audit metadata."""
    if model_name not in result.runs:
        raise KeyError(f"Trained model not found: {model_name}")
    run = result.runs[model_name]
    metrics, _ = evaluate_probabilities(run.y_test, run.probabilities, threshold)
    bundle = {
        "artifact_type": "fraudshield_model_bundle",
        "artifact_version": 1,
        "created_at": result.created_at,
        "model_name": model_name,
        "model_display_name": run.display_name,
        "pipeline": run.pipeline,
        "target_column": result.target_column,
        "fraud_value": result.fraud_value,
        "feature_columns": result.feature_columns,
        "datetime_columns": result.datetime_columns,
        "decision_threshold": threshold,
        "holdout_metrics": asdict(metrics),
        "data_signature": result.data_signature,
        "versions": {
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    buffer = BytesIO()
    joblib.dump(bundle, buffer)
    return buffer.getvalue()


def save_model_bundle(
    result: TrainingResult,
    model_name: str,
    threshold: float,
    destination: Path,
) -> Path:
    """Write a model bundle to a user-approved local destination."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(serialize_model_bundle(result, model_name, threshold))
    return destination
