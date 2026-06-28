"""Generate the repository's synthetic sample executive report."""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from fraudshield.data.loader import load_csv_bytes  # noqa: E402
from fraudshield.reporting.pdf_report import build_executive_report  # noqa: E402
from fraudshield.risk.scoring import RiskThresholds, ScoringResult  # noqa: E402


def sample_training_result() -> SimpleNamespace:
    leaderboard = pd.DataFrame(
        [
            {
                "Model": "Logistic Regression",
                "Precision": 0.80,
                "Recall": 0.75,
                "F1": 0.774,
                "PR-AUC": 0.82,
                "ROC-AUC": 0.91,
            },
            {
                "Model": "Random Forest",
                "Precision": 0.76,
                "Recall": 0.70,
                "F1": 0.729,
                "PR-AUC": 0.79,
                "ROC-AUC": 0.89,
            },
            {
                "Model": "Extra Trees",
                "Precision": 0.72,
                "Recall": 0.78,
                "F1": 0.749,
                "PR-AUC": 0.77,
                "ROC-AUC": 0.88,
            },
        ]
    )
    return SimpleNamespace(
        runs={"logistic_regression": SimpleNamespace(display_name="Logistic Regression")},
        leaderboard=leaderboard,
        best_model_name="logistic_regression",
        train_rows=12,
        test_rows=3,
        test_fraud=1,
    )


def sample_scoring_result(frame: pd.DataFrame) -> ScoringResult:
    scored = frame.copy()
    scored.insert(0, "FS_Source_Index", scored.index.map(str))
    scored.insert(0, "FS_Row_Number", range(len(scored)))
    scored["FS_Risk_Score"] = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 15, 25, 35, 65, 85]
    scored["FS_Risk_Band"] = [
        "Low",
        "Low",
        "Medium",
        "Medium",
        "Medium",
        "High",
        "High",
        "Critical",
        "Critical",
        "Critical",
        "Low",
        "Low",
        "Medium",
        "High",
        "Critical",
    ]
    scored["FS_Decision"] = [
        "Manual review" if score >= 50 else "No automatic hold"
        for score in scored["FS_Risk_Score"]
    ]
    return ScoringResult(
        frame=scored,
        model_name="logistic_regression",
        model_display_name="Logistic Regression",
        decision_threshold=0.50,
        thresholds=RiskThresholds(),
    )


def main() -> None:
    source = PROJECT_ROOT / "examples" / "sample_transactions.csv"
    frame = load_csv_bytes(source.read_bytes(), source.name).frame
    output = PROJECT_ROOT / "output" / "pdf" / "fraudshield_sample_executive_report.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    report = build_executive_report(
        frame,
        app_version="1.0.0",
        dataset_name=source.name,
        training_result=sample_training_result(),
        scoring_result=sample_scoring_result(frame),
        generated_at=datetime(2026, 6, 22, 12, 0, tzinfo=UTC),
    )
    output.write_bytes(report)
    print(output)


if __name__ == "__main__":
    main()

