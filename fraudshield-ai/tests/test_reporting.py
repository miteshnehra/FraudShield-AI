"""Tests for executive PDF content and privacy boundaries."""

import unittest
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from pypdf import PdfReader

from fraudshield.data.loader import load_csv_bytes
from fraudshield.reporting.pdf_report import build_executive_report
from fraudshield.risk.scoring import RiskThresholds, ScoringResult


class ExecutiveReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path = Path("examples/sample_transactions.csv")
        cls.frame = load_csv_bytes(path.read_bytes(), path.name).frame

    def _training_result(self) -> SimpleNamespace:
        leaderboard = pd.DataFrame(
            [
                {
                    "Model": "Logistic Regression",
                    "Precision": 0.80,
                    "Recall": 0.75,
                    "F1": 0.774,
                    "PR-AUC": 0.82,
                    "ROC-AUC": 0.91,
                }
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

    def _scoring_result(self) -> ScoringResult:
        scored = self.frame.copy()
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

    def test_generates_readable_multi_page_report(self) -> None:
        pdf_bytes = build_executive_report(
            self.frame,
            app_version="1.0.0",
            dataset_name="sample_transactions.csv",
            training_result=self._training_result(),
            scoring_result=self._scoring_result(),
            generated_at=datetime(2026, 6, 22, 12, 0, tzinfo=UTC),
        )

        self.assertTrue(pdf_bytes.startswith(b"%PDF"))
        reader = PdfReader(BytesIO(pdf_bytes))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        self.assertGreaterEqual(len(reader.pages), 4)
        self.assertIn("Executive Fraud Risk Report", text)
        self.assertIn("Model Evaluation", text)
        self.assertIn("Investigation Queue", text)
        self.assertIn("Page 1 of", text)
        self.assertNotIn("jewelry", text)

    def test_report_works_before_model_training(self) -> None:
        pdf_bytes = build_executive_report(
            self.frame,
            app_version="1.0.0",
            generated_at=datetime(2026, 6, 22, 12, 0, tzinfo=UTC),
        )

        reader = PdfReader(BytesIO(pdf_bytes))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        self.assertIn("Data Quality", text)
        self.assertNotIn("Model Evaluation", text)


if __name__ == "__main__":
    unittest.main()

