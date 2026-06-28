"""Tests for transparent batch risk scoring."""

import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from fraudshield.risk.scoring import (
    RiskThresholds,
    risk_band,
    risk_band_counts,
    risk_summary,
    score_transactions,
)


class FakePipeline:
    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        probabilities = np.asarray(frame["probability"], dtype=float)
        return np.column_stack((1 - probabilities, probabilities))


class RiskScoringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.thresholds = RiskThresholds()
        run = SimpleNamespace(pipeline=FakePipeline(), display_name="Test Model")
        self.result = SimpleNamespace(
            runs={"test": run},
            feature_columns=("probability", "amount"),
        )
        self.frame = pd.DataFrame(
            {
                "probability": [0.05, 0.35, 0.70, 0.92],
                "amount": [10, 20, 30, 40],
            },
            index=[10, 11, 12, 13],
        )

    def test_risk_band_boundaries(self) -> None:
        self.assertEqual(risk_band(0, self.thresholds), "Low")
        self.assertEqual(risk_band(29, self.thresholds), "Low")
        self.assertEqual(risk_band(30, self.thresholds), "Medium")
        self.assertEqual(risk_band(60, self.thresholds), "High")
        self.assertEqual(risk_band(80, self.thresholds), "Critical")
        self.assertEqual(risk_band(100, self.thresholds), "Critical")

    def test_scores_transactions_without_mutating_source(self) -> None:
        snapshot = self.frame.copy(deep=True)

        scored = score_transactions(
            self.frame,
            self.result,
            "test",
            0.5,
            self.thresholds,
        ).frame

        pd.testing.assert_frame_equal(self.frame, snapshot)
        self.assertEqual(scored["FS_Risk_Band"].tolist(), ["Low", "Medium", "High", "Critical"])
        self.assertEqual(
            scored["FS_Decision"].tolist(),
            ["No automatic hold", "No automatic hold", "Manual review", "Manual review"],
        )
        self.assertEqual(scored["FS_Source_Index"].tolist(), ["10", "11", "12", "13"])

    def test_risk_band_counts_have_stable_order(self) -> None:
        scored = score_transactions(
            self.frame,
            self.result,
            "test",
            0.5,
            self.thresholds,
        ).frame

        counts = risk_band_counts(scored)

        self.assertEqual(counts["Risk band"].tolist(), ["Low", "Medium", "High", "Critical"])
        self.assertEqual(counts["Transactions"].tolist(), [1, 1, 1, 1])

    def test_rejects_invalid_score_and_threshold_policy(self) -> None:
        with self.assertRaises(ValueError):
            risk_band(101, self.thresholds)
        with self.assertRaises(ValueError):
            RiskThresholds(low_max=40, medium_max=30)

    def test_summary_never_claims_transaction_is_safe(self) -> None:
        self.assertNotIn("safe", risk_summary("Low").lower())


if __name__ == "__main__":
    unittest.main()

