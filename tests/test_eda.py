"""Tests for fraud-focused exploratory analysis calculations."""

import unittest

import pandas as pd

from fraudshield.analysis.eda import (
    categorical_fraud_rates,
    correlation_matrix,
    infer_amount_columns,
    infer_binary_target_columns,
    infer_categorical_columns,
    infer_datetime_columns,
    infer_fraud_value,
    profile_target,
    strongest_fraud_correlations,
    time_fraud_trend,
)


class FraudEdaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = pd.DataFrame(
            {
                "transaction_id": ["T1", "T2", "T3", "T4", "T5"],
                "timestamp": [
                    "2026-06-01 10:00:00",
                    "2026-06-01 11:00:00",
                    "2026-06-02 10:00:00",
                    "2026-06-02 12:00:00",
                    "2026-06-02 13:00:00",
                ],
                "amount": [10.0, 20.0, 100.0, 120.0, 80.0],
                "merchant": ["A", "A", "B", "B", "B"],
                "is_foreign": [0, 0, 1, 1, 1],
                "Class": [0, 0, 1, 1, 0],
            }
        )

    def test_infers_common_column_roles(self) -> None:
        targets = infer_binary_target_columns(self.frame)
        amounts = infer_amount_columns(self.frame, exclude=("Class",))
        datetimes = infer_datetime_columns(self.frame)
        categories = infer_categorical_columns(self.frame, exclude=("Class", "timestamp"))

        self.assertEqual(targets[0], "Class")
        self.assertEqual(amounts[0], "amount")
        self.assertIn("timestamp", datetimes)
        self.assertIn("merchant", categories)
        self.assertIn("is_foreign", categories)

    def test_infers_positive_fraud_value(self) -> None:
        self.assertEqual(infer_fraud_value([0, 1]), 1)
        self.assertEqual(infer_fraud_value(["safe", "fraud"]), "fraud")

    def test_profiles_transaction_and_amount_exposure(self) -> None:
        profile = profile_target(self.frame, "Class", 1, "amount")

        self.assertEqual(profile.valid_transactions, 5)
        self.assertEqual(profile.fraud_transactions, 2)
        self.assertEqual(profile.legitimate_transactions, 3)
        self.assertEqual(profile.fraud_rate, 40.0)
        self.assertEqual(profile.total_amount, 330.0)
        self.assertEqual(profile.fraud_amount, 220.0)

    def test_calculates_category_fraud_rates(self) -> None:
        rates = categorical_fraud_rates(self.frame, "Class", 1, "merchant")

        merchant_b = rates.loc[rates["Category"] == "B"].iloc[0]
        self.assertEqual(int(merchant_b["Transactions"]), 3)
        self.assertEqual(int(merchant_b["Fraud"]), 2)
        self.assertAlmostEqual(float(merchant_b["Fraud rate %"]), 66.6667, places=3)

    def test_builds_daily_fraud_trend(self) -> None:
        trend = time_fraud_trend(self.frame, "Class", 1, "timestamp", period="day")

        self.assertEqual(len(trend), 2)
        self.assertEqual(int(trend["Transactions"].sum()), 5)
        self.assertEqual(int(trend["Fraud"].sum()), 2)

    def test_correlation_is_oriented_to_fraud_label(self) -> None:
        matrix = correlation_matrix(self.frame, "Class", 1)
        strongest = strongest_fraud_correlations(matrix)

        self.assertIn("Fraud indicator", matrix.columns)
        self.assertIn("amount", strongest["Feature"].tolist())
        self.assertGreater(
            float(strongest.loc[strongest["Feature"] == "amount", "Correlation"].iloc[0]),
            0,
        )

    def test_missing_target_rows_are_excluded(self) -> None:
        frame = self.frame.copy()
        frame.loc[0, "Class"] = None

        profile = profile_target(frame, "Class", 1)

        self.assertEqual(profile.valid_transactions, 4)
        self.assertEqual(profile.missing_target, 1)


if __name__ == "__main__":
    unittest.main()

