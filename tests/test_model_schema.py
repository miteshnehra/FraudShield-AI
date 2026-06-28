"""Tests for model feature planning and deterministic time features."""

import unittest

import pandas as pd

from fraudshield.modeling.schema import (
    feature_schema_table,
    prepare_feature_frame,
    suggest_feature_plan,
    validate_feature_columns,
)


class FeatureSchemaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = pd.DataFrame(
            {
                "transaction_id": ["T1", "T2", "T3", "T4"],
                "timestamp": [
                    "2026-06-01 10:00:00",
                    "2026-06-02 11:00:00",
                    None,
                    "2026-06-07 20:00:00",
                ],
                "amount": [10.0, 20.0, None, 100.0],
                "merchant": ["A", "B", "A", "C"],
                "constant": [1, 1, 1, 1],
                "empty": [None, None, None, None],
                "Class": [0, 0, 1, 1],
            }
        )

    def test_plan_excludes_target_identifiers_and_unusable_columns(self) -> None:
        plan = suggest_feature_plan(self.frame, "Class")

        self.assertIn("amount", plan.recommended)
        self.assertIn("merchant", plan.recommended)
        self.assertIn("timestamp", plan.datetime_columns)
        self.assertNotIn("transaction_id", plan.recommended)
        self.assertNotIn("constant", plan.recommended)
        self.assertNotIn("empty", plan.recommended)
        self.assertNotIn("Class", plan.recommended)

    def test_prepares_time_parts_without_raw_timestamp(self) -> None:
        prepared = prepare_feature_frame(
            self.frame,
            ("timestamp", "amount", "merchant"),
            ("timestamp",),
        )

        self.assertNotIn("timestamp", prepared.columns)
        self.assertIn("timestamp__hour", prepared.columns)
        self.assertIn("timestamp__is_weekend", prepared.columns)
        self.assertEqual(float(prepared.loc[0, "timestamp__hour"]), 10.0)
        self.assertEqual(float(prepared.loc[3, "timestamp__is_weekend"]), 1.0)
        self.assertTrue(pd.isna(prepared.loc[2, "timestamp__hour"]))

    def test_rejects_target_as_feature(self) -> None:
        with self.assertRaises(ValueError):
            validate_feature_columns(self.frame, "Class", ("amount", "Class"))

    def test_schema_table_explains_recommendations(self) -> None:
        plan = suggest_feature_plan(self.frame, "Class")

        table = feature_schema_table(self.frame, plan)

        target_row = table.loc[table["Column"] == "Class"].iloc[0]
        self.assertEqual(target_row["Recommendation"], "Excluded")
        self.assertEqual(target_row["Reason"], "Fraud target")


if __name__ == "__main__":
    unittest.main()

