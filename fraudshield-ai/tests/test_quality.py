"""Tests for dataset quality measurements."""

import unittest

import pandas as pd

from fraudshield.data.quality import missing_value_table, profile_dataset


class QualityProfileTests(unittest.TestCase):
    def test_profiles_missing_duplicates_and_constant_columns(self) -> None:
        frame = pd.DataFrame(
            [
                {"amount": 10.0, "merchant": "Alpha", "empty": None},
                {"amount": 10.0, "merchant": "Alpha", "empty": None},
                {"amount": None, "merchant": "Alpha", "empty": None},
            ]
        )

        report = profile_dataset(frame)

        self.assertEqual(report.row_count, 3)
        self.assertEqual(report.missing_cells, 4)
        self.assertEqual(report.duplicate_rows, 1)
        self.assertEqual(set(report.constant_columns), {"merchant", "empty"})
        self.assertLess(report.quality_score, 100)

    def test_detects_formula_like_text(self) -> None:
        frame = pd.DataFrame({"merchant": ["Alpha", "=CMD()", "+SUM(1,2)"]})

        report = profile_dataset(frame)

        self.assertEqual(report.formula_like_cells, 2)

    def test_missing_table_is_sorted_by_missing_count(self) -> None:
        frame = pd.DataFrame({"complete": [1, 2], "missing": [None, 2]})

        table = missing_value_table(frame)

        self.assertEqual(table.iloc[0]["Column"], "missing")
        self.assertEqual(int(table.iloc[0]["Missing"]), 1)


if __name__ == "__main__":
    unittest.main()

