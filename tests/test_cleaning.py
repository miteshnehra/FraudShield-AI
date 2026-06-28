"""Tests for the reversible cleaning pipeline."""

import unittest

import pandas as pd

from fraudshield.data.cleaning import CleaningOptions, clean_dataset


class CleaningPipelineTests(unittest.TestCase):
    def test_cleans_without_mutating_original(self) -> None:
        original = pd.DataFrame(
            [
                {" Amount ": 10.0, "Merchant Name": " Alpha ", "Blank": None},
                {" Amount ": 10.0, "Merchant Name": " Alpha ", "Blank": None},
                {" Amount ": None, "Merchant Name": " ", "Blank": None},
            ]
        )
        snapshot = original.copy(deep=True)
        options = CleaningOptions(
            drop_empty_rows=False,
            normalize_column_names=True,
            numeric_missing="median",
            categorical_missing="unknown",
        )

        result = clean_dataset(original, options)

        pd.testing.assert_frame_equal(original, snapshot)
        self.assertEqual(result.cleaned_shape, (2, 2))
        self.assertEqual(list(result.frame.columns), ["amount", "merchant_name"])
        self.assertEqual(float(result.frame.loc[1, "amount"]), 10.0)
        self.assertEqual(result.frame.loc[1, "merchant_name"], "Unknown")

    def test_neutralizes_spreadsheet_formulas(self) -> None:
        frame = pd.DataFrame({"note": ["safe", "=CMD()", "@SUM(1,2)"]})

        result = clean_dataset(frame, CleaningOptions(neutralize_formulas=True))

        self.assertEqual(result.frame.loc[1, "note"], "'=CMD()")
        self.assertEqual(result.frame.loc[2, "note"], "'@SUM(1,2)")

    def test_rejects_unknown_imputation_strategy(self) -> None:
        frame = pd.DataFrame({"amount": [1.0, None]})

        with self.assertRaises(ValueError):
            clean_dataset(frame, CleaningOptions(numeric_missing="magic"))


if __name__ == "__main__":
    unittest.main()
