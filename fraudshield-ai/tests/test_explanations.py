"""Tests for cautious human-readable local explanation text."""

import unittest

import pandas as pd

from fraudshield.risk.explanations import LocalExplanation, human_readable_reasons


class ExplanationTextTests(unittest.TestCase):
    def test_generates_positive_and_negative_reasons(self) -> None:
        contributions = pd.DataFrame(
            [
                {
                    "Feature": "amount",
                    "Readable feature": "Amount",
                    "Contribution": 1.2,
                    "Impact": 1.2,
                    "Direction": "Raises risk",
                },
                {
                    "Feature": "account_age",
                    "Readable feature": "Account Age",
                    "Contribution": -0.8,
                    "Impact": 0.8,
                    "Direction": "Lowers risk",
                },
            ]
        )
        explanation = LocalExplanation("test", 0.0, contributions)

        reasons = human_readable_reasons(explanation)

        self.assertIn("Amount increased the model's fraud signal.", reasons)
        self.assertIn("Account Age reduced the model's fraud signal.", reasons)


if __name__ == "__main__":
    unittest.main()

