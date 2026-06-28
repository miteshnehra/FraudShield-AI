"""Sklearn integration tests that run when model dependencies are installed."""

import unittest

import pandas as pd

try:
    from fraudshield.modeling.training import (
        TrainingConfig,
        evaluate_probabilities,
        feature_importance_table,
        serialize_model_bundle,
        train_models,
    )
    from fraudshield.risk.explanations import explain_transaction

    SKLEARN_AVAILABLE = True
except ModuleNotFoundError:
    SKLEARN_AVAILABLE = False


@unittest.skipUnless(SKLEARN_AVAILABLE, "scikit-learn is not installed in this runtime")
class TrainingIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        rows = []
        for index in range(80):
            is_fraud = 1 if index % 8 == 0 else 0
            rows.append(
                {
                    "timestamp": f"2026-06-{(index % 20) + 1:02d} {index % 24:02d}:00:00",
                    "amount": 5000.0 + index if is_fraud else 50.0 + index,
                    "merchant": "risk" if is_fraud else f"safe_{index % 3}",
                    "velocity": 8 if is_fraud else 1,
                    "Class": is_fraud,
                }
            )
        self.frame = pd.DataFrame(rows)

    def test_trains_evaluates_and_serializes_pipeline(self) -> None:
        config = TrainingConfig(
            model_names=("logistic_regression",),
            run_cross_validation=False,
            random_state=7,
        )

        result = train_models(
            self.frame,
            "Class",
            1,
            ("timestamp", "amount", "merchant", "velocity"),
            ("timestamp",),
            config,
        )

        run = result.runs["logistic_regression"]
        self.assertGreater(result.train_rows, result.test_rows)
        self.assertEqual(len(run.probabilities), result.test_rows)
        self.assertFalse(feature_importance_table(run).empty)
        self.assertGreater(len(serialize_model_bundle(result, run.model_name, 0.5)), 100)
        explanation = explain_transaction(self.frame, result, run.model_name, 0)
        self.assertFalse(explanation.contributions.empty)

    def test_threshold_changes_operating_predictions(self) -> None:
        probabilities = [0.2, 0.4, 0.6, 0.8]
        labels = [0, 0, 1, 1]

        metrics, predictions = evaluate_probabilities(labels, probabilities, 0.7)

        self.assertEqual(predictions.tolist(), [0, 0, 0, 1])
        self.assertEqual(metrics.false_negatives, 1)


if __name__ == "__main__":
    unittest.main()
