# Testing Guide

## Commands

```bash
pytest
ruff check .
python -m compileall -q app.py src tests
```

## Test Layers

| Layer | Coverage |
|---|---|
| Configuration | Required sections and runtime paths |
| CSV intake | Extension, delimiter, encoding, empty-file rejection |
| Cleaning | Missing values, duplicates, formulas, immutability |
| EDA | Target inference, fraud rates, time trends, correlations |
| Feature schema | Leakage exclusions and timestamp derivation |
| Model integration | Training, thresholding, importance, serialization |
| Risk scoring | Scores, bands, decisions, stable counts |
| Explanations | Signed reason text and local contributions |
| Reporting | PDF generation, sections, page count, privacy boundary |

## Manual Release Test

1. Upload `examples/sample_transactions.csv`.
2. Confirm 15 rows, 11 columns, 3 missing cells, and 1 duplicate.
3. Run cleaning and confirm the original remains restorable.
4. Open Fraud Analysis and confirm `Class`, `amount`, and `timestamp` suggestions.
5. Train Logistic Regression only for a quick smoke test.
6. Review confusion matrix and threshold behavior.
7. Score the active dataset and filter the risk queue.
8. Generate one local explanation.
9. Generate and open the executive PDF.
10. Confirm a new upload invalidates models, scores, explanations, and reports.

The sample dataset validates workflow only. It is too small for performance claims.

