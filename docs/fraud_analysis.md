# Fraud Analysis

Phase 3 adds label-aware exploratory analysis for the active Data Center dataset.

## Required Input

The dataset must contain a target column with exactly two non-null values, such as `0/1`, `safe/fraud`, or `false/true`. The user confirms which value represents fraud.

Amount and timestamp columns are optional. Fraud prevalence can still be analyzed without them.

## Available Analysis

- Valid, missing-label, legitimate, and fraud transaction counts
- Fraud transaction rate
- Total and fraud-linked amount exposure
- Category-level transaction counts and fraud rates
- Fraud and legitimate amount distributions
- Calendar trend and hour-of-day analysis
- Spearman numeric correlation matrix
- Ranked feature associations with a consistently oriented fraud indicator

## Interpretation Guardrails

- Correlation is association, not causation.
- Category rates based on small counts are unstable; use the minimum-transaction filter.
- Fewer than ten fraud rows triggers a small-sample warning.
- Strongly imbalanced data requires precision, recall, F1, PR-AUC, and ROC-AUC rather than accuracy alone.
- EDA must be completed before splitting model data, but learned preprocessing in Phase 4 must fit on training data only to prevent leakage.

## Privacy

Analysis reads the in-session active dataset and does not modify or persist it. The fraud-row preview is limited to 100 records.

