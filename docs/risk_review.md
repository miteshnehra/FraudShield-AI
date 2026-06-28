# Risk Review

Phase 5 turns model probabilities into a transparent investigation queue. It does not authorize automatic declines, account freezes, or accusations of fraud.

## Score Definition

`Risk score = predicted fraud probability x 100`

The score is not a separate hidden formula. Default operational bands are:

| Score | Band | Suggested handling |
|---|---|---|
| 0-29 | Low | Continue normal monitoring |
| 30-59 | Medium | Monitor and apply business rules |
| 60-79 | High | Prioritize manual review |
| 80-100 | Critical | Investigate promptly before action |

The manual-review threshold is independent from these descriptive bands. Investigators can tune it to balance missed fraud against false alerts.

## Queue Output

Each scored row includes:

- Original row number and source index
- Fraud probability
- 0-100 risk score
- Risk band
- Manual-review or no-automatic-hold decision
- Model display name

The queue is sortable, filterable, limited to 500 displayed rows, and downloadable in full as CSV.

## Local Explanations

- Logistic Regression uses exact signed feature contributions in log-odds space.
- Random Forest and Extra Trees use Tree SHAP contributions.
- Positive contributions raise the model fraud signal; negative contributions lower it.
- Human-readable reasons are generated only from the largest signed contributions.

SHAP explains the selected model, not the real-world cause of fraud. A strong contribution can reflect correlation, proxy variables, data quality problems, or historical bias.

## Evaluation Boundary

Risk Review can score the same active rows used during training. These scores are operational demonstrations and may be optimistic. Only untouched holdout or properly designed time-based validation metrics belong in model-performance reporting.

## Safety

- No transaction is labeled safe.
- No transaction is blocked automatically.
- No risk score is treated as proof of fraud.
- Human review and documented business rules remain mandatory.
- Changing the active dataset or retraining invalidates old scores and explanations.

