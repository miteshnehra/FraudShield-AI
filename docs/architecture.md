# FraudShield AI Architecture

FraudShield AI is organized as a modular Python application with a Streamlit operations console.

## Runtime Layers

1. **Interface:** Streamlit pages for data operations, analysis, model training, risk review, and reports.
2. **Services:** Dataset validation, cleaning, feature engineering, prediction, and explanation services.
3. **Modeling:** Reproducible training pipelines, evaluation, threshold selection, and saved model artifacts.
4. **Storage:** Local project directories during development, with database/object storage adapters planned for deployment.
5. **Observability:** Shared configuration, structured application logs, validation errors, and test coverage.

## Safety Boundary

The platform is decision support. It produces probabilities, risk bands, and explanations. It does not automatically block transactions in the portfolio version. Production use would require access control, audit trails, drift monitoring, privacy controls, and a human-review workflow.

## Development Phases

| Phase | Scope | Exit condition |
|---|---|---|
| 1 | Foundation | App, settings, logging, and tests run locally |
| 2 | Data Center | Complete: CSV upload, quality checks, and cleaning work |
| 3 | Analysis | Complete: fraud-focused EDA renders from validated data |
| 4 | Model Lab | Complete: baselines train and compare without leakage |
| 5 | Risk Review | Complete: probabilities, thresholds, reasons, and SHAP work |
| 6 | Reporting | Reports, documentation, and deployment are verified |
