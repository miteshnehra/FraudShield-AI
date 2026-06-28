# Model Lab

Phase 4 trains and compares supervised fraud classifiers using an untouched stratified holdout set.

## Included Models

- Logistic Regression: interpretable linear baseline with balanced class weights
- Random Forest: nonlinear bagged-tree baseline with balanced subsamples
- Extra Trees: randomized tree ensemble for a second nonlinear comparison

These dependable scikit-learn baselines establish whether the dataset contains useful predictive signal. Optional XGBoost, LightGBM, or CatBoost adapters should be added only after the baseline pipeline and evaluation are stable.

## Leakage Controls

1. Rows without a target label are excluded.
2. The labeled data is split with stratification before fitting any imputer, scaler, encoder, or model.
3. Numeric imputation, scaling, categorical imputation, and one-hot encoding live inside each sklearn pipeline.
4. Cross-validation clones and refits the complete pipeline inside every fold.
5. The holdout set is used only for final probabilities and metrics.
6. Changing or cleaning the active dataset invalidates any model stored in the session.

## Feature Safety

- The fraud target is never offered as an input feature.
- High-cardinality identifiers, constant columns, and fully empty columns are excluded by default.
- Timestamps become hour, weekday, day-of-month, month, and weekend features inside the pipeline.
- Unknown categorical values are handled without failing inference.

## Evaluation

Model ranking prioritizes holdout PR-AUC, followed by F1 and recall. The dashboard also reports precision, ROC-AUC, accuracy, cross-validation F1, cross-validation PR-AUC, false alerts, and missed fraud.

The threshold view is an operating-policy tool. Lowering the threshold generally increases recall and false alerts; raising it generally increases precision and missed fraud. Threshold selection must reflect investigation cost and fraud-loss tolerance.

## Model Artifacts

The exported Joblib bundle contains the full feature builder, preprocessing pipeline, estimator, threshold, feature schema, holdout metrics, data signature, and library versions.

Joblib uses Python pickle internally and can execute code during loading. Load only trusted artifacts created by this project.

## Limitations

- A small sample can demonstrate workflow but cannot support performance claims.
- Random splitting is not suitable when deployment predicts strictly future transactions; a time-based validation strategy should then replace it.
- Holdout results do not replace production drift monitoring, fairness review, access controls, or human investigation.

