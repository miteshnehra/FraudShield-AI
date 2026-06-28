# FraudShield AI - 100 Viva Questions and Answers

## Fundamentals

1. **What is FraudShield AI?**  
It is a decision-support application that analyzes transaction data, trains fraud classifiers, produces risk scores, and helps prioritize manual review.

2. **What problem does it solve?**  
It reduces the amount of transaction data investigators must inspect manually by ranking suspicious transactions.

3. **Is the project an automatic blocking system?**  
No. It recommends manual review and never automatically blocks a transaction.

4. **Why is human review required?**  
Models can make false predictions, reflect biased data, or miss context that an investigator understands.

5. **What is supervised learning?**  
It is learning a relationship between input features and known labels, such as legitimate or fraud.

6. **What is binary classification?**  
It is a prediction task with two classes. Here the classes are legitimate and fraud.

7. **What is a feature?**  
A feature is an input variable used by a model, such as transaction amount or account age.

8. **What is a target?**  
The target is the outcome the model learns to predict, such as the `Class` fraud label.

9. **Why is fraud detection difficult?**  
Fraud is rare, patterns change, labels can be delayed, and fraudulent behavior can resemble legitimate behavior.

10. **What is class imbalance?**  
It means one class has far fewer examples than the other, which is common because fraud is usually rare.

## Data Intake and Quality

11. **Which file type does v1.0 accept?**  
It accepts CSV transaction datasets.

12. **How are delimiters detected?**  
Python's CSV sniffer checks common delimiters such as comma, semicolon, tab, and pipe.

13. **Which encodings are supported?**  
UTF-8 with BOM, Windows-1252, and Latin-1 are attempted in order.

14. **Why sanitize filenames?**  
It prevents path components in an uploaded filename from being treated as local filesystem paths.

15. **What is a missing value?**  
It is a cell where the expected data is absent or unknown.

16. **What is a duplicate row?**  
It is an additional row with the same values as an earlier row.

17. **What is a constant column?**  
It is a column with only one distinct value and therefore no predictive variation.

18. **How is the quality score calculated?**  
It uses 50 points for completeness, 30 for row uniqueness, and 20 for non-constant columns.

19. **Is the quality score model accuracy?**  
No. It describes dataset hygiene, not prediction performance.

20. **Why detect formula-like CSV cells?**  
Spreadsheet applications may execute text beginning with formula characters, creating a CSV-injection risk.

## Cleaning

21. **Why is cleaning reversible?**  
The pipeline works on a deep copy so the original dataset can be restored.

22. **What is imputation?**  
Imputation fills missing values using a chosen value or statistic.

23. **Which numeric imputation methods exist?**  
Median, mean, zero, or keeping the values missing.

24. **Which categorical imputation methods exist?**  
Most frequent value, `Unknown`, or keeping the values missing.

25. **Why is median often safer than mean?**  
Median is less affected by extreme transaction amounts.

26. **Why can imputation be dangerous?**  
It can hide meaningful missingness or introduce artificial patterns.

27. **Why trim text whitespace?**  
Values such as `Visa` and ` Visa ` should not become separate categories.

28. **Why normalize column names?**  
It creates consistent lowercase underscore-separated names for code and analysis.

29. **When are fully empty rows removed?**  
After blank text is converted to missing values, allowing truly empty rows to be detected.

30. **Does cleaning modify the uploaded DataFrame directly?**  
No. It creates and returns a cleaned copy.

## Exploratory Analysis

31. **What is EDA?**  
Exploratory Data Analysis is the process of understanding distributions, relationships, and data problems before modeling.

32. **How is the fraud target suggested?**  
Binary columns are detected and common names such as `Class`, `is_fraud`, and `target` are prioritized.

33. **How is the fraud label suggested?**  
Values such as `1`, `true`, `fraud`, and `yes` are prioritized as the positive class.

34. **What is fraud rate?**  
It is fraud transactions divided by valid labeled transactions, multiplied by 100.

35. **What is fraud amount exposure?**  
It is the share of total transaction amount associated with fraud-labeled rows.

36. **Why filter categories by minimum count?**  
Fraud rates from very small categories are unstable and can be misleading.

37. **Why analyze time patterns?**  
Fraud frequency and risk can change by hour, day, or period.

38. **Why use Spearman correlation?**  
It measures monotonic relationships and is less dependent on linearity than Pearson correlation.

39. **Does correlation prove causation?**  
No. It only measures statistical association.

40. **Why exclude transaction IDs from category analysis?**  
Nearly unique identifiers create meaningless categories and can mislead charts and models.

## Feature Engineering and Leakage

41. **What is feature engineering?**  
It is transforming raw data into model-ready variables.

42. **How are timestamps engineered?**  
They become hour, weekday, day of month, month, and weekend indicators.

43. **Why not one-hot encode raw timestamps?**  
Most timestamps are unique, creating huge sparse features with poor generalization.

44. **What is data leakage?**  
Leakage occurs when training uses information that would not be available during real prediction or comes from validation data.

45. **How does the project prevent preprocessing leakage?**  
Imputation, scaling, and encoding are inside pipelines fitted only on training folds.

46. **Why split before fitting an imputer?**  
Statistics calculated from the test set would leak test information into training.

47. **Why exclude the target from features?**  
Including it would give the model the answer and produce invalid performance.

48. **Why exclude high-cardinality identifiers?**  
They encourage memorization and usually do not generalize to new transactions.

49. **What happens to unknown categories at prediction time?**  
The one-hot encoder handles them without failing the pipeline.

50. **Why is a complete pipeline useful?**  
It keeps transformations and the estimator reproducible and consistent during training and prediction.

## Models

51. **Why use Logistic Regression?**  
It is a fast, interpretable baseline that produces probabilities and signed coefficients.

52. **Why use Random Forest?**  
It captures nonlinear interactions using many decision trees and is robust on mixed tabular patterns.

53. **Why use Extra Trees?**  
It adds stronger split randomization and provides another nonlinear ensemble baseline.

54. **What is class weight?**  
It gives minority fraud examples more influence during model fitting.

55. **Why not use SMOTE by default?**  
Class weights are simpler and avoid creating synthetic records; incorrect SMOTE placement can also cause leakage.

56. **What is an estimator?**  
It is a fitted algorithm that learns from data and makes predictions.

57. **What does `predict_proba` return?**  
It returns estimated probabilities for each class.

58. **What is a random state?**  
It is a seed that makes splits and randomized models reproducible.

59. **Why compare multiple models?**  
Different algorithms capture different relationships, so comparison reveals whether complexity adds value.

60. **Why establish baselines before XGBoost?**  
Strong simple baselines show whether advanced models actually improve performance enough to justify complexity.

## Evaluation

61. **What is a holdout set?**  
It is labeled data excluded from training and used for final evaluation.

62. **Why stratify the split?**  
Stratification preserves the fraud proportion in training and holdout sets.

63. **What is accuracy?**  
It is correct predictions divided by all predictions.

64. **Why can accuracy mislead in fraud detection?**  
A model predicting every transaction as legitimate can have high accuracy when fraud is rare.

65. **What is precision?**  
It is true fraud alerts divided by all predicted fraud alerts.

66. **What is recall?**  
It is detected fraud divided by all actual fraud.

67. **What is F1 score?**  
It is the harmonic mean of precision and recall.

68. **What is ROC-AUC?**  
It measures how well scores rank positive above negative cases across thresholds.

69. **What is PR-AUC?**  
It summarizes precision-recall performance and is especially useful for imbalanced fraud data.

70. **Why rank primarily by PR-AUC?**  
It focuses on fraud detection quality without being dominated by the large legitimate class.

71. **What is a confusion matrix?**  
It counts true negatives, false positives, false negatives, and true positives.

72. **What is a false positive?**  
It is a legitimate transaction incorrectly sent to fraud review.

73. **What is a false negative?**  
It is a fraud transaction the model fails to flag.

74. **What is cross-validation?**  
It repeats training and validation across multiple folds to estimate stability.

75. **Does cross-validation replace a final holdout?**  
No. A separate untouched holdout is still needed for final evaluation.

## Thresholds and Risk

76. **What is a decision threshold?**  
It is the probability boundary used to send a transaction to manual review.

77. **What happens when the threshold is lowered?**  
Recall usually rises, but false alerts usually increase.

78. **What happens when the threshold is raised?**  
Precision may rise, but more fraud may be missed.

79. **How is the risk score calculated?**  
It is predicted fraud probability multiplied by 100.

80. **Are risk bands learned by the model?**  
No. They are transparent operational ranges configured by the application.

81. **What are the default bands?**  
Low 0-29, Medium 30-59, High 60-79, and Critical 80-100.

82. **Does a Critical score prove fraud?**  
No. It means the model signal is high and investigation should be prioritized.

83. **Why separate bands from the review threshold?**  
Bands communicate severity, while the threshold controls queue capacity and operating policy.

84. **Why are full-dataset scores not evaluation metrics?**  
They may include training rows and therefore can be optimistic.

85. **What happens to scores after retraining?**  
They are invalidated and must be regenerated with the new model.

## Explainability

86. **What is explainable AI?**  
It provides information about which features influenced a model output.

87. **How is Logistic Regression explained?**  
The project multiplies transformed feature values by learned coefficients to get exact local log-odds contributions.

88. **What is SHAP?**  
SHAP assigns each feature a contribution based on Shapley-value principles.

89. **Which models use Tree SHAP here?**  
Random Forest and Extra Trees.

90. **What does a positive SHAP value mean?**  
It pushes the selected transaction toward a higher fraud model output.

91. **What does a negative contribution mean?**  
It pushes the model output toward lower fraud risk.

92. **Does SHAP prove why fraud happened?**  
No. It explains model behavior, not real-world causation.

93. **Why show only top contributions?**  
It keeps investigator explanations concise and focused on the strongest model signals.

## Reporting, Security, and Deployment

94. **What is included in the executive PDF?**  
Data quality, fraud profile, model evaluation, risk-band counts, top queue identifiers, governance, and readiness checks.

95. **Why exclude raw transactions from the PDF?**  
It reduces unnecessary exposure of potentially sensitive data.

96. **Why are Joblib files risky?**  
They use Python pickle and can execute malicious code when an untrusted file is loaded.

97. **Which deployment options are documented?**  
Local execution, Docker, Streamlit Community Cloud, and Render.

98. **Why is the public demo not production-ready?**  
It lacks authentication, durable audit storage, formal privacy controls, monitoring, and incident response.

99. **What tests are included?**  
Configuration, loading, cleaning, EDA, feature schema, model integration, risk scoring, explanations, and PDF reporting tests.

100. **What is the main learning outcome?**  
Responsible fraud ML requires reliable data, leakage prevention, suitable metrics, threshold policy, explanations, human review, and governance - not only a high accuracy score.

