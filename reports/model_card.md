# Model Card

## Model
- Name: logistic_regression
- Application version: 0.5.0

## Use case
Experimental binary classification score only; not a real credit decision.

Model selection and the reported metrics use the same single stratified validation split. No probability calibration, fairness analysis, or external validation is available. The unlabeled `data/test.csv` is not scored. Dataset source and license are not documented in this repository.

## Metrics
- Accuracy: 0.8618
- F1: 0.9081
- ROC-AUC: 0.8523

## Limitations
- Metrics are from one validation split, not an independent test set.
- No calibration, fairness analysis, or external validation.
