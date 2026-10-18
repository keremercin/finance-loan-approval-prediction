# Validation Demo — Loan Approval Prediction

## Context
This repository is a code demonstration. It does not describe a financial institution, customer deployment, or measured lending outcome.

## Approach
- standardized preprocessing with explicit numeric/categorical handling,
- model comparison across logistic regression, random forest, and gradient boosting,
- selection based on validation F1,
- a local API returning a class prediction and an uncalibrated model score.

The three models are compared on one stratified validation split, and model selection uses that same split's F1. `data/test.csv` is unlabeled and is not scored. The dataset source and license are not documented here.

## Next Improvements
- probability calibration,
- fairness checks by protected groups,
- external validation on an independently labeled dataset,
- drift monitoring before any operational use.
