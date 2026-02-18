# Case Study — Loan Approval Prediction

## Context
Financial teams need fast, consistent decision support for loan approvals.

## Approach
- standardized preprocessing with explicit numeric/categorical handling,
- model comparison across logistic regression, random forest, and gradient boosting,
- selection based on validation F1,
- API wrapping for integration into downstream workflows.

## Outcome
- best model: logistic regression,
- strong F1 with reliable baseline behavior,
- deploy-ready prediction endpoint.

## Next Improvements
- probability calibration,
- fairness checks by protected groups,
- drift monitoring for production data.
