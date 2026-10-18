# finance-loan-approval-prediction

[![CI](https://github.com/keremercin/finance-loan-approval-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/keremercin/finance-loan-approval-prediction/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688)
![ML](https://img.shields.io/badge/ML-Classification-orange)

Experimental tabular classification baseline for a loan-approval dataset. It is a portfolio exercise and must not be used to make or recommend real credit decisions.

## Problem
The repository demonstrates preprocessing, a single stratified train/validation split, model comparison, and a local prediction API.

## Architecture
- Data + preprocessing pipeline under `src/loan_approval`
- Multi-model training and ranking in `train.py`
- FastAPI inference service in `api/main.py`
- Artifacts under `models/` and `reports/`

See `docs/ARCHITECTURE.md`.

## Local Run
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
python scripts/train_model.py
python scripts/make_model_card.py
uvicorn loan_approval.api.main:app --reload --port 8300
```

## API Spec
- `GET /health`
- `GET /version`
- `POST /v1/predict`

Response envelope:
```json
{
  "status": "ok",
  "data": {},
  "meta": {"app_version": "0.5.0", "latency_ms": null},
  "error": null
}
```

## Evaluation
```bash
python scripts/train_model.py
```

Evaluation artifacts:
- `reports/metrics.csv`
- `reports/metrics.json`
- `reports/model_card.md`

## Results
The code compares three classifiers on the same stratified validation split, selects the best by F1, and reports the metrics from that same split. `data/test.csv` has no `Loan_Status` label and is not used as a test result. The API's `approval_score` is the model's raw `predict_proba` output; it is not calibrated. `latency_ms` is `null` because API request duration is not measured.

`app_version` identifies the running API application; the installable package version is defined separately in `pyproject.toml`. No trained model version is claimed.

## Limitations
- No calibration, fairness analysis, or external validation is implemented.
- Dataset source and license are not identified in this repository; verify them before redistribution or use.
- No explainability endpoint or drift monitoring is implemented.
- This experiment is not validated for production use or real credit decisions.

## Roadmap
- Add SHAP-based explainability summary endpoint.
- Add drift check job and threshold alerts.
- Add model registry metadata for release tracking.

## Docs
- `docs/CASE_STUDY.md`
- `docs/ARCHITECTURE.md`
- `docs/DEMO_SCRIPT_90S.md`
