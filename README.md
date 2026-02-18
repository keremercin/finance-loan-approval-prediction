# finance-loan-approval-prediction

[![CI](https://github.com/keremercin/finance-loan-approval-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/keremercin/finance-loan-approval-prediction/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688)
![ML](https://img.shields.io/badge/ML-Classification-orange)

Production-style machine learning system for loan approval prediction with reproducible training and API inference.

## Problem
Loan approval support is often inconsistent when rules and model outputs are not standardized.

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
  "meta": {"model_version": "0.5.0", "latency_ms": 0},
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
Current best model and key metrics are generated and stored automatically during training.

## Limitations
- Model explainability endpoints are not included yet.
- Data drift monitoring is not implemented.
- Current model set is tabular baseline-focused.

## Roadmap
- Add SHAP-based explainability summary endpoint.
- Add drift check job and threshold alerts.
- Add model registry metadata for release tracking.

## Docs
- `docs/CASE_STUDY.md`
- `docs/ARCHITECTURE.md`
- `docs/DEMO_SCRIPT_90S.md`
