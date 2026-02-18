# 💳 finance-loan-approval-prediction

[![CI](https://github.com/keremercin/finance-loan-approval-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/keremercin/finance-loan-approval-prediction/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688)
![ML](https://img.shields.io/badge/ML-Classification-orange)

Production-style machine learning project for **loan approval prediction** with a clean training pipeline and inference API.

---

## Problem
Loan approval decisions depend on multiple factors (income, credit history, property area, etc.).
Manual decision support is error-prone and inconsistent.

## Solution
This repository provides:
- robust preprocessing pipeline (imputation + scaling + one-hot encoding),
- model comparison across multiple classifiers,
- saved best model artifact,
- FastAPI prediction endpoint for integration.

---

## Validation results (current)

| Model | Accuracy | F1 | ROC-AUC |
|---|---:|---:|---:|
| logistic_regression | 0.8618 | **0.9081** | 0.8523 |
| random_forest | 0.8211 | 0.8764 | 0.7913 |
| gradient_boosting | 0.8049 | 0.8667 | 0.7551 |

Best model: **logistic_regression**

---

## API

- `GET /health`
- `POST /v1/predict`

Example request:

```json
{
  "Gender": "Male",
  "Married": "Yes",
  "Dependents": "1",
  "Education": "Graduate",
  "Self_Employed": "No",
  "ApplicantIncome": 4583,
  "CoapplicantIncome": 1508,
  "LoanAmount": 128,
  "Loan_Amount_Term": 360,
  "Credit_History": 1,
  "Property_Area": "Rural"
}
```

Example response:

```json
{
  "loan_approved": true,
  "approval_probability": 0.78
}
```

---

## Quickstart

```bash
cp .env.example .env  # optional (if needed later)
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]

# train model + metrics
python scripts/train_model.py
python scripts/make_model_card.py

# run API
uvicorn loan_approval.api.main:app --reload --port 8300
```

Swagger docs: `http://localhost:8300/docs`

---

## Project structure

```text
src/loan_approval/
├─ api/main.py
├─ data.py
├─ preprocess.py
└─ train.py

scripts/
├─ train_model.py
└─ make_model_card.py

tests/
├─ test_api.py
└─ test_training.py
```

---

## Engineering quality

- test coverage for training artifacts and API endpoint
- lint checks with `ruff`
- CI workflow on push/PR
- reproducible model card output

---

## Docs

- Model card: `reports/model_card.md`
- Case study: `docs/CASE_STUDY.md`

---

## Hiring signal

This project demonstrates that I can ship ML work beyond notebooks:
**data prep + model evaluation + artifact management + deployable inference API**.
