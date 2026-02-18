# Architecture

## Components
- Training + model selection: `src/loan_approval/train.py`
- Data loading and split helpers: `src/loan_approval/data.py`
- Feature preprocessing: `src/loan_approval/preprocess.py`
- Inference API: `src/loan_approval/api/main.py`

## Flow
1. Train and evaluate candidate models.
2. Persist best model artifact and metrics outputs.
3. Serve prediction endpoint via FastAPI.
4. Return standard response envelope for operational consistency.
