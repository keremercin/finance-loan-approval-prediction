# 90-Second Demo Script

## 0-20s: Problem framing
"This is a portfolio classification exercise; it is not validated for real credit decisions."

## 20-55s: Training and artifacts
- Run `python scripts/train_model.py`
- Show `reports/metrics.csv`
- Show `reports/metrics.json`
- Show `reports/model_card.md`

## 55-80s: API demo
- Open `/docs`
- Send request to `/v1/predict`
- Show the uncalibrated `approval_score` and class prediction

## 80-90s: Hiring close
"This repo demonstrates one stratified validation workflow. It has no calibration, fairness review, or external validation."
