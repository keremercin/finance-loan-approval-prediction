import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from loan_approval.data import load_train_data, split_xy
from loan_approval.preprocess import make_preprocessor


APP_VERSION = "0.5.0"


def _models(random_state: int = 42):
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=random_state),
        "gradient_boosting": GradientBoostingClassifier(random_state=random_state),
    }


def train_and_evaluate(
    train_path: str = "data/train.csv",
    model_out: str = "models/best_model.joblib",
    metrics_out: str = "reports/metrics.csv",
    random_state: int = 42,
) -> dict:
    df = load_train_data(train_path)
    x, y = split_xy(df)

    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=random_state
    )

    preprocessor = make_preprocessor()
    rows = []
    best_name = None
    best_score = -1.0
    best_pipeline = None

    for name, model in _models(random_state=random_state).items():
        pipe = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )
        pipe.fit(x_train, y_train)

        pred = pipe.predict(x_val)
        prob = pipe.predict_proba(x_val)[:, 1] if hasattr(pipe, "predict_proba") else pred

        acc = float(accuracy_score(y_val, pred))
        f1 = float(f1_score(y_val, pred))
        auc = float(roc_auc_score(y_val, prob))

        rows.append({"model": name, "accuracy": acc, "f1": f1, "roc_auc": auc})

        if f1 > best_score:
            best_score = f1
            best_name = name
            best_pipeline = pipe

    metrics_df = pd.DataFrame(rows).sort_values("f1", ascending=False)

    Path("reports").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(metrics_out, index=False)

    metrics_json = {
        "schema_version": "1.0",
        "project": "finance-loan-approval-prediction",
        "model_version": APP_VERSION,
        "best_model": best_name,
        "best_f1": round(best_score, 4),
        "metrics": metrics_df.to_dict(orient="records"),
    }
    Path("reports/metrics.json").write_text(json.dumps(metrics_json, indent=2), encoding="utf-8")

    joblib.dump(best_pipeline, model_out)

    return {
        "best_model": best_name,
        "best_f1": best_score,
        "metrics": metrics_df.to_dict(orient="records"),
        "model_path": model_out,
        "metrics_path": metrics_out,
        "metrics_json_path": "reports/metrics.json",
    }


def predict(payload: dict, model_path: str = "models/best_model.joblib") -> dict:
    model = joblib.load(model_path)
    x = pd.DataFrame([payload])
    proba = float(model.predict_proba(x)[0][1])
    pred = int(proba >= 0.5)
    return {
        "loan_approved": bool(pred),
        "approval_probability": round(proba, 4),
    }
