import json
from pathlib import Path

import pytest

from loan_approval.data import load_train_data, split_xy
from loan_approval.train import predict, train_and_evaluate


def test_training_outputs_artifacts(tmp_path, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(tmp_path)
    model_path = tmp_path / "model.joblib"
    metrics_path = tmp_path / "metrics.csv"

    out = train_and_evaluate(
        train_path=str(repo_root / "data" / "train.csv"),
        model_out=str(model_path),
        metrics_out=str(metrics_path),
        random_state=42,
    )

    assert out["best_model"] in {"logistic_regression", "random_forest", "gradient_boosting"}
    assert model_path.exists()
    assert metrics_path.exists()
    metadata = json.loads((tmp_path / "reports" / "metrics.json").read_text(encoding="utf-8"))
    assert metadata["app_version"] == "0.5.0"
    assert metadata["evaluation_method"] == (
        "single stratified validation split; model selection and reporting use the "
        "same split"
    )
    assert "model_version" not in metadata


def test_unlabelled_test_csv_is_rejected_as_training_data():
    repo_root = Path(__file__).resolve().parents[1]
    test_frame = load_train_data(repo_root / "data" / "test.csv")

    with pytest.raises(ValueError, match="Loan_Status target column"):
        split_xy(test_frame)


def test_predict_exposes_an_uncalibrated_score_not_a_calibrated_probability(monkeypatch):
    class Model:
        def predict_proba(self, _frame):
            return [[0.25, 0.75]]

    monkeypatch.setattr("loan_approval.train.joblib.load", lambda _path: Model())
    result = predict({"ApplicantIncome": 1000})

    assert result == {"approval_class_prediction": True, "approval_score": 0.75}
