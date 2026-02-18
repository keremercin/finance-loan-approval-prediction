from loan_approval.train import train_and_evaluate


def test_training_outputs_artifacts(tmp_path) -> None:
    model_path = tmp_path / "model.joblib"
    metrics_path = tmp_path / "metrics.csv"

    out = train_and_evaluate(
        train_path="data/train.csv",
        model_out=str(model_path),
        metrics_out=str(metrics_path),
        random_state=42,
    )

    assert out["best_model"] in {"logistic_regression", "random_forest", "gradient_boosting"}
    assert model_path.exists()
    assert metrics_path.exists()
