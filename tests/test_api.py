from fastapi.testclient import TestClient
from loan_approval.api.main import app
from loan_approval.train import train_and_evaluate


def test_health() -> None:
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["meta"]["app_version"] == "0.5.0"
    assert r.json()["meta"]["latency_ms"] is None
    assert "model_version" not in r.json()["meta"]


def test_version() -> None:
    client = TestClient(app)
    r = client.get("/version")
    assert r.status_code == 200
    assert r.json()["data"]["version"] == "0.5.0"


def test_predict_endpoint(tmp_path, monkeypatch) -> None:
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(tmp_path)
    train_and_evaluate(train_path=str(repo_root / "data" / "train.csv"))

    client = TestClient(app)
    payload = {
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
        "Property_Area": "Rural",
    }
    r = client.post("/v1/predict", json=payload)
    assert r.status_code == 200
    body = r.json()["data"]
    assert "approval_class_prediction" in body
    assert "approval_score" in body
    assert "approval_probability" not in body
