from fastapi.testclient import TestClient

from loan_approval.api.main import app
from loan_approval.train import train_and_evaluate


def test_health() -> None:
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200


def test_predict_endpoint() -> None:
    train_and_evaluate()

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
    body = r.json()
    assert "loan_approved" in body
    assert "approval_probability" in body
