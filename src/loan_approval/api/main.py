from fastapi import FastAPI
from pydantic import BaseModel

from loan_approval.train import predict

app = FastAPI(title="Loan Approval Prediction API", version="0.2.0")


class PredictRequest(BaseModel):
    Gender: str
    Married: str
    Dependents: str
    Education: str
    Self_Employed: str
    ApplicantIncome: float
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: float
    Credit_History: float
    Property_Area: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "loan-approval-api"}


@app.post("/v1/predict")
def predict_loan(req: PredictRequest) -> dict:
    return predict(req.model_dump())
