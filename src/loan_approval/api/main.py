from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from loan_approval.train import predict

APP_VERSION = "0.5.0"
app = FastAPI(title="Loan Approval Prediction API", version=APP_VERSION)


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


def api_response(*, data: Any = None, status: str = "ok", error: Any = None, latency_ms: int = 0) -> dict:
    return {
        "status": status,
        "data": data if data is not None else {},
        "meta": {"model_version": APP_VERSION, "latency_ms": latency_ms},
        "error": error,
    }


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content=api_response(
            status="error",
            error={"code": "VALIDATION_ERROR", "message": "Invalid request payload", "details": exc.errors()},
        ),
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=api_response(status="error", error={"code": "HTTP_ERROR", "message": str(exc.detail)}),
    )


@app.get("/health")
def health() -> dict:
    return api_response(data={"service": "loan-approval-api"})


@app.get("/version")
def version() -> dict:
    return api_response(data={"service": "loan-approval-api", "version": APP_VERSION})


@app.post("/v1/predict")
def predict_loan(req: PredictRequest) -> dict:
    return api_response(data=predict(req.model_dump()))
