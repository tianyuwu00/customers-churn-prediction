"""
api.py — FastAPI inference service for the LightGBM churn model.

Usage
-----
    uvicorn app.api:app --reload
    uvicorn app.api:app --host 0.0.0.0 --port 8000

Endpoints
---------
    GET  /health                  — liveness check
    POST /predict                 — single customer prediction
    POST /predict/batch           — batch prediction (list of customers)
"""

from contextlib import asynccontextmanager
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.predict import DEFAULT_MODEL_PATH, load_model, predict as run_predict


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class CustomerFeatures(BaseModel):
    id: Optional[int] = None
    gender: str
    SeniorCitizen: int = Field(..., ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(..., ge=0)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

    model_config = {"json_schema_extra": {"example": {
        "id": 1,
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "Yes",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 358.20,
    }}}


class PredictRequest(BaseModel):
    customer: CustomerFeatures
    threshold: float = Field(0.5, ge=0.0, le=1.0)
    return_proba: bool = False


class BatchPredictRequest(BaseModel):
    customers: list[CustomerFeatures]
    threshold: float = Field(0.5, ge=0.0, le=1.0)
    return_proba: bool = False


class PredictionResult(BaseModel):
    id: Optional[int]
    churn_proba: float
    churn: Optional[int] = None


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

model_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_state["model"] = load_model(DEFAULT_MODEL_PATH)
    yield
    model_state.clear()


app = FastAPI(
    title="Churn Predictor API",
    description="LightGBM-based customer churn prediction service.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _customers_to_df(customers: list[CustomerFeatures]) -> pd.DataFrame:
    return pd.DataFrame([c.model_dump() for c in customers])


def _format_results(result_df: pd.DataFrame, return_proba: bool) -> list[PredictionResult]:
    out = []
    for row in result_df.itertuples(index=False):
        out.append(PredictionResult(
            id=getattr(row, "id", None),
            churn_proba=round(row.churn_proba, 6),
            churn=getattr(row, "Churn", None) if not return_proba else None,
        ))
    return out


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in model_state}


@app.post("/predict", response_model=PredictionResult)
def predict_single(req: PredictRequest):
    try:
        df = _customers_to_df([req.customer])
        result = run_predict(df, model_state["model"], req.threshold, req.return_proba)
        return _format_results(result, req.return_proba)[0]
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.post("/predict/batch", response_model=list[PredictionResult])
def predict_batch(req: BatchPredictRequest):
    if not req.customers:
        raise HTTPException(status_code=422, detail="customers list must not be empty")
    try:
        df = _customers_to_df(req.customers)
        result = run_predict(df, model_state["model"], req.threshold, req.return_proba)
        return _format_results(result, req.return_proba)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))
