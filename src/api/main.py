"""FastAPI service that scores one transaction with the LightGBM model.

The request carries the transaction fields plus the customer-history features
(uid_prev_tx, uid_tx_last_24h, ...). In production those history features
would come from a feature store or the PostgreSQL feature layer; here the
caller sends them, which keeps the service stateless and easy to deploy.

Run locally:  uvicorn src.api.main:app --reload
Demo page:    http://localhost:8000/
Docs:         http://localhost:8000/docs
"""
import json
import math
from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.config import CATEGORICAL, MODELS, NUMERIC
from src.preprocess import as_category

MODEL_DIR = Path(MODELS)
STATIC_DIR = Path(__file__).parent / "static"
preprocessor = joblib.load(MODEL_DIR / "preprocessor.joblib")
booster = lgb.Booster(model_file=str(MODEL_DIR / "lightgbm.txt"))
api_config = json.loads((MODEL_DIR / "api_config.json").read_text())


class Transaction(BaseModel):
    """Any field may be omitted; the model treats missing values natively."""

    amt: float = Field(..., gt=0, description="Transaction amount, USD")
    hour: int | None = Field(None, ge=0, le=23)
    product_cd: str | None = Field(None, examples=["W"])
    card1: str | None = None
    card2: str | None = None
    card3: str | None = None
    card4: str | None = Field(None, examples=["visa"])
    card5: str | None = None
    card6: str | None = Field(None, examples=["debit"])
    addr1: str | None = None
    addr2: str | None = None
    dist1: float | None = None
    p_emaildomain: str | None = Field(None, examples=["gmail.com"])
    r_emaildomain: str | None = None
    c1: float | None = None
    c2: float | None = None
    c5: float | None = None
    c6: float | None = None
    c9: float | None = None
    c11: float | None = None
    c13: float | None = None
    c14: float | None = None
    d1: float | None = None
    d2: float | None = None
    d3: float | None = None
    d4: float | None = None
    d10: float | None = None
    d15: float | None = None
    m4: str | None = None
    m5: str | None = None
    m6: str | None = None
    device_type: str | None = Field(None, examples=["mobile"])
    has_identity: int = 0
    # Customer-history features (computed from previous transactions only)
    uid_prev_tx: float = 0
    uid_prev_mean_amt: float | None = None
    uid_prev_std_amt: float | None = None
    uid_tx_last_1h: float = 0
    uid_tx_last_24h: float = 0
    card1_tx_last_24h: float = 0
    secs_since_prev_tx: float | None = None
    uid_same_amt_prev: float = 0


class Score(BaseModel):
    fraud_probability: float
    alert: bool
    threshold: float
    model: str


def to_features(tx: Transaction) -> pd.DataFrame:
    """Recreate the derived columns exactly as sql/04_features.sql does."""
    row = tx.model_dump()
    row["log_amt"] = math.log1p(row["amt"])
    row["amt_cents"] = round(row["amt"] - math.floor(row["amt"]), 3)
    row["email_mismatch"] = int(
        row["r_emaildomain"] is not None and row["p_emaildomain"] != row["r_emaildomain"]
    )
    mean = row["uid_prev_mean_amt"]
    row["amt_to_uid_mean"] = row["amt"] / mean if mean else None
    df = pd.DataFrame([row])[NUMERIC + CATEGORICAL]
    df[NUMERIC] = df[NUMERIC].astype("float64")
    return df


app = FastAPI(
    title="Fraud Detection API",
    description="Scores a card transaction with a LightGBM model trained on IEEE-CIS data.",
    version="1.0.0",
)


@app.get("/", include_in_schema=False)
def demo_page() -> FileResponse:
    """Human-friendly demo: a form that calls /predict and shows the verdict."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": api_config["model"]}


@app.post("/predict", response_model=Score)
def predict(tx: Transaction) -> Score:
    x = as_category(preprocessor.transform_tree(to_features(tx)), preprocessor)
    probability = float(booster.predict(x)[0])
    threshold = api_config["alert_threshold"]
    return Score(
        fraud_probability=round(probability, 5),
        alert=probability >= threshold,
        threshold=round(threshold, 5),
        model=api_config["model"],
    )
