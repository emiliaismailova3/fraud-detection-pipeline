"""Smoke tests for the scoring API. Requires trained models in models/."""
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_returns_probability():
    payload = {"amt": 117.0, "hour": 3, "product_cd": "C", "card4": "visa",
               "card6": "credit", "p_emaildomain": "gmail.com", "uid_tx_last_24h": 6}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert isinstance(body["alert"], bool)


def test_minimal_payload_is_accepted():
    # Only the amount is required; everything else may be missing
    response = client.post("/predict", json={"amt": 25.0})
    assert response.status_code == 200


def test_invalid_amount_is_rejected():
    response = client.post("/predict", json={"amt": -10})
    assert response.status_code == 422
