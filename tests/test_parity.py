"""Training/serving parity: the API must reproduce the offline model score.

If the API computed a derived feature slightly differently from SQL (say,
log_amt or email_mismatch), the model would silently get wrong inputs.
This test feeds real test-period rows through the API and compares.
"""
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.main import app, booster, preprocessor
from src.config import FEATURES_PARQUET, VALID_END_DAY
from src.preprocess import as_category

API_FIELDS = set(app.openapi()["components"]["schemas"]["Transaction"]["properties"])


@pytest.mark.skipif(not FEATURES_PARQUET.exists(), reason="needs data/processed/features.parquet")
def test_api_matches_offline_scores():
    df = pd.read_parquet(FEATURES_PARQUET)
    sample = df[df["day"] >= VALID_END_DAY].sample(200, random_state=0)
    offline = booster.predict(as_category(preprocessor.transform_tree(sample), preprocessor))

    client = TestClient(app)
    online = []
    for _, row in sample.iterrows():
        payload = {k: (None if pd.isna(v) else v) for k, v in row.items() if k in API_FIELDS}
        payload["hour"] = int(payload["hour"])
        online.append(client.post("/predict", json=payload).json()["fraud_probability"])

    np.testing.assert_allclose(online, offline, atol=1e-4)
