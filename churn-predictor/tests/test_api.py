"""
Tests for api.py, run against the committed model.

Run with:
    pytest tests/test_api.py -v
"""

import pytest
from fastapi.testclient import TestClient

from app.api import app


SAMPLE_CUSTOMER = {
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
}


@pytest.fixture
def sample_customer() -> dict:
    """A single valid customer payload (same as the README example)."""
    return dict(SAMPLE_CUSTOMER)


@pytest.fixture(scope="module")
def client():
    # Using the client as a context manager runs the lifespan, which loads the model.
    with TestClient(app) as c:
        yield c


def make_customers(sample_customer: dict, n: int) -> list[dict]:
    return [{**sample_customer, "id": i, "tenure": 5 + i * 20} for i in range(n)]


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok", "model_loaded": True}


# ---------------------------------------------------------------------------
# /predict
# ---------------------------------------------------------------------------

class TestPredictSingle:
    def test_returns_label_and_proba(self, client, sample_customer):
        resp = client.post("/predict", json={"customer": sample_customer})
        assert resp.status_code == 200
        body = resp.json()
        assert body["id"] == sample_customer["id"]
        assert 0.0 <= body["churn_proba"] <= 1.0
        assert body["churn"] in (0, 1)

    def test_return_proba_omits_label(self, client, sample_customer):
        resp = client.post(
            "/predict", json={"customer": sample_customer, "return_proba": True}
        )
        assert resp.status_code == 200
        assert resp.json()["churn"] is None

    def test_threshold_zero_is_churn(self, client, sample_customer):
        resp = client.post(
            "/predict", json={"customer": sample_customer, "threshold": 0.0}
        )
        assert resp.json()["churn"] == 1

    def test_threshold_one_is_no_churn(self, client, sample_customer):
        resp = client.post(
            "/predict", json={"customer": sample_customer, "threshold": 1.0}
        )
        assert resp.json()["churn"] == 0

    def test_label_matches_threshold(self, client, sample_customer):
        body = client.post(
            "/predict", json={"customer": sample_customer, "threshold": 0.5}
        ).json()
        assert body["churn"] == int(body["churn_proba"] >= 0.5)

    def test_id_is_optional(self, client, sample_customer):
        del sample_customer["id"]
        resp = client.post("/predict", json={"customer": sample_customer})
        assert resp.status_code == 200
        assert resp.json()["id"] is None


# ---------------------------------------------------------------------------
# /predict/batch
# ---------------------------------------------------------------------------

class TestPredictBatch:
    def test_one_result_per_customer(self, client, sample_customer):
        customers = make_customers(sample_customer, 3)
        resp = client.post("/predict/batch", json={"customers": customers})
        assert resp.status_code == 200
        body = resp.json()
        assert [r["id"] for r in body] == [0, 1, 2]
        assert all(0.0 <= r["churn_proba"] <= 1.0 for r in body)
        assert all(r["churn"] in (0, 1) for r in body)

    def test_batch_matches_single(self, client, sample_customer):
        # Categories are inferred per request, so a one-row request and a
        # multi-row request must still be encoded the same way.
        customers = make_customers(sample_customer, 3)
        batch = client.post("/predict/batch", json={"customers": customers}).json()
        for customer, batch_result in zip(customers, batch):
            single = client.post("/predict", json={"customer": customer}).json()
            assert single["churn_proba"] == pytest.approx(batch_result["churn_proba"])

    def test_return_proba_omits_labels(self, client, sample_customer):
        customers = make_customers(sample_customer, 2)
        resp = client.post(
            "/predict/batch", json={"customers": customers, "return_proba": True}
        )
        assert all(r["churn"] is None for r in resp.json())

    def test_empty_batch_rejected(self, client):
        resp = client.post("/predict/batch", json={"customers": []})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Request validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_missing_field(self, client, sample_customer):
        del sample_customer["Contract"]
        resp = client.post("/predict", json={"customer": sample_customer})
        assert resp.status_code == 422

    @pytest.mark.parametrize(
        "field, value",
        [
            ("SeniorCitizen", 2),
            ("SeniorCitizen", -1),
            ("tenure", -1),
            ("MonthlyCharges", "abc"),
        ],
    )
    def test_invalid_field_value(self, client, sample_customer, field, value):
        sample_customer[field] = value
        resp = client.post("/predict", json={"customer": sample_customer})
        assert resp.status_code == 422

    @pytest.mark.parametrize("threshold", [-0.1, 1.5])
    def test_threshold_out_of_range(self, client, sample_customer, threshold):
        resp = client.post(
            "/predict", json={"customer": sample_customer, "threshold": threshold}
        )
        assert resp.status_code == 422

    def test_missing_customer_key(self, client):
        resp = client.post("/predict", json={})
        assert resp.status_code == 422
