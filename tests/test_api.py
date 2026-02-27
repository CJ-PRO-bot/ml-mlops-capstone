from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def client():
    os.environ["SKIP_DB"] = "1"
    from deployments.fastapi_app.main import app  # import after env set

    return TestClient(app)


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "model" in data
    assert "n_features" in data


def test_predict_happy_path(client):
    payload = {
        "features": {
            "CO(GT)": 2.6,
            "T": 13.6,
            "hour": 18,
            "dayofweek": 3,
            "month": 3,
        }
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "y_cls" in data
    assert isinstance(data["y_cls"], int)
    assert "latency_ms" in data
