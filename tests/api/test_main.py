from pathlib import Path

from fastapi.testclient import TestClient

from house_prediction.api.main import app, create_app
from house_prediction.config import Settings


def make_valid_payload() -> dict:
    return {
        "date": "2014-05-02",
        "bedrooms": 3,
        "bathrooms": 1.5,
        "floors": 1.0,
        "waterfront": 0,
        "view": 0,
        "condition": 3,
        "sqft_living": 1340,
        "sqft_lot": 7912,
        "sqft_above": 1340,
        "sqft_basement": 0,
        "yr_built": 1955,
        "yr_renovated": 0,
        "city": "Seattle",
        "state": "WA",
        "zip_code": "98101",
    }


def test_root_endpoint() -> None:
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the House Price Prediction API!"}


def test_health_endpoint() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_ready_endpoint() -> None:
    client = TestClient(create_app(Settings()))
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_ready_endpoint_not_ready(tmp_path: Path) -> None:
    settings = Settings()
    settings.model_path = tmp_path / "missing_model.joblib"
    client = TestClient(create_app(settings))
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "not_ready"}


def test_model_info_endpoint() -> None:
    client = TestClient(create_app(Settings()))

    response = client.get("/model-info")

    assert response.status_code == 200
    assert response.json()["model_version"] == "0.1.0"
    assert "evaluation_metrics" in response.json()


def test_model_info_returns_503_when_metadata_is_missing(tmp_path: Path) -> None:
    settings = Settings()
    settings.metadata_path = tmp_path / "missing_metadata.json"
    client = TestClient(create_app(settings))

    response = client.get("/model-info")

    assert response.status_code == 503
    assert response.json()["detail"] == "Model metadata not available"


def test_predict_endpoint() -> None:
    client = TestClient(create_app(Settings()))

    payload = make_valid_payload()
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert response.json()["prediction"] > 0
    assert response.json()["model_version"] == "0.1.0"


def test_invalid_predict_endpoint() -> None:
    client = TestClient(create_app(Settings()))

    payload = make_valid_payload()
    payload["bedrooms"] = "invalid_value"

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_returns_503_when_model_is_missing(tmp_path: Path) -> None:
    settings = Settings()
    settings.model_path = tmp_path / "missing_model.joblib"
    client = TestClient(create_app(settings))

    payload = make_valid_payload()
    response = client.post("/predict", json=payload)

    assert response.status_code == 503
    assert response.json()["detail"] == "Model not available"
