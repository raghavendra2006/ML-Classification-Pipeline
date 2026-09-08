"""
Unit Tests for the Inference API
=================================
Tests the /health and /predict endpoints using FastAPI's TestClient.
Model and scaler are mocked to avoid requiring a running MLflow server.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Build a test app that mirrors the real app's routes but skips the
# lifespan (which would try to connect to MLflow).
# ---------------------------------------------------------------------------
from src.inference_api import health_check, predict, PredictionRequest
import src.inference_api as inference_module

# Create a test app WITHOUT the lifespan that connects to MLflow
test_app = FastAPI()
test_app.get("/health")(health_check)
test_app.post("/predict")(predict)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def mock_model_and_scaler():
    """
    Mock the MODEL and SCALER globals in the inference_api module
    so tests run without needing MLflow or a trained model.
    """
    # Create a mock model that always predicts class 1
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([1])

    # Create a mock scaler that returns data unchanged
    mock_scaler = MagicMock()
    mock_scaler.transform.return_value = np.array([[1.0] * 13])

    # Directly set the globals on the module
    original_model = inference_module.MODEL
    original_scaler = inference_module.SCALER

    inference_module.MODEL = mock_model
    inference_module.SCALER = mock_scaler

    yield mock_model, mock_scaler

    # Restore originals
    inference_module.MODEL = original_model
    inference_module.SCALER = original_scaler


@pytest.fixture
def client():
    """Create a TestClient using the test app (no MLflow lifespan)."""
    with TestClient(test_app) as c:
        yield c


# ---------------------------------------------------------------------------
# Health Endpoint Tests
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_200(self, client):
        """Health endpoint must return HTTP 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_healthy_status(self, client):
        """Health endpoint must return JSON with status 'healthy'."""
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"


# ---------------------------------------------------------------------------
# Predict Endpoint Tests — Happy Path
# ---------------------------------------------------------------------------

class TestPredictEndpointValid:
    """Tests for POST /predict with valid inputs."""

    def test_predict_valid_features(self, client):
        """Valid 13-feature input must return HTTP 200 with a prediction."""
        payload = {"features": [14.23, 1.71, 2.43, 15.6, 127.0,
                                2.80, 3.06, 0.28, 2.29, 5.64,
                                1.04, 3.92, 1065.0]}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200

    def test_predict_response_has_prediction_key(self, client):
        """Response JSON must contain the 'prediction' key."""
        payload = {"features": [14.23, 1.71, 2.43, 15.6, 127.0,
                                2.80, 3.06, 0.28, 2.29, 5.64,
                                1.04, 3.92, 1065.0]}
        response = client.post("/predict", json=payload)
        data = response.json()
        assert "prediction" in data

    def test_predict_returns_integer(self, client):
        """Prediction value must be an integer."""
        payload = {"features": [14.23, 1.71, 2.43, 15.6, 127.0,
                                2.80, 3.06, 0.28, 2.29, 5.64,
                                1.04, 3.92, 1065.0]}
        response = client.post("/predict", json=payload)
        data = response.json()
        assert isinstance(data["prediction"], int)


# ---------------------------------------------------------------------------
# Predict Endpoint Tests — Error Handling
# ---------------------------------------------------------------------------

class TestPredictEndpointErrors:
    """Tests for POST /predict with invalid inputs."""

    def test_predict_missing_features_key(self, client):
        """Missing 'features' key must return HTTP 422."""
        response = client.post("/predict", json={"data": [1.0, 2.0]})
        assert response.status_code == 422

    def test_predict_empty_body(self, client):
        """Empty JSON body must return HTTP 422."""
        response = client.post("/predict", json={})
        assert response.status_code == 422

    def test_predict_wrong_feature_count(self, client):
        """Wrong number of features must return HTTP 400."""
        payload = {"features": [1.0, 2.0, 3.0]}  # 3 instead of 13
        response = client.post("/predict", json=payload)
        assert response.status_code == 400

    def test_predict_wrong_feature_count_error_message(self, client):
        """Error response must include descriptive message."""
        payload = {"features": [1.0, 2.0, 3.0]}
        response = client.post("/predict", json=payload)
        data = response.json()
        assert "detail" in data
        assert "13" in data["detail"]

    def test_predict_invalid_content_type(self, client):
        """Non-JSON content must return HTTP 422."""
        response = client.post(
            "/predict",
            content="not json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422

    def test_predict_features_not_numeric(self, client):
        """Non-numeric features must return HTTP 422."""
        payload = {"features": ["a", "b", "c", "d", "e",
                                "f", "g", "h", "i", "j",
                                "k", "l", "m"]}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422
