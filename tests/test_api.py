"""Tests for the FastAPI endpoints."""
import sys
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

# Ensure ml_testing is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ml_testing"))


@pytest.fixture
def mock_services():
    """Mock the ML services to avoid needing actual model files."""
    mock_result = {"prediction": "Yes", "probability": 0.92}
    with patch("app.services.predict_from_dict", return_value=mock_result):
        yield mock_result


@pytest.fixture
def client(mock_services):
    """Create a test client with mocked services."""
    from main import app
    return TestClient(app)


def test_root_endpoint_returns_200(client):
    """Test that the root endpoint returns 200 when the frontend exists."""
    response = client.get("/")
    assert response.status_code == 200


def test_index_controller_status(client):
    """Test that the /ml/metadata endpoint is reachable (not 405 or 500 on GET)."""
    response = client.get("/ml/metadata")
    assert response.status_code in (200, 404)


def test_predict_valid_input(client):
    """Test the predict endpoint with valid input returns a prediction."""
    payload = {
        "features": {
            "A1": 1, "A2": 0, "A3": 1, "A4": 0, "A5": 1,
            "A6": 1, "A7": 0, "A8": 1, "A9": 0, "A10": 1,
            "Age_Mons": 36,
            "Qchat-10-Score": 6,
            "Sex": "m",
            "Ethnicity": "White European",
            "Jaundice": "no",
            "Family_mem_with_ASD": "no",
            "Who completed the test": "family member"
        }
    }
    response = client.post("/ml/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert data["prediction"] in ("Yes", "No")


def test_predict_returns_probability(client):
    """Test that the predict endpoint may return a probability."""
    payload = {
        "features": {
            "A1": 0, "A2": 0, "A3": 0, "A4": 0, "A5": 0,
            "A6": 0, "A7": 0, "A8": 0, "A9": 0, "A10": 0,
            "Age_Mons": 24,
            "Qchat-10-Score": 2,
            "Sex": "f",
            "Ethnicity": "Asian",
            "Jaundice": "yes",
            "Family_mem_with_ASD": "yes",
            "Who completed the test": "mother"
        }
    }
    response = client.post("/ml/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    if data.get("probability") is not None:
        assert 0.0 <= data["probability"] <= 1.0


def test_predict_missing_features_returns_422(client):
    """Test that sending an empty features dict returns 422 (validation error)."""
    payload = {"features": {}}
    response = client.post("/ml/predict", json=payload)
    assert response.status_code == 422


def test_predict_invalid_payload_returns_422(client):
    """Test that an invalid payload (no features key) returns 422."""
    response = client.post("/ml/predict", json={"wrong_key": {}})
    assert response.status_code == 422


def test_metadata_endpoint_returns_404_when_missing():
    """Test that the metadata endpoint returns 404 when metadata file is missing."""
    mock_result = {"prediction": "Yes", "probability": 0.92}
    with patch("app.services.predict_from_dict", return_value=mock_result):
        from main import app
        import app.controllers as ctrl
        test_client = TestClient(app)
        with patch.object(ctrl, "MODEL_DIR", Path("/nonexistent/path")):
            response = test_client.get("/ml/metadata")
    assert response.status_code == 404
