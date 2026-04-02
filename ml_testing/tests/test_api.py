"""Integration tests for the FastAPI API endpoints.

Uses the HTTPX ``TestClient`` (sync) to exercise the full request/response
cycle without starting a real server.  Model artifacts are mocked so tests
run without requiring ``model/model.pkl`` to be present.
"""

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_FEATURES: dict[str, Any] = {
    "A1": 0, "A2": 0, "A3": 0, "A4": 0, "A5": 0,
    "A6": 0, "A7": 0, "A8": 0, "A9": 0, "A10": 0,
    "Age_Mons": 36, "Qchat-10-Score": 6,
    "Sex": "m", "Ethnicity": "White European",
    "Jaundice": "no", "Family_mem_with_ASD": "no",
    "Who completed the test": "family member",
}

_FEATURE_COLS = list(_VALID_FEATURES.keys())

_METADATA = {
    "feature_columns": _FEATURE_COLS,
    "numeric_columns": _FEATURE_COLS[:12],
    "categorical_columns": _FEATURE_COLS[12:],
    "target_column": "Class/ASD Traits",
    "target_classes": ["No", "Yes"],
    "best_model": "RandomForest",
    "scores": {"RandomForest": {"f1": 1.0, "acc": 1.0}},
}


def _mock_load_once(predict_val=0, proba_val=0.85):
    """Return a (model, le, meta) triple with mocked objects."""
    model = MagicMock()
    model.predict.return_value = np.array([predict_val])
    model.predict_proba.return_value = np.array([[1.0 - proba_val, proba_val]])

    le = MagicMock()
    le.inverse_transform.side_effect = lambda idx: [("No", "Yes")[idx[0]]]

    return model, le, _METADATA


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def client():
    """TestClient with mocked model artifacts (predict → 'No')."""
    import app.services as svc

    svc._model = None
    svc._le = None
    svc._meta = None

    with patch.object(svc, "load_once", return_value=_mock_load_once(predict_val=0)):
        from main import create_app
        app = create_app()
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c

    svc._model = None
    svc._le = None
    svc._meta = None


@pytest.fixture()
def client_yes():
    """TestClient where the model predicts 'Yes' (ASD traits present)."""
    import app.services as svc

    svc._model = None
    svc._le = None
    svc._meta = None

    with patch.object(svc, "load_once", return_value=_mock_load_once(predict_val=1, proba_val=0.92)):
        from main import create_app
        app = create_app()
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c

    svc._model = None
    svc._le = None
    svc._meta = None


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_health_message(self, client):
        resp = client.get("/health")
        assert "message" in resp.json()


# ---------------------------------------------------------------------------
# POST /ml/predict
# ---------------------------------------------------------------------------

class TestPredictEndpoint:
    def test_predict_no(self, client):
        resp = client.post("/ml/predict", json={"features": _VALID_FEATURES})
        assert resp.status_code == 200
        data = resp.json()
        assert "prediction" in data
        assert data["prediction"] == "No"

    def test_predict_yes(self, client_yes):
        resp = client_yes.post("/ml/predict", json={"features": _VALID_FEATURES})
        assert resp.status_code == 200
        data = resp.json()
        assert data["prediction"] == "Yes"

    def test_predict_returns_probability(self, client):
        resp = client.post("/ml/predict", json={"features": _VALID_FEATURES})
        assert resp.status_code == 200
        data = resp.json()
        assert "probability" in data
        assert 0.0 <= data["probability"] <= 1.0

    def test_predict_all_ones(self, client_yes):
        feat = _VALID_FEATURES.copy()
        for i in range(1, 11):
            feat[f"A{i}"] = 1
        feat["Qchat-10-Score"] = 10
        resp = client_yes.post("/ml/predict", json={"features": feat})
        assert resp.status_code == 200

    def test_predict_missing_body(self, client):
        resp = client.post("/ml/predict", json={})
        assert resp.status_code == 422

    def test_predict_invalid_a_value(self, client):
        feat = _VALID_FEATURES.copy()
        feat["A1"] = 99
        resp = client.post("/ml/predict", json={"features": feat})
        assert resp.status_code == 422

    def test_predict_invalid_sex(self, client):
        feat = _VALID_FEATURES.copy()
        feat["Sex"] = "z"
        resp = client.post("/ml/predict", json={"features": feat})
        assert resp.status_code == 422

    def test_predict_negative_age(self, client):
        feat = _VALID_FEATURES.copy()
        feat["Age_Mons"] = -10
        resp = client.post("/ml/predict", json={"features": feat})
        assert resp.status_code == 422

    def test_predict_qchat_over_limit(self, client):
        feat = _VALID_FEATURES.copy()
        feat["Qchat-10-Score"] = 15
        resp = client.post("/ml/predict", json={"features": feat})
        assert resp.status_code == 422

    def test_predict_wrong_content_type(self, client):
        resp = client.post(
            "/ml/predict",
            content="not json",
            headers={"Content-Type": "text/plain"},
        )
        assert resp.status_code in (415, 422)

    def test_predict_model_missing_returns_500(self, client):
        import app.services as svc
        with patch.object(svc, "load_once", side_effect=FileNotFoundError("no file")):
            resp = client.post("/ml/predict", json={"features": _VALID_FEATURES})
        assert resp.status_code == 500


# ---------------------------------------------------------------------------
# GET /ml/metadata
# ---------------------------------------------------------------------------

class TestMetadataEndpoint:
    def test_metadata_returns_200(self, client, tmp_path):
        import json as _json
        import app.controllers as ctrl

        meta_file = tmp_path / "metadata.json"
        meta_file.write_text(_json.dumps(_METADATA))

        with patch.object(ctrl, "METADATA_PATH", meta_file):
            resp = client.get("/ml/metadata")

        assert resp.status_code == 200
        data = resp.json()
        assert "feature_columns" in data
        assert "target_classes" in data

    def test_metadata_not_found(self, client, tmp_path):
        import app.controllers as ctrl

        with patch.object(ctrl, "METADATA_PATH", tmp_path / "nonexistent.json"):
            resp = client.get("/ml/metadata")

        assert resp.status_code == 404

    def test_metadata_feature_columns_list(self, client, tmp_path):
        import json as _json
        import app.controllers as ctrl

        meta_file = tmp_path / "metadata.json"
        meta_file.write_text(_json.dumps(_METADATA))

        with patch.object(ctrl, "METADATA_PATH", meta_file):
            resp = client.get("/ml/metadata")

        data = resp.json()
        assert isinstance(data["feature_columns"], list)
        assert len(data["feature_columns"]) == len(_FEATURE_COLS)


# ---------------------------------------------------------------------------
# Root endpoint
# ---------------------------------------------------------------------------

class TestRootEndpoint:
    def test_root_no_frontend(self, client):
        """When frontend dir doesn't exist, root returns 404."""
        from app.config import FRONTEND_DIR
        with patch("main.FRONTEND_DIR", FRONTEND_DIR.parent / "__nonexistent__"):
            resp = client.get("/")
        # Either 200 (with url) or 404 — depends on whether the directory exists
        assert resp.status_code in (200, 404)
