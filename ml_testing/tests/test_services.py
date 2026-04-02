"""Unit tests for the ML service layer (app/services.py).

These tests mock the model artifacts so they can run without actual .pkl files.
"""

import json
import pickle
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_mock_model(predict_val=0, proba_val=None):
    """Create a simple sklearn-like pipeline mock."""
    model = MagicMock()
    model.predict.return_value = np.array([predict_val])
    if proba_val is not None:
        model.predict_proba.return_value = np.array([[1.0 - proba_val, proba_val]])
    else:
        del model.predict_proba  # model without predict_proba
    return model


def _make_mock_label_encoder(classes=("No", "Yes")):
    """Create a simple LabelEncoder-like mock."""
    le = MagicMock()
    le.inverse_transform.side_effect = lambda idx: [classes[idx[0]]]
    return le


def _make_metadata(feature_cols: list[str]) -> dict:
    return {
        "feature_columns": feature_cols,
        "numeric_columns": feature_cols[:12],
        "categorical_columns": feature_cols[12:],
        "target_column": "Class/ASD Traits",
        "target_classes": ["No", "Yes"],
        "best_model": "RandomForest",
        "scores": {"RandomForest": {"f1": 1.0, "acc": 1.0}},
    }


_FEATURE_COLS = [
    "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10",
    "Age_Mons", "Qchat-10-Score",
    "Sex", "Ethnicity", "Jaundice", "Family_mem_with_ASD",
    "Who completed the test",
]

_VALID_FEATURES = {
    "A1": 0, "A2": 0, "A3": 0, "A4": 0, "A5": 0,
    "A6": 0, "A7": 0, "A8": 0, "A9": 0, "A10": 0,
    "Age_Mons": 36, "Qchat-10-Score": 6,
    "Sex": "m", "Ethnicity": "White European",
    "Jaundice": "no", "Family_mem_with_ASD": "no",
    "Who completed the test": "family member",
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPredictFromDict:
    """Tests for ``services.predict_from_dict``."""

    def _patch_load_once(self, predict_val=0, proba_val=0.85):
        """Context manager that patches ``services.load_once``."""
        import app.services as svc

        mock_model = _make_mock_model(predict_val=predict_val, proba_val=proba_val)
        mock_le = _make_mock_label_encoder()
        mock_meta = _make_metadata(_FEATURE_COLS)

        return patch.object(svc, "load_once", return_value=(mock_model, mock_le, mock_meta))

    def test_returns_prediction_and_probability(self):
        import app.services as svc

        with self._patch_load_once(predict_val=0, proba_val=0.85):
            result = svc.predict_from_dict(_VALID_FEATURES)

        assert "prediction" in result
        assert "probability" in result
        assert result["prediction"] in ("Yes", "No", "0", "1")
        assert 0.0 <= result["probability"] <= 1.0

    def test_prediction_no(self):
        """When model predicts 0 → label 'No'."""
        import app.services as svc

        with self._patch_load_once(predict_val=0, proba_val=0.1):
            result = svc.predict_from_dict(_VALID_FEATURES)

        assert result["prediction"] == "No"

    def test_prediction_yes(self):
        """When model predicts 1 → label 'Yes'."""
        import app.services as svc

        with self._patch_load_once(predict_val=1, proba_val=0.9):
            result = svc.predict_from_dict(_VALID_FEATURES)

        assert result["prediction"] == "Yes"

    def test_missing_features_raises_key_error(self):
        import app.services as svc

        incomplete = {k: v for k, v in _VALID_FEATURES.items() if k != "A1"}

        with self._patch_load_once():
            with pytest.raises(KeyError, match="Missing features"):
                svc.predict_from_dict(incomplete)

    def test_file_not_found_propagates(self):
        """FileNotFoundError from load_once propagates to caller."""
        import app.services as svc

        with patch.object(svc, "load_once", side_effect=FileNotFoundError("no model")):
            with pytest.raises(FileNotFoundError):
                svc.predict_from_dict(_VALID_FEATURES)

    def test_model_runtime_error_wrapped(self):
        """A RuntimeError from model.predict is re-raised as RuntimeError."""
        import app.services as svc

        mock_model = MagicMock()
        mock_model.predict.side_effect = ValueError("dtype error")
        mock_le = _make_mock_label_encoder()
        mock_meta = _make_metadata(_FEATURE_COLS)

        with patch.object(svc, "load_once", return_value=(mock_model, mock_le, mock_meta)):
            with pytest.raises(RuntimeError, match="Model prediction failed"):
                svc.predict_from_dict(_VALID_FEATURES)

    def test_no_predict_proba(self):
        """When model lacks predict_proba, probability should be None."""
        import app.services as svc

        mock_model = MagicMock(spec=["predict"])  # no predict_proba attribute
        mock_model.predict.return_value = np.array([0])
        mock_le = _make_mock_label_encoder()
        mock_meta = _make_metadata(_FEATURE_COLS)

        with patch.object(svc, "load_once", return_value=(mock_model, mock_le, mock_meta)):
            result = svc.predict_from_dict(_VALID_FEATURES)

        assert result["probability"] is None


class TestLoadOnce:
    """Tests for the artifact-loading logic (using temp files)."""

    def test_load_once_with_real_files(self, tmp_path: Path):
        """load_once should succeed when all three artifact files exist."""
        import app.services as svc
        from sklearn.dummy import DummyClassifier
        from sklearn.preprocessing import LabelEncoder

        # Build minimal real (picklable) artifacts
        clf = DummyClassifier(strategy="most_frequent")
        clf.fit([[0], [1]], [0, 1])  # minimal fit

        le = LabelEncoder()
        le.fit(["No", "Yes"])

        meta = _make_metadata(_FEATURE_COLS)

        model_path = tmp_path / "model.pkl"
        le_path = tmp_path / "label_encoder.pkl"
        meta_path = tmp_path / "metadata.json"

        with open(model_path, "wb") as f:
            pickle.dump(clf, f)
        with open(le_path, "wb") as f:
            pickle.dump(le, f)
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        # Reset module-level caches and patch paths
        svc._model = None
        svc._le = None
        svc._meta = None

        with (
            patch.object(svc, "MODEL_PATH", model_path),
            patch.object(svc, "LABEL_ENCODER_PATH", le_path),
            patch.object(svc, "METADATA_PATH", meta_path),
        ):
            model, le_loaded, loaded_meta = svc.load_once()

        assert model is not None
        assert le_loaded is not None
        assert loaded_meta["best_model"] == "RandomForest"

        # Restore module caches to avoid test pollution
        svc._model = None
        svc._le = None
        svc._meta = None

    def test_load_once_missing_model_raises(self, tmp_path: Path):
        import app.services as svc

        svc._model = None
        svc._le = None
        svc._meta = None

        with patch.object(svc, "MODEL_PATH", tmp_path / "nonexistent.pkl"):
            with pytest.raises(FileNotFoundError):
                svc.load_once()

        svc._model = None
        svc._le = None
        svc._meta = None
