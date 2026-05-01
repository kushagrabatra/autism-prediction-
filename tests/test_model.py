"""Tests for ML model prediction logic."""
import sys
import os
from unittest.mock import patch, MagicMock

import pytest
import pandas as pd
import numpy as np

# Ensure ml_testing is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ml_testing"))


SAMPLE_FEATURES = {
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

FEATURE_COLUMNS = list(SAMPLE_FEATURES.keys())

METADATA = {
    "feature_columns": FEATURE_COLUMNS,
    "numeric_columns": [
        "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10",
        "Age_Mons", "Qchat-10-Score"
    ],
    "categorical_columns": [
        "Sex", "Ethnicity", "Jaundice", "Family_mem_with_ASD",
        "Who completed the test"
    ],
    "target_column": "Class/ASD Traits",
    "target_classes": ["No", "Yes"],
    "best_model": "RandomForest",
}


def _make_mock_model(pred_value=1, prob_values=None):
    """Create a mock sklearn pipeline."""
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([pred_value])
    if prob_values is None:
        prob_values = [0.08, 0.92]
    mock_model.predict_proba.return_value = np.array([prob_values])
    return mock_model


def _make_mock_label_encoder(classes=None):
    """Create a mock label encoder."""
    if classes is None:
        classes = ["No", "Yes"]
    mock_le = MagicMock()
    mock_le.inverse_transform.side_effect = lambda x: [classes[i] for i in x]
    return mock_le


def test_predict_from_dict_returns_prediction():
    """Test that predict_from_dict returns a prediction string."""
    from app import services

    mock_model = _make_mock_model(pred_value=1)
    mock_le = _make_mock_label_encoder()

    with patch.object(services, "_model", mock_model), \
         patch.object(services, "_le", mock_le), \
         patch.object(services, "_meta", METADATA):
        result = services.predict_from_dict(SAMPLE_FEATURES)

    assert "prediction" in result
    assert result["prediction"] in ("No", "Yes")


def test_predict_from_dict_returns_probability():
    """Test that predict_from_dict returns a probability when model supports it."""
    from app import services

    mock_model = _make_mock_model(pred_value=1, prob_values=[0.08, 0.92])
    mock_le = _make_mock_label_encoder()

    with patch.object(services, "_model", mock_model), \
         patch.object(services, "_le", mock_le), \
         patch.object(services, "_meta", METADATA):
        result = services.predict_from_dict(SAMPLE_FEATURES)

    assert result.get("probability") is not None
    assert 0.0 <= result["probability"] <= 1.0


def test_predict_from_dict_missing_feature():
    """Test that missing features raise a KeyError."""
    from app import services

    mock_model = _make_mock_model()
    mock_le = _make_mock_label_encoder()

    incomplete_features = {k: v for k, v in SAMPLE_FEATURES.items() if k != "A1"}

    with patch.object(services, "_model", mock_model), \
         patch.object(services, "_le", mock_le), \
         patch.object(services, "_meta", METADATA):
        with pytest.raises(KeyError):
            services.predict_from_dict(incomplete_features)


def test_predict_from_dict_no_probability():
    """Test predict_from_dict when model does not support predict_proba."""
    from app import services

    mock_model = _make_mock_model(pred_value=0)
    del mock_model.predict_proba  # Remove predict_proba to simulate no-proba model
    mock_le = _make_mock_label_encoder()

    with patch.object(services, "_model", mock_model), \
         patch.object(services, "_le", mock_le), \
         patch.object(services, "_meta", METADATA):
        result = services.predict_from_dict(SAMPLE_FEATURES)

    assert "prediction" in result
    assert result.get("probability") is None


def test_load_once_raises_when_model_missing():
    """Test that load_once raises FileNotFoundError when model file is absent."""
    from app import services
    import importlib

    # Reset the module-level globals so load_once will try to load
    original_model = services._model
    original_le = services._le
    original_meta = services._meta

    try:
        services._model = None
        services._le = None
        services._meta = None
        with patch("pathlib.Path.exists", return_value=False):
            with pytest.raises(FileNotFoundError):
                services.load_once()
    finally:
        services._model = original_model
        services._le = original_le
        services._meta = original_meta
