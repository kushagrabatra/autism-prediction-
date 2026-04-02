"""ML service layer for model loading and prediction.

This module handles all machine-learning operations: loading model artifacts
from disk and running inference.  Artifacts are loaded once at startup and
cached in module-level globals so that subsequent requests are fast.
"""

import logging
import pickle
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from app.config import LABEL_ENCODER_PATH, METADATA_PATH, MODEL_PATH

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level caches – loaded lazily on first prediction request
# ---------------------------------------------------------------------------
_model: Any = None
_le: Any = None
_meta: Optional[dict] = None


def load_once() -> tuple[Any, Any, dict]:
    """Load model artifacts from disk (runs once; results are cached).

    Returns:
        A 3-tuple ``(pipeline, label_encoder, metadata_dict)``.

    Raises:
        FileNotFoundError: When any required artifact file is missing.
    """
    global _model, _le, _meta

    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        logger.info("Loading model from %s", MODEL_PATH)
        with open(MODEL_PATH, "rb") as fh:
            _model = pickle.load(fh)

    if _le is None:
        if not LABEL_ENCODER_PATH.exists():
            raise FileNotFoundError(
                f"Label encoder file not found at {LABEL_ENCODER_PATH}"
            )
        logger.info("Loading label encoder from %s", LABEL_ENCODER_PATH)
        with open(LABEL_ENCODER_PATH, "rb") as fh:
            _le = pickle.load(fh)

    if _meta is None:
        if not METADATA_PATH.exists():
            raise FileNotFoundError(f"Metadata file not found at {METADATA_PATH}")
        logger.info("Loading metadata from %s", METADATA_PATH)
        import json

        with open(METADATA_PATH, "r", encoding="utf-8") as fh:
            _meta = json.load(fh)

    return _model, _le, _meta


def predict_from_dict(features: dict[str, Any]) -> dict[str, Any]:
    """Run inference for a single observation described by *features*.

    Args:
        features: A mapping of feature name → value.  Must contain every
            column listed in the model's ``metadata.json``.

    Returns:
        A dict with keys ``"prediction"`` (str label) and
        ``"probability"`` (float or ``None``).

    Raises:
        FileNotFoundError: If model artifacts are not present.
        KeyError: If required feature columns are missing from *features*.
        RuntimeError: If the underlying model raises during inference.
    """
    model, le, meta = load_once()
    feature_cols: list[str] = meta.get("feature_columns", [])

    missing = [c for c in feature_cols if c not in features]
    if missing:
        raise KeyError(f"Missing features: {missing}")

    # Build a one-row DataFrame with the correct column order
    row_dict = {col: features[col] for col in feature_cols}
    df_row = pd.DataFrame([row_dict], columns=feature_cols)

    logger.debug("Running model inference for row: %s", row_dict)

    try:
        pred_arr = model.predict(df_row)
    except Exception as exc:
        logger.exception("Model prediction failed")
        raise RuntimeError(f"Model prediction failed: {exc}") from exc

    pred = pred_arr[0]
    prob: Optional[float] = None

    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(df_row)[0]
            prob = float(probs.max())
        except Exception:
            logger.warning("Could not compute prediction probability", exc_info=True)
            prob = None

    try:
        label: str = le.inverse_transform([int(pred)])[0]
    except Exception:
        label = str(pred)

    logger.info("Prediction: %s (probability=%.4f)", label, prob or 0.0)
    return {"prediction": str(label), "probability": prob}
