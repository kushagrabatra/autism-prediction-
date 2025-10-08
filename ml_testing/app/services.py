import pickle
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "model" / "model.pkl"
LABEL_PATH = ROOT / "model" / "label_encoder.pkl"
META_PATH = ROOT / "model" / "metadata.json"

_model = None
_le = None
_meta = None

def load_once():
    global _model, _le, _meta
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
    if _le is None:
        if not LABEL_PATH.exists():
            raise FileNotFoundError(f"Label encoder file not found at {LABEL_PATH}")
        with open(LABEL_PATH, "rb") as f:
            _le = pickle.load(f)
    if _meta is None:
        if not META_PATH.exists():
            raise FileNotFoundError(f"Metadata file not found at {META_PATH}")
        with open(META_PATH, "r", encoding="utf-8") as f:
            _meta = json.load(f)
    return _model, _le, _meta

def predict_from_dict(features: dict):
    model, le, meta = load_once()
    feature_cols = meta.get("feature_columns", [])

    missing = [c for c in feature_cols if c not in features]
    if missing:
        raise KeyError(f"Missing features: {missing}")

    # Build one-row DataFrame with correct column order/names
    row_dict = {col: features[col] for col in feature_cols}
    df_row = pd.DataFrame([row_dict], columns=feature_cols)

    try:
        pred_arr = model.predict(df_row)
    except Exception as e:
        raise RuntimeError(f"Model prediction failed: {e}")

    pred = pred_arr[0]
    prob = None
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(df_row)[0]
            prob = float(probs.max())
        except Exception:
            prob = None

    try:
        label = le.inverse_transform([int(pred)])[0]
    except Exception:
        label = str(pred)

    return {"prediction": str(label), "probability": prob}
