from fastapi import APIRouter, HTTPException
from app.schema import PredictRequest, PredictResponse
from app import services
import json
from pathlib import Path

router = APIRouter(prefix="/ml", tags=["ml"])

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"

# -------------------- PREDICT ENDPOINT -------------------- #
@router.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    try:
        res = services.predict_from_dict(req.features)
        return PredictResponse(prediction=res["prediction"], probability=res.get("probability"))
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Model file not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")


# -------------------- METADATA ENDPOINT -------------------- #
@router.get("/metadata")
async def get_metadata():
    """Return model metadata like feature names, numeric/categorical columns, etc."""
    meta_path = MODEL_DIR / "metadata.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail=f"metadata.json not found in {MODEL_DIR}")
    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
        return meta
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading metadata.json: {e}")
