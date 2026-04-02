"""ML prediction and metadata API routes.

All routes are prefixed with ``/ml`` and grouped under the ``ml`` tag.
"""

import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.schema import PredictRequest, PredictResponse
from app import services
from app.config import METADATA_PATH

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ml", tags=["ml"])


# -------------------- PREDICT ENDPOINT -------------------- #


@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Run ASD screening prediction",
    description=(
        "Submit demographic and behavioural screening answers to receive a "
        "model prediction. ⚠️ This is **not** a medical diagnosis tool."
    ),
)
async def predict(req: PredictRequest) -> PredictResponse:
    """Return an ASD-trait prediction for the supplied feature set.

    Args:
        req: A :class:`~app.schema.PredictRequest` validated by Pydantic.

    Returns:
        A :class:`~app.schema.PredictResponse` with ``prediction`` and
        optional ``probability``.

    Raises:
        HTTPException 422: Returned automatically by FastAPI when the request
            body fails Pydantic validation.
        HTTPException 500: When a model artifact is missing or inference fails.
    """
    logger.info("Received prediction request with %d features", len(req.features))
    try:
        res = services.predict_from_dict(req.features)
        return PredictResponse(
            prediction=res["prediction"], probability=res.get("probability")
        )
    except FileNotFoundError as exc:
        logger.error("Model artifact missing: %s", exc)
        raise HTTPException(status_code=500, detail="Model file not found") from exc
    except KeyError as exc:
        logger.warning("Missing features in request: %s", exc)
        raise HTTPException(
            status_code=422, detail=f"Missing required features: {exc}"
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected prediction error")
        raise HTTPException(
            status_code=500, detail=f"Prediction error: {exc}"
        ) from exc


# -------------------- METADATA ENDPOINT -------------------- #


@router.get(
    "/metadata",
    summary="Return model metadata",
    description=(
        "Returns information about the trained model: feature names, "
        "target classes, and performance metrics."
    ),
)
async def get_metadata() -> dict:
    """Return model metadata (feature names, classes, performance scores).

    Raises:
        HTTPException 404: When ``metadata.json`` is not found.
        HTTPException 500: When the metadata file cannot be parsed.
    """
    if not METADATA_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=f"metadata.json not found at {METADATA_PATH}",
        )
    try:
        with open(METADATA_PATH, "r") as fh:
            meta = json.load(fh)
        return meta
    except Exception as exc:
        logger.exception("Error reading metadata.json")
        raise HTTPException(
            status_code=500, detail=f"Error reading metadata.json: {exc}"
        ) from exc
