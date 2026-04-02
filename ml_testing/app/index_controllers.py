"""Root/health-check router for the Autism Prediction API."""

import logging

from fastapi import APIRouter

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", tags=["health"], summary="Health check")
async def health_check() -> dict:
    """Return a simple health-check payload.

    Returns:
        A dict with ``status`` and a short ``message``.
    """
    return {"status": "ok", "message": "Autism Prediction API is running"}
