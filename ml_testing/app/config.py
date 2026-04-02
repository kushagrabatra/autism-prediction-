"""Centralized configuration for the Autism Prediction API.

Settings can be overridden via environment variables or a .env file.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Base paths
# ---------------------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parents[1]
MODEL_DIR: Path = BASE_DIR / "model"
FRONTEND_DIR: Path = BASE_DIR / "frontend"

# ---------------------------------------------------------------------------
# Model artifact paths (can be overridden via environment variables)
# ---------------------------------------------------------------------------
MODEL_PATH: Path = Path(os.getenv("MODEL_PATH", str(MODEL_DIR / "model.pkl")))
LABEL_ENCODER_PATH: Path = Path(
    os.getenv("LABEL_ENCODER_PATH", str(MODEL_DIR / "label_encoder.pkl"))
)
METADATA_PATH: Path = Path(
    os.getenv("METADATA_PATH", str(MODEL_DIR / "metadata.json"))
)

# ---------------------------------------------------------------------------
# API settings
# ---------------------------------------------------------------------------
APP_TITLE: str = os.getenv("APP_TITLE", "Autism Prediction API")
APP_DESCRIPTION: str = os.getenv(
    "APP_DESCRIPTION",
    (
        "A machine-learning-based screening API for autism traits in toddlers. "
        "⚠️ This tool is for educational/research purposes only and is NOT a "
        "substitute for professional medical diagnosis."
    ),
)
APP_VERSION: str = os.getenv("APP_VERSION", "1.0.0")
API_HOST: str = os.getenv("API_HOST", "127.0.0.1")
API_PORT: int = int(os.getenv("API_PORT", "8000"))
DEBUG: bool = os.getenv("DEBUG", "false").lower() in ("1", "true", "yes")

# ---------------------------------------------------------------------------
# CORS settings
# ---------------------------------------------------------------------------
ALLOWED_ORIGINS: list[str] = os.getenv("ALLOWED_ORIGINS", "*").split(",")
