"""FastAPI application entry point for the Autism Prediction service.

Run locally with::

    uvicorn main:app --reload

Or via Docker::

    docker compose up
"""

import logging
import logging.config
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import (
    ALLOWED_ORIGINS,
    API_HOST,
    API_PORT,
    APP_DESCRIPTION,
    APP_TITLE,
    APP_VERSION,
    DEBUG,
    FRONTEND_DIR,
)
from app.index_controllers import router as index_router
from app.controllers import router as ml_router

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        A fully-configured :class:`fastapi.FastAPI` instance.
    """
    app = FastAPI(
        title=APP_TITLE,
        description=APP_DESCRIPTION,
        version=APP_VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers
    app.include_router(index_router)
    app.include_router(ml_router)

    # Static frontend (optional – only if the directory exists)
    if FRONTEND_DIR.exists():
        app.mount(
            "/frontend",
            StaticFiles(directory=str(FRONTEND_DIR)),
            name="frontend",
        )
        logger.info("Frontend mounted at /frontend from %s", FRONTEND_DIR)
    else:
        logger.warning("Frontend directory not found: %s", FRONTEND_DIR)

    @app.get("/", include_in_schema=False)
    async def root() -> dict:
        """Redirect hint for the root path."""
        if (FRONTEND_DIR / "index.html").exists():
            return {
                "message": "Frontend available",
                "url": "/frontend/index.html",
                "api_docs": "/docs",
            }
        raise HTTPException(status_code=404, detail="Frontend not found")

    logger.info("App '%s' v%s created (debug=%s)", APP_TITLE, APP_VERSION, DEBUG)
    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run("main:app", host=API_HOST, port=API_PORT, reload=DEBUG)
