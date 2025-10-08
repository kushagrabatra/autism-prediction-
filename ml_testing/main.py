from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import uvicorn

# import routers
from app.index_controllers import router as index_router
from app.controllers import router as ml_router

# --- Path Setup ---
BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"   # <-- this is where your index.html is

# --- FastAPI App ---
def create_app():
    app = FastAPI(title="ml_test")

    # Enable CORS (important for frontend)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # include routers
    app.include_router(index_router)
    app.include_router(ml_router)

    # Mount static frontend
    if FRONTEND_DIR.exists():
        app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")
    else:
        print(f"⚠️ Frontend directory not found: {FRONTEND_DIR}")

    # Root route helper
    @app.get("/", include_in_schema=False)
    async def root():
        if (FRONTEND_DIR / "index.html").exists():
            return {"message": "Frontend is available at /frontend/index.html"}
        raise HTTPException(status_code=404, detail="Frontend not found")

    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
