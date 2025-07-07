# file: app/main.py

import os
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
# --- CORRECT: Import the initializer from dependencies ---
from app.dependencies import initialize_global_services
from app.routers import routers

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles application startup events. This is the correct place
    to initialize all shared services for the FastAPI app.
    """
    # Call the central service initializer from dependencies.py
    await initialize_global_services()
    yield
    print("--- [FastAPI Shutdown] Application shutdown. ---")

app = FastAPI(
    lifespan=lifespan,
    title="ESG Insight Engine API",
    description="API suite for processing documents, managing graphs, and generating insights.",
    version="1.0.0"
)

# Configure CORS Middleware
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include your API routers
app.include_router(routers.router, prefix="/api/v1")

@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "Welcome to the ESG Insight Engine API"}

def start():
    """Launched with `python -m app.main`"""
    port = int(os.getenv("PORT", 8000))
    print(f"Starting Uvicorn server on http://0.0.0.0:{port}")
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)

if __name__ == "__main__":
    start()