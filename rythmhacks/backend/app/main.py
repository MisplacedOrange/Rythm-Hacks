"""
MediLytica Backend - FastAPI Application
ML Visualization and Collaboration Platform
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

# Import core module early to load custom PyTorch models
import app.core  # noqa: F401

# Import routers
from app.api.routes import regression, decision_tree, datasets, models, code

app = FastAPI(
    title="MediLytica API",
    description="ML Visualization and Collaboration Platform - Backend API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware - Allow frontend to communicate
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server (default)
        "http://localhost:5174",  # Vite dev server (alternative)
        "http://localhost:3000",  # Alternative port
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(regression.router, tags=["Regression"])
app.include_router(decision_tree.router, tags=["Decision Trees"])
app.include_router(datasets.router, tags=["Datasets"])
app.include_router(models.router, tags=["Models"])
app.include_router(code.router, tags=["Code Execution"])

# Ensure data directories exist
@app.on_event("startup")
async def startup_event():
    """Create necessary directories on startup"""
    data_dirs = [
        Path("data/regression_sessions"),
        Path("data/tree_sessions"),
        Path("data/datasets"),
        Path("data/models"),
        Path("data/models_metadata")
    ]
    for dir_path in data_dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
    print("✅ Backend server started successfully")
    print("📊 Data directories initialized")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "MediLytica Backend",
        "status": "running",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "services": {
            "api": "operational",
            "ml_engine": "ready"
        }
    }
