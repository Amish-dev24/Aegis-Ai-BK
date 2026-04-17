"""
Main FastAPI application entry point.
"""
import logging
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import uvicorn
from app.config import settings
from app.database import engine, Base
from app.api.v1 import api_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events for startup and shutdown."""
    # Warn about insecure defaults
    if settings.SECRET_KEY == "your-secret-key-change-in-production":
        logger.warning("SECRET_KEY is set to the default value – change it before deploying!")
    # Startup: Create database tables
    Base.metadata.create_all(bind=engine)

    # Reload completed video jobs from disk so results survive process restarts
    try:
        from app.api.v1.video_processing import load_jobs_from_disk
        load_jobs_from_disk()
    except Exception as e:
        logger.warning("Video job restore failed: %s", e)

    # Run data retention cleanup on startup (non-blocking)
    if settings.DATA_RETENTION_DAYS > 0:
        try:
            from app.database import SessionLocal
            from app.services.retention_service import run_retention_cleanup
            db = SessionLocal()
            result = run_retention_cleanup(db)
            logger.info("Startup retention cleanup: %s", result.get("message", "done"))
            db.close()
        except Exception as e:
            logger.warning("Startup retention cleanup failed: %s", e)

    yield
    # Shutdown: stop inference worker threads (avoids orphaned processes on reload / kill)
    try:
        from app.services.detection_service import detection_service
        detection_service._inference_pool.shutdown(wait=True)
    except Exception as e:
        logger.warning("Inference executor shutdown: %s", e)


app = FastAPI(
    title="Aegis AI - Intelligent Surveillance Platform",
    description="End-to-end intelligent surveillance platform with AI-powered threat detection",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware — restrict origins in production
allowed_origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix=settings.API_V1_PREFIX)

# Serve evidence images as static files
app.mount("/evidence", StaticFiles(directory=settings.EVIDENCE_DIR), name="evidence")

# Serve processed videos directly as a deployment-friendly fallback.
app.mount("/processed_videos", StaticFiles(directory=settings.PROCESSED_VIDEO_DIR), name="processed_videos")


@app.middleware("http")
async def add_cache_headers(request: Request, call_next):
    """Add browser-cache headers for static image/video assets so clients
    don't re-download the same files on every page visit."""
    response = await call_next(request)
    path = request.url.path
    if path.startswith("/evidence/") or path.startswith("/processed_videos/"):
        # Evidence images and processed videos are immutable once written.
        response.headers["Cache-Control"] = "private, max-age=86400"
    return response


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Aegis AI Surveillance Platform API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler — never leak internal details in production."""
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    detail = f"Internal server error: {str(exc)}" if settings.DEBUG else "Internal server error"
    return JSONResponse(
        status_code=500,
        content={"detail": detail},
    )


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        ssl_keyfile=settings.SSL_KEY_PATH if settings.SSL_KEY_PATH else None,
        ssl_certfile=settings.SSL_CERT_PATH if settings.SSL_CERT_PATH else None
    )

