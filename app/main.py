"""
Main FastAPI application entry point.
"""
import logging
from fastapi import FastAPI, Request
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
    # Shutdown: Cleanup if needed
    pass


app = FastAPI(
    title="Aegis AI - Intelligent Surveillance Platform",
    description="End-to-end intelligent surveillance platform with AI-powered threat detection",
    version="1.0.0",
    lifespan=lifespan,
    # Disable docs in production to reduce attack surface
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
)

# CORS middleware — restrict origins in production
allowed_origins = ["*"] if settings.DEBUG else settings.ALLOWED_ORIGINS.split(",") if settings.ALLOWED_ORIGINS else []
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

