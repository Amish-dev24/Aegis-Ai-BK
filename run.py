"""
Simple script to run the FastAPI application.
"""
import uvicorn
from app.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        ssl_keyfile=settings.SSL_KEY_PATH if settings.SSL_KEY_PATH else None,
        ssl_certfile=settings.SSL_CERT_PATH if settings.SSL_CERT_PATH else None
    )

