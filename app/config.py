"""
Configuration settings for Aegis AI backend.
"""
from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Database
    DATABASE_URL: str = "postgresql://postgres:1234@localhost:5432/aegis_db"
    
    # Security
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_HOURS: int = 2
    
    # Email
    SMTP_HOST: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    SMTP_USER: str = ""
    SMTP_PASSWORD: str = ""
    SMTP_FROM_EMAIL: str = "noreply@aegisai.com"
    SMTP_FROM_NAME: str = "Aegis AI Security System"
    
    # Detection Models
    MODEL_PATH: str = "./models/weapon_detection_v2.onnx"      # YOLO11m — Bags, Box, Weapons (v2)
    FACE_MODEL_PATH: str = "./models/face_detection.pt"        # YOLOv8 — covered, uncovered
    CROWD_MODEL_PATH: str = "./models/csrnet_crowd.pth.tar"    # CSRNet density estimator
    CROWD_INFERENCE_MAX_SIDE: int = 512  # CSRNet input: longer side in px (4:3, e.g. 512x384)
    POSE_MODEL_PATH: str = "./models/pose_estimation.pb"       # MediaPipe (optional)
    CONFIDENCE_THRESHOLD: float = 0.5
    ABANDONED_OBJECT_THRESHOLD_SECONDS: int = 60
    
    # File Storage
    UPLOAD_DIR: str = "./uploads"
    EVIDENCE_DIR: str = "./evidence"
    PROCESSED_VIDEO_DIR: str = "./processed_videos"
    MAX_FILE_SIZE_MB: int = 500
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # Application
    API_V1_PREFIX: str = "/api/v1"
    DEBUG: bool = True
    ENVIRONMENT: str = "development"
    ALLOWED_ORIGINS: str = ""  # Comma-separated origins for CORS in production
    
    # Twilio SMS
    TWILIO_ACCOUNT_SID: str = ""
    TWILIO_AUTH_TOKEN: str = ""
    TWILIO_PHONE_NUMBER: str = ""  # Twilio phone number to send FROM

    # Data Retention (auto-cleanup of old evidence/detections)
    DATA_RETENTION_DAYS: int = 90  # Delete data older than this (0 = never delete)

    # TLS/SSL
    SSL_CERT_PATH: Optional[str] = None
    SSL_KEY_PATH: Optional[str] = None
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

# Create directories if they don't exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.EVIDENCE_DIR, exist_ok=True)
os.makedirs(settings.PROCESSED_VIDEO_DIR, exist_ok=True)
os.makedirs("./models", exist_ok=True)

