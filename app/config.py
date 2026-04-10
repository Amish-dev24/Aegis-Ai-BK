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
    MODEL_PATH: str = "./models/weapon-box-bags-v3.pt"         # YOLO — Bags, Box, Weapons (v3)
    FACE_MODEL_PATH: str = "./models/face_detection.pt"        # YOLOv8 — covered, uncovered
    # Prefer .pth + auto single-file ONNX if split .onnx/.onnx.data is broken; or set path to merged .onnx only
    CROWD_MODEL_PATH: str = "./models/sanet_partB_best.onnx"
    # "sanet" (default, ShanghaiTech Part-B) | "csrnet" (ImageNet norm + VGG CSRNet weights)
    CROWD_MODEL_ARCH: str = "sanet"
    CROWD_INFERENCE_MAX_SIDE: int = 512  # Speed profile for CPU; increase only if you need more detail
    # Performance: run SANet every N analyzed frames and reuse previous crowd result
    CROWD_EVERY_N_ANALYSIS_FRAMES: int = 20
    # Max wait for one crowd inference task before reusing last result
    CROWD_INFERENCE_TIMEOUT_SECONDS: int = 20
    # Persistence tuning: skip heavy snapshot/evidence writes for crowd detections by default
    CROWD_SAVE_EVIDENCE: bool = True
    # False = resize SANet + YOLO calibration to CROWD_INFERENCE_MAX_SIDE (recommended on CPU). True = notebook full-res (slow)
    CROWD_SANET_FULL_FRAME: bool = False
    # Match notebook: YOLO(first_frame, verbose=False), count cls==0, SCALE=person_count/raw_sum once
    CROWD_SANET_NOTEBOOK_CALIBRATION: bool = True
    # SANet: scale raw map sum to match YOLO person count (see flags above)
    CROWD_AUTO_CALIBRATE: bool = True
    # Local .pt path if you have it; if missing, Ultralytics auto-downloads yolov8n.pt (network once)
    CROWD_CALIBRATION_YOLO_PATH: str = "./models/yolov8n.pt"
    # Used only when CROWD_SANET_NOTEBOOK_CALIBRATION is false (multi-frame tuning)
    CROWD_CALIBRATION_YOLO_CONF: float = 0.45
    CROWD_CALIBRATION_YOLO_IMGSZ: int = 960
    CROWD_CALIBRATION_SAMPLE_FRAMES: int = 8
    # If you know headcount (e.g. 22), set this to skip YOLO drift: scale = EXPECTED / raw_sum
    CROWD_CALIBRATION_EXPECTED_COUNT: Optional[int] = None
    CROWD_CALIBRATION_SCALE: float = 1.0  # Used when auto-calibration is off or YOLO weights are missing
    # Heatmap (matches SANet notebook: resize → clip → /max → COLORMAP_JET → blend)
    CROWD_HEATMAP_COLORMAP: str = "jet"  # jet | hot | inferno
    CROWD_HEATMAP_FRAME_WEIGHT: float = 0.65  # cv2.addWeighted: frame alpha (overlay gets 1 - this)
    # Terminal debug for video pipeline (count, density, cached, persisted)
    CROWD_DEBUG_LOG: bool = True
    # Threat levels for crowd: use estimated people count vs density ratio
    CROWD_THREAT_USE_PEOPLE_COUNT: bool = True
    # value = min(1, count / CROWD_THREAT_MAX_PEOPLE_SCALE) for threshold compare (e.g. 0.7 high ≈ 70 people if scale=100)
    CROWD_THREAT_MAX_PEOPLE_SCALE: float = 100.0
    POSE_MODEL_PATH: str = "./models/pose_estimation.pb"       # MediaPipe (optional)
    CONFIDENCE_THRESHOLD: float = 0.5
    ABANDONED_OBJECT_THRESHOLD_SECONDS: int = 60
    # Weapon YOLO max inference side when using PyTorch weights (no ONNX fixed-input cap)
    WEAPON_YOLO_IMGSZ: int = 640
    # Video resize max side when weapon module runs (640 matches typical ONNX export on CPU)
    WEAPON_VIDEO_AI_MAX_SIDE: int = 640
    # Fixed export size for ONNX (must match exported graph)
    ONNX_YOLO_IMGSZ: int = 640
    # ONNX Runtime: use CUDA when ``onnxruntime-gpu`` is installed and a GPU is visible
    ONNX_PREFER_GPU: bool = True
    # Ultralytics ``.pt`` / PyTorch weights: move to CUDA when available
    TORCH_PREFER_GPU: bool = True
    
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

