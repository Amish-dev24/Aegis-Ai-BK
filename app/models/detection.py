"""
Detection models for storing AI detection results.
"""

import enum

from sqlalchemy import JSON, Column, DateTime, Enum, Float, ForeignKey, Index, Integer
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.database import Base


class DetectionType(str, enum.Enum):
    """Types of detections."""

    WEAPON = "weapon"
    VIOLENCE = "violence"
    ABANDONED_OBJECT = "abandoned_object"
    MASK_FACE = "mask_face"
    CROWD_DENSITY = "crowd_density"


class ThreatLevel(str, enum.Enum):
    """Threat level classification."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Detection(Base):
    """Detection model for storing AI detection results."""

    __tablename__ = "detections"

    __table_args__ = (
        # Most queries filter by company + time range; cover both with one composite index.
        Index("ix_detections_company_detected_at", "company_id", "detected_at"),
        # Camera-level queries (live feed history, per-camera analytics).
        Index("ix_detections_camera_detected_at", "camera_id", "detected_at"),
        # Analytics GROUP BY / WHERE filters.
        Index("ix_detections_detection_type", "detection_type"),
        Index("ix_detections_threat_level", "threat_level"),
    )

    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=False)
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=False)
    detection_type = Column(Enum(DetectionType), nullable=False)
    threat_level = Column(Enum(ThreatLevel), nullable=False)
    confidence = Column(Float, nullable=False)  # Model confidence score
    frame_timestamp = Column(DateTime(timezone=True), nullable=False)
    detected_at = Column(DateTime(timezone=True), server_default=func.now())

    # Bounding box coordinates (normalized 0-1)
    bbox_x = Column(Float)
    bbox_y = Column(Float)
    bbox_width = Column(Float)
    bbox_height = Column(Float)

    # Additional metadata (JSON)
    detection_metadata = Column(JSON)  # Store pose data, object class, etc.

    # Relationships
    camera = relationship("Camera")
    company = relationship("Company", back_populates="detections")
    alerts = relationship("Alert", back_populates="detection")
    evidence = relationship("Evidence", back_populates="detection")
