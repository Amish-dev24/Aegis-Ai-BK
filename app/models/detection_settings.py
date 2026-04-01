"""
Detection settings models for global and per-company module control.

- GlobalModuleSettings: Aegis admin controls — enable/disable entire detection modules for all users.
- CompanyDetectionSettings: Per-company controls — which modules a company wants + custom thresholds.
"""
from sqlalchemy import Column, Integer, String, Boolean, Float, DateTime, ForeignKey, UniqueConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base


class GlobalModuleSettings(Base):
    """Global on/off switch for each detection module (Aegis admin only)."""
    __tablename__ = "global_module_settings"

    id = Column(Integer, primary_key=True, index=True)
    module_name = Column(String(50), unique=True, nullable=False, index=True)  # e.g. "weapon", "mask_face"
    is_enabled = Column(Boolean, default=True, nullable=False)
    updated_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class CompanyDetectionSettings(Base):
    """Per-company detection module settings with custom thresholds."""
    __tablename__ = "company_detection_settings"
    __table_args__ = (
        UniqueConstraint("company_id", "module_name", name="uq_company_module"),
    )

    id = Column(Integer, primary_key=True, index=True)
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=False)
    module_name = Column(String(50), nullable=False)  # e.g. "weapon", "mask_face"
    is_enabled = Column(Boolean, default=True, nullable=False)

    # Custom threat-level thresholds (override defaults if set)
    critical_threshold = Column(Float, nullable=True)  # confidence >= this → CRITICAL
    high_threshold = Column(Float, nullable=True)       # confidence >= this → HIGH
    medium_threshold = Column(Float, nullable=True)     # confidence >= this → MEDIUM

    # Custom confidence threshold for the AI model itself
    min_confidence = Column(Float, nullable=True)       # ignore detections below this

    # Which threat levels trigger alerts + email (comma-separated: "medium,high,critical")
    alert_on_levels = Column(String(100), nullable=True, default="medium,high,critical")

    # Abandoned object duration threshold (seconds) — only for abandoned_object module
    abandoned_seconds = Column(Integer, nullable=True)  # default: 60

    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    company = relationship("Company")
