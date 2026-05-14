"""
Alert model for managing security alerts.
"""

import enum

from sqlalchemy import Boolean, Column, DateTime, Enum, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.database import Base


class AlertStatus(str, enum.Enum):
    """Alert status."""

    PENDING = "pending"
    SENT = "sent"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    FALSE_POSITIVE = "false_positive"


class Alert(Base):
    """Alert model for security notifications."""

    __tablename__ = "alerts"

    __table_args__ = (
        Index("ix_alerts_company_created_at", "company_id", "created_at"),
        Index("ix_alerts_detection_id", "detection_id"),
        Index("ix_alerts_status", "status"),
    )

    id = Column(Integer, primary_key=True, index=True)
    detection_id = Column(Integer, ForeignKey("detections.id"), nullable=False)
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=False)
    title = Column(String(200), nullable=False)
    message = Column(Text)
    status = Column(Enum(AlertStatus), default=AlertStatus.PENDING)
    email_sent = Column(Boolean, default=False)
    email_sent_to = Column(String(500), nullable=True)  # Comma-separated recipient emails
    email_sent_at = Column(DateTime(timezone=True), nullable=True)
    acknowledged_by = Column(String(100), nullable=True)
    acknowledged_at = Column(DateTime(timezone=True), nullable=True)
    resolved_by = Column(String(100), nullable=True)
    resolved_at = Column(DateTime(timezone=True), nullable=True)
    false_positive_by = Column(String(100), nullable=True)
    false_positive_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    detection = relationship("Detection", back_populates="alerts")
    company = relationship("Company", back_populates="alerts")
    logs = relationship("AlertLog", back_populates="alert", order_by="AlertLog.created_at")
