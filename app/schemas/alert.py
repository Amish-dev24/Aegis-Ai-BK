"""
Pydantic schemas for alert management.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel

from app.models.alert import AlertStatus


class AlertBase(BaseModel):
    """Base alert schema."""

    title: str
    message: Optional[str] = None


class AlertCreate(AlertBase):
    """Schema for creating an alert."""

    detection_id: int


class AlertUpdate(BaseModel):
    """Schema for updating an alert."""

    status: Optional[AlertStatus] = None
    message: Optional[str] = None


class AlertResponse(AlertBase):
    """Schema for alert response."""

    id: int
    detection_id: int
    company_id: int
    status: AlertStatus
    email_sent: bool
    email_sent_to: Optional[str] = None
    email_sent_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    false_positive_by: Optional[str] = None
    false_positive_at: Optional[datetime] = None
    created_at: datetime

    class Config:
        from_attributes = True


# --- Alert Log schemas ---


class AlertLogCreate(BaseModel):
    action: str = "note"  # note, dispatched, escalated, status_change, etc.
    message: str


class AlertLogResponse(BaseModel):
    id: int
    alert_id: int
    user_id: int
    username: Optional[str] = None
    action: str
    message: str
    created_at: datetime

    class Config:
        from_attributes = True


# --- Enriched alert with detection context + logs ---


class AlertDetailResponse(AlertResponse):
    """Enriched alert response with detection context, camera info, evidence image, and logs."""

    # Detection context
    detection_type: Optional[str] = None
    threat_level: Optional[str] = None
    confidence: Optional[float] = None
    detection_timestamp: Optional[datetime] = None

    # Camera info
    camera_id: Optional[int] = None
    camera_name: Optional[str] = None
    camera_location: Optional[str] = None

    # Evidence snapshot
    evidence_id: Optional[int] = None
    evidence_image_url: Optional[str] = None

    # Incident logs
    logs: list[AlertLogResponse] = []

    # Latest incident note (AlertLog) — for header feed without loading full logs
    latest_note: Optional[str] = None
    latest_note_at: Optional[datetime] = None
