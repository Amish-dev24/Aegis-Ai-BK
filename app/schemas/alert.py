"""
Pydantic schemas for alert management.
"""
from pydantic import BaseModel
from datetime import datetime
from typing import Optional
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
    email_sent_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    created_at: datetime
    
    class Config:
        from_attributes = True

