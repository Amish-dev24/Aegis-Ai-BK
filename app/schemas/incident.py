"""
Pydantic schemas for incident management.
"""
from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List
from app.models.incident import IncidentStatus, IncidentSeverity


class IncidentCreate(BaseModel):
    title: str
    description: Optional[str] = None
    severity: Optional[IncidentSeverity] = IncidentSeverity.MEDIUM
    assigned_to: Optional[str] = None
    alert_ids: List[int] = []


class IncidentUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    status: Optional[IncidentStatus] = None
    severity: Optional[IncidentSeverity] = None
    assigned_to: Optional[str] = None


class IncidentAddAlerts(BaseModel):
    alert_ids: List[int]


class IncidentAlertSummary(BaseModel):
    id: int
    title: str
    status: str
    detection_type: Optional[str] = None
    threat_level: Optional[str] = None
    confidence: Optional[float] = None
    camera_name: Optional[str] = None
    evidence_image_url: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class IncidentResponse(BaseModel):
    id: int
    company_id: int
    title: str
    description: Optional[str] = None
    status: IncidentStatus
    severity: IncidentSeverity
    assigned_to: Optional[str] = None
    created_by: int
    created_by_username: Optional[str] = None
    alert_count: int = 0
    resolved_at: Optional[datetime] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class IncidentDetailResponse(IncidentResponse):
    """Full incident with linked alerts and their evidence."""
    alerts: List[IncidentAlertSummary] = []
