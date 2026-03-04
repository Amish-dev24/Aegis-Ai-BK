"""
Pydantic schemas for detection management.
"""
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, Dict, Any
from app.models.detection import DetectionType, ThreatLevel


class DetectionBase(BaseModel):
    """Base detection schema."""
    camera_id: int
    detection_type: DetectionType
    threat_level: ThreatLevel
    confidence: float
    frame_timestamp: datetime
    bbox_x: Optional[float] = None
    bbox_y: Optional[float] = None
    bbox_width: Optional[float] = None
    bbox_height: Optional[float] = None
    detection_metadata: Optional[Dict[str, Any]] = None


class DetectionCreate(DetectionBase):
    """Schema for creating a detection."""
    pass


class DetectionResponse(DetectionBase):
    """Schema for detection response."""
    id: int
    company_id: int
    detected_at: datetime
    
    class Config:
        from_attributes = True


class DetectionFilter(BaseModel):
    """Schema for filtering detections."""
    camera_id: Optional[int] = None
    detection_type: Optional[DetectionType] = None
    threat_level: Optional[ThreatLevel] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    min_confidence: Optional[float] = None
    limit: int = 100
    offset: int = 0

