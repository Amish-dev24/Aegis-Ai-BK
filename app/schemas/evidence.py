"""
Pydantic schemas for evidence management.
"""
from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class EvidenceBase(BaseModel):
    """Base evidence schema."""
    detection_id: int
    image_path: str
    video_path: Optional[str] = None
    frame_number: Optional[int] = None
    metadata_json: Optional[str] = None


class EvidenceCreate(EvidenceBase):
    """Schema for creating evidence."""
    pass


class EvidenceResponse(BaseModel):
    """Schema for evidence response with detection context."""
    id: int
    detection_id: int
    image_path: str
    image_url: Optional[str] = None
    video_path: Optional[str] = None
    metadata_json: Optional[str] = None
    created_at: datetime

    # Detection context — so you know what this evidence is about
    detection_type: Optional[str] = None
    threat_level: Optional[str] = None
    confidence: Optional[float] = None
    camera_id: Optional[int] = None
    camera_name: Optional[str] = None
    detection_timestamp: Optional[datetime] = None

    class Config:
        from_attributes = True
