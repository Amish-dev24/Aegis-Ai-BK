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


class EvidenceResponse(EvidenceBase):
    """Schema for evidence response."""
    id: int
    created_at: datetime
    
    class Config:
        from_attributes = True

