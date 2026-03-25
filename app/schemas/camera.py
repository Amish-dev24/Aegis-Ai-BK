"""
Pydantic schemas for camera management.
"""
from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class CameraBase(BaseModel):
    """Base camera schema."""
    name: str
    location: Optional[str] = None
    description: Optional[str] = None
    stream_url: Optional[str] = None  # Optional — only needed for future live streaming
    zone: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None


class CameraCreate(CameraBase):
    """Schema for creating a camera."""
    company_id: Optional[int] = None  # Required for company users, optional for Aegis AI admins


class CameraUpdate(BaseModel):
    """Schema for updating a camera."""
    name: Optional[str] = None
    location: Optional[str] = None
    description: Optional[str] = None
    stream_url: Optional[str] = None
    is_active: Optional[bool] = None
    zone: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None


class CameraResponse(CameraBase):
    """Schema for camera response."""
    id: int
    company_id: int
    is_active: bool
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

