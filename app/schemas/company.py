"""
Pydantic schemas for company management.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr


class CompanyBase(BaseModel):
    """Base company schema."""

    name: str
    domain: Optional[str] = None
    contact_email: Optional[EmailStr] = None
    contact_phone: Optional[str] = None


class CompanyCreate(CompanyBase):
    """Schema for creating a company."""

    pass


class CompanyUpdate(BaseModel):
    """Schema for updating a company."""

    name: Optional[str] = None
    domain: Optional[str] = None
    contact_email: Optional[EmailStr] = None
    contact_phone: Optional[str] = None
    is_active: Optional[bool] = None
    is_verified: Optional[bool] = None


class CompanyResponse(CompanyBase):
    """Schema for company response."""

    id: int
    is_active: bool
    is_verified: bool
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class CompanyWithUsers(CompanyResponse):
    """Company response with user count."""

    user_count: int
    active_user_count: int


class CompanyStats(BaseModel):
    """Company statistics for Aegis AI admin."""

    company_id: int
    company_name: str
    total_users: int
    active_users: int
    total_cameras: int
    active_cameras: int
    total_detections: int
    total_alerts: int
    last_activity: Optional[datetime] = None
