"""
Pydantic schemas for ZoneOfficerAssignment.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class ZoneOfficerAssignmentBase(BaseModel):
    zone_name: str = Field(
        ..., min_length=1, max_length=100, description="Zone label (must match Camera.zone)"
    )
    officer_id: int = Field(..., description="ID of the User with role SECURITY_OFFICER")


class ZoneOfficerAssignmentCreate(ZoneOfficerAssignmentBase):
    pass


class ZoneOfficerAssignmentUpdate(BaseModel):
    officer_id: int = Field(..., description="ID of the new officer to assign")


class OfficerInfo(BaseModel):
    id: int
    username: str
    full_name: Optional[str] = None
    email: Optional[str] = None

    class Config:
        from_attributes = True


class ZoneOfficerAssignmentResponse(ZoneOfficerAssignmentBase):
    id: int
    company_id: int
    created_at: datetime
    updated_at: Optional[datetime] = None
    officer: Optional[OfficerInfo] = None

    class Config:
        from_attributes = True
