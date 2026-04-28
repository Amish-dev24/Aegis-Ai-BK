"""
Pydantic schemas for audit logs.
"""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel


class AuditLogBase(BaseModel):
    """Base audit log schema."""

    action: str
    resource_type: Optional[str] = None
    resource_id: Optional[int] = None
    details: Optional[dict[str, Any]] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


class AuditLogResponse(AuditLogBase):
    """Schema for audit log response."""

    id: int
    user_id: Optional[int] = None
    created_at: datetime

    class Config:
        from_attributes = True
