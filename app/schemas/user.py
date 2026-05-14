"""
Pydantic schemas for user management.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr

from app.models.user import Role


class UserBase(BaseModel):
    """Base user schema."""

    username: str
    email: EmailStr
    full_name: Optional[str] = None
    phone_number: Optional[str] = None
    alert_preference: str = "email"  # "email", "email_sms", "sms"
    zone: Optional[str] = None  # Assigned zone label (e.g. "Lobby", "Gate A")
    role: Role = Role.VIEWER


class UserSignup(BaseModel):
    """Schema for public signup (no login required)."""

    username: str
    email: EmailStr
    password: str
    full_name: Optional[str] = None
    phone_number: Optional[str] = None
    alert_preference: str = "email"
    zone: Optional[str] = None
    company_id: Optional[int] = None


class UserCreate(UserBase):
    """Schema for creating a user (admin only)."""

    password: str
    company_id: Optional[int] = None


class UserUpdate(BaseModel):
    """Schema for updating a user."""

    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    phone_number: Optional[str] = None
    alert_preference: Optional[str] = None
    zone: Optional[str] = None
    role: Optional[Role] = None
    is_active: Optional[bool] = None
    admin_verified: Optional[bool] = None
    email_verified: Optional[bool] = None
    phone_verified: Optional[bool] = None
    password: Optional[str] = None
    company_id: Optional[int] = None


class UserResponse(UserBase):
    """Schema for user response."""

    id: int
    company_id: Optional[int] = None
    zone: Optional[str] = None
    is_active: bool
    admin_verified: bool = False
    email_verified: bool = False
    phone_verified: bool = False
    alert_preference: str = "email"
    created_at: datetime
    updated_at: Optional[datetime] = None
    last_login: Optional[datetime] = None

    class Config:
        from_attributes = True


class Token(BaseModel):
    """Token response schema."""

    access_token: str
    refresh_token: str
    token_type: str


class TokenData(BaseModel):
    """Token data schema."""

    username: Optional[str] = None
