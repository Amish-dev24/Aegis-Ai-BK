"""
Pydantic schemas for user management.
"""
from pydantic import BaseModel, EmailStr
from datetime import datetime
from typing import Optional
from app.models.user import Role


class UserBase(BaseModel):
    """Base user schema."""
    username: str
    email: EmailStr
    full_name: Optional[str] = None
    role: Role = Role.VIEWER


class UserCreate(UserBase):
    """Schema for creating a user."""
    password: str
    company_id: Optional[int] = None  # Required for company users, optional for Aegis AI admins


class UserUpdate(BaseModel):
    """Schema for updating a user."""
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    role: Optional[Role] = None
    is_active: Optional[bool] = None
    password: Optional[str] = None
    company_id: Optional[int] = None


class UserResponse(UserBase):
    """Schema for user response."""
    id: int
    company_id: Optional[int] = None
    is_active: bool
    created_at: datetime
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

