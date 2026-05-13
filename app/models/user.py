"""
User and Role models for authentication and RBAC.
"""

import enum

from sqlalchemy import Boolean, Column, DateTime, Enum, ForeignKey, Integer, String
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.database import Base


class Role(str, enum.Enum):
    """User roles for RBAC."""

    AEGIS_ADMIN = "aegis_admin"  # Platform admin; operational data scoped to assigned company_id
    ADMIN = "admin"  # Company admin - full control within their company
    SECURITY_OFFICER = "security_officer"  # Company security officer - operational role
    VIEWER = "viewer"  # Read-only access within their company


class User(Base):
    """User model for authentication and authorization."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(100))
    phone_number = Column(String(20), nullable=True)
    alert_preference = Column(
        String(20), default="email", nullable=False
    )  # "email", "email_sms", "sms"
    role = Column(Enum(Role), default=Role.SECURITY_OFFICER, nullable=False)
    company_id = Column(
        Integer, ForeignKey("companies.id"), nullable=True, index=True
    )  # NULL for Aegis AI admins, set after admin verification
    is_active = Column(Boolean, default=True, index=True)
    admin_verified = Column(Boolean, default=False)  # Verified by admin
    email_verified = Column(Boolean, default=False)  # Email verified
    phone_verified = Column(Boolean, default=False)  # Phone number verified
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_login = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    company = relationship("Company", back_populates="users")
    audit_logs = relationship("AuditLog", back_populates="user")
