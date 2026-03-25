"""
Audit log model for tracking user actions.
"""
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from fastapi import Request
from app.database import Base


class AuditLog(Base):
    """Audit log model for tracking system events."""
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    action = Column(String(100), nullable=False)  # e.g., "login", "create_alert", "export_evidence"
    resource_type = Column(String(50))  # e.g., "detection", "user", "camera"
    resource_id = Column(Integer, nullable=True)
    ip_address = Column(String(45))
    user_agent = Column(String(500))
    details = Column(JSON)  # Additional action details
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    user = relationship("User", back_populates="audit_logs")


def create_audit_log(
    request: Request,
    user_id: int,
    action: str,
    resource_type: str,
    resource_id: int = None,
    details: dict = None,
) -> AuditLog:
    """Create an AuditLog with IP address and user agent from the request."""
    ip = request.headers.get("x-forwarded-for", request.client.host if request.client else None)
    ua = request.headers.get("user-agent")
    return AuditLog(
        user_id=user_id,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        ip_address=ip,
        user_agent=ua,
        details=details,
    )

