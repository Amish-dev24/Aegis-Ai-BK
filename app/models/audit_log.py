"""
Audit log model for tracking user actions.
"""
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
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

