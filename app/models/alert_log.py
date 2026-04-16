"""
Alert log model for tracking actions/notes on alerts.
"""
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base


class AlertLog(Base):
    """Log entries for actions taken on alerts."""
    __tablename__ = "alert_logs"

    id = Column(Integer, primary_key=True, index=True)
    alert_id = Column(Integer, ForeignKey("alerts.id"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    action = Column(String(100), nullable=False)  # e.g. "note", "dispatched", "escalated", "status_change"
    message = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    alert = relationship("Alert", back_populates="logs")
    user = relationship("User")
