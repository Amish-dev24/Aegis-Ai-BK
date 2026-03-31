"""
Emergency contact directory for quick-dial during incidents.
"""
from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base


class EmergencyContact(Base):
    """Emergency contact for a company — shown during critical alerts."""
    __tablename__ = "emergency_contacts"

    id = Column(Integer, primary_key=True, index=True)
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=False)
    name = Column(String(100), nullable=False)          # e.g. "Police", "Fire Dept", "Head of Security"
    phone_number = Column(String(20), nullable=False)    # e.g. "+923001234567"
    role = Column(String(50), nullable=True)              # e.g. "Police", "Security", "Medical"
    is_primary = Column(Boolean, default=False)           # Primary contact called first
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    company = relationship("Company")
