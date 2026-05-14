"""
Zone-to-officer assignment model.

Each company zone (matched by Camera.zone string) can be assigned to one
security officer.  When a camera in that zone triggers an alert, both the
company admin(s) and the assigned officer receive an email notification.
"""

from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship

from app.database import Base


class ZoneOfficerAssignment(Base):
    __tablename__ = "zone_officer_assignments"

    id = Column(Integer, primary_key=True, index=True)
    company_id = Column(Integer, ForeignKey("companies.id", ondelete="CASCADE"), nullable=False, index=True)
    zone_name = Column(String(100), nullable=False)
    officer_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # One officer per zone per company
    __table_args__ = (
        UniqueConstraint("company_id", "zone_name", name="uq_zone_company"),
    )

    company = relationship("Company", backref="zone_officer_assignments")
    officer = relationship("User", backref="zone_assignments")
