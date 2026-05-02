"""
Cross-user notifications for detection policy changes and similar broadcasts.

Fed to the app header bell / activity feed alongside security alerts.
"""

import enum

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.sql import func

from app.database import Base


class NotificationAudience(str, enum.Enum):
    PLATFORM = "platform"  # all tenant users see (company staff + optional aegis filtering)
    COMPANY = "company"  # only users of company_id


class NotificationFeed(Base):
    __tablename__ = "notification_feed"

    id = Column(Integer, primary_key=True, index=True)
    audience = Column(String(20), nullable=False, index=True)
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=True, index=True)
    title = Column(String(200), nullable=False)
    body = Column(Text, nullable=True)
    actor_username = Column(String(150), nullable=True)
    actor_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
