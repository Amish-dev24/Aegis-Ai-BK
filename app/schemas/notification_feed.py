"""Schemas for detection-policy / broadcast notifications."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class NotificationFeedResponse(BaseModel):
    id: int
    audience: str
    company_id: Optional[int] = None
    title: str
    body: Optional[str] = None
    actor_username: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True
