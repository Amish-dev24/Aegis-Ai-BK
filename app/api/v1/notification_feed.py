"""User-visible notification feed (policy changes, etc.) for the app header."""

from fastapi import APIRouter, Depends, Query
from sqlalchemy import and_, or_
from sqlalchemy.orm import Session

from app.core.security import require_any_authenticated
from app.database import get_db
from app.models.notification_feed import NotificationAudience, NotificationFeed
from app.models.user import User
from app.schemas.notification_feed import NotificationFeedResponse

router = APIRouter(prefix="/notifications", tags=["notifications"])


@router.get("/feed", response_model=list[NotificationFeedResponse])
async def list_notification_feed(
    limit: int = Query(40, ge=1, le=200),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_any_authenticated),
):
    """Recent broadcast notifications scoped to this user (platform-wide + own company).

    Uses the same JWT session as alerts so other users polling this endpoint see updates
    when an admin toggles global or company detection modules.
    """
    audience_filters = [NotificationFeed.audience == NotificationAudience.PLATFORM.value]
    if current_user.company_id:
        audience_filters.append(
            and_(
                NotificationFeed.audience == NotificationAudience.COMPANY.value,
                NotificationFeed.company_id == current_user.company_id,
            )
        )

    scoped = (
        db.query(NotificationFeed)
        .filter(or_(*audience_filters))
        .order_by(NotificationFeed.created_at.desc())
        .limit(limit)
        .all()
    )
    return scoped
