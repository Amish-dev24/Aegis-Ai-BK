"""
Data retention endpoints — manage automatic cleanup of old data.

Section 9 of proposal: "Define retention period (e.g., 30-90 days)
after which evidence is archived or deleted."
"""
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from app.database import get_db
from app.config import settings
from app.core.security import require_admin
from app.models.user import User
from app.services.retention_service import run_retention_cleanup

router = APIRouter(prefix="/retention", tags=["data-retention"])


@router.get("/policy")
async def get_retention_policy(
    current_user: User = Depends(require_admin),
):
    """Get current data retention policy settings."""
    return {
        "retention_days": settings.DATA_RETENTION_DAYS,
        "policy": f"Data older than {settings.DATA_RETENTION_DAYS} days is automatically cleaned up"
        if settings.DATA_RETENTION_DAYS > 0
        else "Data retention is disabled (data kept indefinitely)",
    }


@router.post("/cleanup")
async def trigger_cleanup(
    days: int = Query(None, description="Override retention days (default: use config)"),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """
    Manually trigger data retention cleanup.
    Deletes detections, evidence, alerts, and files older than the retention period.
    """
    result = run_retention_cleanup(db, retention_days=days)
    return result
