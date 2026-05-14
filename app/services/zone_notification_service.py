"""
Zone notification helper.

Resolves the full list of email recipients for a camera alert:
  1. All active ADMIN users of the camera's company.
  2. The security officer assigned to the camera's zone (if any).

Used by every detection/alert path so notification logic stays in one place.
"""

from __future__ import annotations

import logging
from typing import Optional

from sqlalchemy.orm import Session

from app.models.camera import Camera
from app.models.user import Role, User
from app.models.zone_officer_assignment import ZoneOfficerAssignment

logger = logging.getLogger(__name__)


def get_alert_emails_for_camera(
    db: Session,
    camera: Camera,
    triggering_user_email: Optional[str] = None,
) -> list[str]:
    """
    Return a deduplicated list of email addresses that should receive an alert
    triggered by *camera*.

    Recipients:
    - The user who triggered the detection/started live detection (if provided).
    - All active ADMIN users belonging to the camera's company.
    - The security officer assigned to camera.zone for that company (if any).

    Args:
        db: Active SQLAlchemy session.
        camera: The Camera ORM object.
        triggering_user_email: Optional email of the user who directly
            initiated the detection (e.g. current_user.email).

    Returns:
        Deduplicated list of non-empty email strings.
    """
    seen: set[str] = set()
    emails: list[str] = []

    def _add(addr: Optional[str]) -> None:
        if addr and addr.strip() and addr.strip() not in seen:
            seen.add(addr.strip())
            emails.append(addr.strip())

    _add(triggering_user_email)

    if not camera.company_id:
        return emails

    try:
        # Company admins
        admins = (
            db.query(User)
            .filter(
                User.company_id == camera.company_id,
                User.role == Role.ADMIN,
                User.is_active.is_(True),
                User.email.isnot(None),
            )
            .all()
        )
        for admin in admins:
            _add(admin.email)

        zone_name = (camera.zone or "").strip()
        if zone_name:
            # Source 1: explicit ZoneOfficerAssignment table
            assignment = (
                db.query(ZoneOfficerAssignment)
                .filter(
                    ZoneOfficerAssignment.company_id == camera.company_id,
                    ZoneOfficerAssignment.zone_name == zone_name,
                )
                .first()
            )
            if assignment and assignment.officer:
                _add(assignment.officer.email)

            # Source 2: security officers whose User.zone matches the camera zone
            zone_officers = (
                db.query(User)
                .filter(
                    User.company_id == camera.company_id,
                    User.role == Role.SECURITY_OFFICER,
                    User.zone == zone_name,
                    User.is_active.is_(True),
                    User.email.isnot(None),
                )
                .all()
            )
            for officer in zone_officers:
                _add(officer.email)

    except Exception as exc:
        logger.warning("get_alert_emails_for_camera error: %s", exc)

    return emails


def get_alert_emails_for_camera_id(
    db: Session,
    camera_id: int,
    triggering_user_email: Optional[str] = None,
) -> list[str]:
    """Convenience wrapper that accepts camera_id instead of Camera object."""
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if camera is None:
        if triggering_user_email:
            return [triggering_user_email]
        return []
    return get_alert_emails_for_camera(db, camera, triggering_user_email)
