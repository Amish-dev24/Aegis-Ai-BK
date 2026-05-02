"""Write helper for NotificationFeed rows (called from detection settings API)."""

from sqlalchemy.orm import Session

from app.models.notification_feed import NotificationAudience, NotificationFeed


def record_platform_module_change(
    db: Session,
    *,
    module_name: str,
    is_enabled: bool,
    actor_username: str,
    actor_user_id: int,
) -> NotificationFeed:
    state = "enabled" if is_enabled else "disabled"
    row = NotificationFeed(
        audience=NotificationAudience.PLATFORM.value,
        company_id=None,
        title=f"Global module {module_name} {state}",
        body=(
            f"The Aegis administrator set platform-wide detection '{module_name}' to {state}. "
            "This affects all organisations using the cloud models."
        ),
        actor_username=actor_username,
        actor_user_id=actor_user_id,
    )
    db.add(row)
    return row


def record_company_module_change(
    db: Session,
    *,
    company_id: int,
    module_name: str,
    message: str,
    actor_username: str,
    actor_user_id: int,
) -> NotificationFeed:
    row = NotificationFeed(
        audience=NotificationAudience.COMPANY.value,
        company_id=company_id,
        title=f"Detection settings · {module_name}",
        body=message,
        actor_username=actor_username,
        actor_user_id=actor_user_id,
    )
    db.add(row)
    return row
