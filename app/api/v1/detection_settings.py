"""
Detection settings endpoints.

- Global module settings (Aegis admin): enable/disable modules for ALL users.
- Company detection settings (admin/officer): per-company module toggle + custom thresholds.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.security import (
    check_company_access,
    require_admin,
    require_aegis_admin,
)
from app.database import get_db
from app.models.detection import DetectionType
from app.models.detection_settings import CompanyDetectionSettings, GlobalModuleSettings
from app.models.user import User
from app.schemas.detection_settings import (
    CompanyDetectionSettingsCreate,
    CompanyDetectionSettingsResponse,
    CompanyDetectionSettingsUpdate,
    GlobalModuleSettingsResponse,
    GlobalModuleSettingsUpdate,
)
from app.services.notification_feed_service import (
    record_company_module_change,
    record_platform_module_change,
)

router = APIRouter(prefix="/detection-settings", tags=["detection-settings"])

VALID_MODULES = [dt.value for dt in DetectionType]


# =====================================================================
# Global Module Settings (Aegis Admin only)
# =====================================================================


@router.get("/global", response_model=list[GlobalModuleSettingsResponse])
async def list_global_settings(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_aegis_admin),
):
    """List all global module settings. Creates defaults if missing."""
    _ensure_global_defaults(db, current_user.id)
    return db.query(GlobalModuleSettings).order_by(GlobalModuleSettings.module_name).all()


@router.put("/global/{module_name}", response_model=GlobalModuleSettingsResponse)
async def update_global_setting(
    module_name: str,
    data: GlobalModuleSettingsUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_aegis_admin),
):
    """Enable or disable a detection module globally (Aegis admin only)."""
    if module_name not in VALID_MODULES:
        raise HTTPException(
            status_code=400, detail=f"Invalid module: {module_name}. Valid: {VALID_MODULES}"
        )

    _ensure_global_defaults(db, current_user.id)
    setting = (
        db.query(GlobalModuleSettings)
        .filter(GlobalModuleSettings.module_name == module_name)
        .first()
    )
    setting.is_enabled = data.is_enabled
    setting.updated_by = current_user.id
    record_platform_module_change(
        db,
        module_name=module_name,
        is_enabled=data.is_enabled,
        actor_username=current_user.username,
        actor_user_id=current_user.id,
    )
    db.commit()
    db.refresh(setting)
    return setting


def _ensure_global_defaults(db: Session, user_id: int):
    """Create default global settings rows if they don't exist yet."""
    existing = {s.module_name for s in db.query(GlobalModuleSettings).all()}
    for module in VALID_MODULES:
        if module not in existing:
            db.add(GlobalModuleSettings(module_name=module, is_enabled=True, updated_by=user_id))
    db.commit()


# =====================================================================
# Company Detection Settings (Admin / Security Officer)
# =====================================================================


@router.get("/company/{company_id}", response_model=list[CompanyDetectionSettingsResponse])
async def list_company_settings(
    company_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """List detection settings for a company."""
    if not check_company_access(current_user, company_id):
        raise HTTPException(status_code=403, detail="No access to this company")

    return (
        db.query(CompanyDetectionSettings)
        .filter(CompanyDetectionSettings.company_id == company_id)
        .order_by(CompanyDetectionSettings.module_name)
        .all()
    )


@router.post(
    "/company/{company_id}", response_model=CompanyDetectionSettingsResponse, status_code=201
)
async def create_company_setting(
    company_id: int,
    data: CompanyDetectionSettingsCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """Create a detection setting for a company module."""
    if not check_company_access(current_user, company_id):
        raise HTTPException(status_code=403, detail="No access to this company")

    if data.module_name not in VALID_MODULES:
        raise HTTPException(
            status_code=400, detail=f"Invalid module: {data.module_name}. Valid: {VALID_MODULES}"
        )

    existing = (
        db.query(CompanyDetectionSettings)
        .filter(
            CompanyDetectionSettings.company_id == company_id,
            CompanyDetectionSettings.module_name == data.module_name,
        )
        .first()
    )
    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"Setting already exists for module '{data.module_name}'. Use PUT to update.",
        )

    setting = CompanyDetectionSettings(company_id=company_id, **data.dict())
    db.add(setting)
    record_company_module_change(
        db,
        company_id=company_id,
        module_name=data.module_name,
        message=(
            f"{current_user.username} added company overrides for '{data.module_name}' "
            f"(module enabled={'yes' if data.is_enabled else 'no'})."
        ),
        actor_username=current_user.username,
        actor_user_id=current_user.id,
    )
    db.commit()
    db.refresh(setting)
    return setting


@router.put("/company/{company_id}/{module_name}", response_model=CompanyDetectionSettingsResponse)
async def update_company_setting(
    company_id: int,
    module_name: str,
    data: CompanyDetectionSettingsUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """Update a company's detection module settings (toggle, thresholds)."""
    if not check_company_access(current_user, company_id):
        raise HTTPException(status_code=403, detail="No access to this company")

    if module_name not in VALID_MODULES:
        raise HTTPException(
            status_code=400, detail=f"Invalid module: {module_name}. Valid: {VALID_MODULES}"
        )

    setting = (
        db.query(CompanyDetectionSettings)
        .filter(
            CompanyDetectionSettings.company_id == company_id,
            CompanyDetectionSettings.module_name == module_name,
        )
        .first()
    )
    if not setting:
        raise HTTPException(
            status_code=404,
            detail=f"No setting found for module '{module_name}'. Create it first with POST.",
        )

    update_data = data.dict(exclude_unset=True)
    for key, value in update_data.items():
        setattr(setting, key, value)
    bits = ", ".join(sorted(update_data.keys())) or "settings"
    record_company_module_change(
        db,
        company_id=company_id,
        module_name=module_name,
        message=f"{current_user.username} updated '{module_name}' ({bits}).",
        actor_username=current_user.username,
        actor_user_id=current_user.id,
    )
    db.commit()
    db.refresh(setting)
    return setting


@router.delete("/company/{company_id}/{module_name}", status_code=204)
async def delete_company_setting(
    company_id: int,
    module_name: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """Delete a company module setting (resets to global defaults)."""
    if not check_company_access(current_user, company_id):
        raise HTTPException(status_code=403, detail="No access to this company")

    setting = (
        db.query(CompanyDetectionSettings)
        .filter(
            CompanyDetectionSettings.company_id == company_id,
            CompanyDetectionSettings.module_name == module_name,
        )
        .first()
    )
    if not setting:
        raise HTTPException(status_code=404, detail="Setting not found")

    db.delete(setting)
    record_company_module_change(
        db,
        company_id=company_id,
        module_name=module_name,
        message=(
            f"{current_user.username} removed custom '{module_name}' configuration; "
            "global defaults apply."
        ),
        actor_username=current_user.username,
        actor_user_id=current_user.id,
    )
    db.commit()
