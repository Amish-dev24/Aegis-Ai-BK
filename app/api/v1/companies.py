"""
Company management endpoints for Aegis AI admins.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.security import (
    check_company_access,
    get_current_user,
    require_admin,
    require_aegis_admin,
)
from app.database import get_db
from app.models.alert import Alert
from app.models.audit_log import create_audit_log
from app.models.camera import Camera
from app.models.company import Company
from app.models.detection import Detection
from app.models.user import Role, User
from app.schemas.company import (
    CompanyCreate,
    CompanyResponse,
    CompanyStats,
    CompanyUpdate,
    CompanyWithUsers,
)

router = APIRouter(prefix="/companies", tags=["companies"])


@router.get("", response_model=list[CompanyResponse])
async def list_companies(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_aegis_admin),
    verified_only: bool = False,
    limit: int = Query(default=200, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    """List all companies (Aegis AI admin only)."""
    query = db.query(Company)
    if verified_only:
        query = query.filter(Company.is_verified is True)
    companies = query.order_by(Company.id).offset(offset).limit(limit).all()
    return companies


@router.get("/stats", response_model=list[CompanyStats])
async def get_companies_stats(
    db: Session = Depends(get_db), current_user: User = Depends(require_aegis_admin)
):
    """Get statistics for all companies (Aegis AI admin only).

    Uses 5 single GROUP BY queries instead of 7 queries per company (N+1 fix).
    """
    companies = db.query(Company.id, Company.name).all()
    company_ids = [c.id for c in companies]

    if not company_ids:
        return []

    # --- aggregate all companies in one pass each ---
    user_totals = {
        row.company_id: row.total
        for row in db.query(User.company_id, func.count(User.id).label("total"))
        .filter(User.company_id.in_(company_ids))
        .group_by(User.company_id)
        .all()
    }
    user_active = {
        row.company_id: row.total
        for row in db.query(User.company_id, func.count(User.id).label("total"))
        .filter(User.company_id.in_(company_ids), User.is_active is True)
        .group_by(User.company_id)
        .all()
    }
    camera_totals = {
        row.company_id: row.total
        for row in db.query(Camera.company_id, func.count(Camera.id).label("total"))
        .filter(Camera.company_id.in_(company_ids))
        .group_by(Camera.company_id)
        .all()
    }
    camera_active = {
        row.company_id: row.total
        for row in db.query(Camera.company_id, func.count(Camera.id).label("total"))
        .filter(Camera.company_id.in_(company_ids), Camera.is_active is True)
        .group_by(Camera.company_id)
        .all()
    }
    detection_totals = {
        row.company_id: row.total
        for row in db.query(Detection.company_id, func.count(Detection.id).label("total"))
        .filter(Detection.company_id.in_(company_ids))
        .group_by(Detection.company_id)
        .all()
    }
    alert_totals = {
        row.company_id: row.total
        for row in db.query(Alert.company_id, func.count(Alert.id).label("total"))
        .filter(Alert.company_id.in_(company_ids))
        .group_by(Alert.company_id)
        .all()
    }
    last_logins = {
        row.company_id: row.last_login
        for row in db.query(User.company_id, func.max(User.last_login).label("last_login"))
        .filter(User.company_id.in_(company_ids))
        .group_by(User.company_id)
        .all()
    }
    last_detections = {
        row.company_id: row.last_det
        for row in db.query(Detection.company_id, func.max(Detection.detected_at).label("last_det"))
        .filter(Detection.company_id.in_(company_ids))
        .group_by(Detection.company_id)
        .all()
    }

    stats = []
    for company in companies:
        cid = company.id
        last_login = last_logins.get(cid)
        last_det = last_detections.get(cid)
        if last_login and last_det:
            last_activity = max(last_login, last_det)
        else:
            last_activity = last_login or last_det

        stats.append(
            CompanyStats(
                company_id=cid,
                company_name=company.name,
                total_users=user_totals.get(cid, 0),
                active_users=user_active.get(cid, 0),
                total_cameras=camera_totals.get(cid, 0),
                active_cameras=camera_active.get(cid, 0),
                total_detections=detection_totals.get(cid, 0),
                total_alerts=alert_totals.get(cid, 0),
                last_activity=last_activity,
            )
        )

    return stats


@router.get("/{company_id}", response_model=CompanyWithUsers)
async def get_company(
    company_id: int, db: Session = Depends(get_db), current_user: User = Depends(require_admin)
):
    """Get company details with user count. Admins can only see their own company."""
    if not check_company_access(current_user, company_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to access this company",
        )

    company = db.query(Company).filter(Company.id == company_id).first()
    if not company:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Company not found")

    # Count users
    user_count = db.query(func.count(User.id)).filter(User.company_id == company.id).scalar()
    active_user_count = (
        db.query(func.count(User.id))
        .filter(User.company_id == company.id, User.is_active is True)
        .scalar()
    )

    company_data = CompanyWithUsers(
        id=company.id,
        name=company.name,
        domain=company.domain,
        contact_email=company.contact_email,
        contact_phone=company.contact_phone,
        is_active=company.is_active,
        is_verified=company.is_verified,
        created_at=company.created_at,
        updated_at=company.updated_at,
        user_count=user_count or 0,
        active_user_count=active_user_count or 0,
    )

    return company_data


@router.get("/{company_id}/users", response_model=list[dict])
async def get_company_users(
    company_id: int, db: Session = Depends(get_db), current_user: User = Depends(require_admin)
):
    """Get all users for a company. Admins can only see their own company's users."""
    if not check_company_access(current_user, company_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to access this company",
        )

    company = db.query(Company).filter(Company.id == company_id).first()
    if not company:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Company not found")

    users = db.query(User).filter(User.company_id == company_id).all()

    return [
        {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "full_name": user.full_name,
            "role": user.role.value,
            "is_active": user.is_active,
            "last_login": user.last_login,
            "created_at": user.created_at,
        }
        for user in users
    ]


@router.post("", response_model=CompanyResponse, status_code=status.HTTP_201_CREATED)
async def create_company(
    request: Request,
    company_data: CompanyCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """Create a new company. Admin creates unverified, aegis_admin creates verified."""
    # Check if company name already exists
    if db.query(Company).filter(Company.name == company_data.name).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Company name already exists"
        )

    # Check if domain already exists (if provided)
    if (
        company_data.domain
        and db.query(Company).filter(Company.domain == company_data.domain).first()
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Company domain already exists"
        )

    db_company = Company(**company_data.dict())
    # Aegis admin creates verified companies, regular admin creates unverified
    if current_user.role.value != "aegis_admin":
        db_company.is_verified = False
    db.add(db_company)
    db.commit()
    db.refresh(db_company)

    # Assign the creating admin to this company (if they don't have one)
    if not current_user.company_id and current_user.role.value == "admin":
        current_user.company_id = db_company.id
        db.commit()

    # Log creation
    db.add(
        create_audit_log(
            request,
            current_user.id,
            "create_company",
            "company",
            db_company.id,
            {"name": db_company.name, "domain": db_company.domain},
        )
    )
    db.commit()

    return db_company


@router.put("/{company_id}", response_model=CompanyResponse)
async def update_company(
    request: Request,
    company_id: int,
    company_update: CompanyUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Update company. Aegis admin can update any company; company admin can update their own."""
    if current_user.role != Role.AEGIS_ADMIN:
        if not check_company_access(current_user, company_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions"
            )
    company = db.query(Company).filter(Company.id == company_id).first()
    if not company:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Company not found")

    # Update fields
    update_data = company_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(company, field, value)

    db.commit()
    db.refresh(company)

    # Log update
    db.add(
        create_audit_log(
            request,
            current_user.id,
            "update_company",
            "company",
            company_id,
            {"updated_fields": list(update_data.keys()), "company_name": company.name},
        )
    )
    db.commit()

    return company


@router.post("/{company_id}/verify", response_model=CompanyResponse)
async def verify_company(
    request: Request,
    company_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_aegis_admin),
):
    """Verify a company (Aegis AI admin only)."""
    company = db.query(Company).filter(Company.id == company_id).first()
    if not company:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Company not found")

    company.is_verified = True
    db.commit()
    db.refresh(company)

    # Log verification
    db.add(
        create_audit_log(
            request,
            current_user.id,
            "verify_company",
            "company",
            company_id,
            {"company_name": company.name},
        )
    )
    db.commit()

    return company
