"""
Company management endpoints for Aegis AI admins.
"""
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.database import get_db
from app.core.security import require_aegis_admin, require_admin, get_current_user, check_company_access
from app.schemas.company import (
    CompanyCreate, 
    CompanyResponse, 
    CompanyUpdate, 
    CompanyWithUsers,
    CompanyStats
)
from app.models.company import Company
from app.models.user import User, Role
from app.models.camera import Camera
from app.models.detection import Detection
from app.models.alert import Alert
from app.models.audit_log import create_audit_log

router = APIRouter(prefix="/companies", tags=["companies"])


@router.get("", response_model=List[CompanyResponse])
async def list_companies(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_aegis_admin),
    verified_only: bool = False
):
    """List all companies (Aegis AI admin only)."""
    query = db.query(Company)
    if verified_only:
        query = query.filter(Company.is_verified == True)
    companies = query.all()
    return companies


@router.get("/stats", response_model=List[CompanyStats])
async def get_companies_stats(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_aegis_admin)
):
    """Get statistics for all companies (Aegis AI admin only)."""
    companies = db.query(Company).all()
    stats = []
    
    for company in companies:
        # Count users
        total_users = db.query(func.count(User.id)).filter(User.company_id == company.id).scalar()
        active_users = db.query(func.count(User.id)).filter(
            User.company_id == company.id,
            User.is_active == True
        ).scalar()
        
        # Count cameras
        total_cameras = db.query(func.count(Camera.id)).filter(Camera.company_id == company.id).scalar()
        active_cameras = db.query(func.count(Camera.id)).filter(
            Camera.company_id == company.id,
            Camera.is_active == True
        ).scalar()
        
        # Count detections
        total_detections = db.query(func.count(Detection.id)).filter(
            Detection.company_id == company.id
        ).scalar()
        
        # Count alerts
        total_alerts = db.query(func.count(Alert.id)).filter(
            Alert.company_id == company.id
        ).scalar()
        
        # Get last activity (most recent user login or detection)
        last_user_login = db.query(func.max(User.last_login)).filter(
            User.company_id == company.id
        ).scalar()
        last_detection = db.query(func.max(Detection.detected_at)).filter(
            Detection.company_id == company.id
        ).scalar()
        
        last_activity = None
        if last_user_login and last_detection:
            last_activity = max(last_user_login, last_detection)
        elif last_user_login:
            last_activity = last_user_login
        elif last_detection:
            last_activity = last_detection
        
        stats.append(CompanyStats(
            company_id=company.id,
            company_name=company.name,
            total_users=total_users or 0,
            active_users=active_users or 0,
            total_cameras=total_cameras or 0,
            active_cameras=active_cameras or 0,
            total_detections=total_detections or 0,
            total_alerts=total_alerts or 0,
            last_activity=last_activity
        ))
    
    return stats


@router.get("/{company_id}", response_model=CompanyWithUsers)
async def get_company(
    company_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin)
):
    """Get company details with user count. Admins can only see their own company."""
    if not check_company_access(current_user, company_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to access this company"
        )

    company = db.query(Company).filter(Company.id == company_id).first()
    if not company:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Company not found"
        )

    # Count users
    user_count = db.query(func.count(User.id)).filter(User.company_id == company.id).scalar()
    active_user_count = db.query(func.count(User.id)).filter(
        User.company_id == company.id,
        User.is_active == True
    ).scalar()
    
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
        active_user_count=active_user_count or 0
    )
    
    return company_data


@router.get("/{company_id}/users", response_model=List[dict])
async def get_company_users(
    company_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin)
):
    """Get all users for a company. Admins can only see their own company's users."""
    if not check_company_access(current_user, company_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to access this company"
        )

    company = db.query(Company).filter(Company.id == company_id).first()
    if not company:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Company not found"
        )
    
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
            "created_at": user.created_at
        }
        for user in users
    ]


@router.post("", response_model=CompanyResponse, status_code=status.HTTP_201_CREATED)
async def create_company(
    request: Request,
    company_data: CompanyCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin)
):
    """Create a new company. Admin creates unverified, aegis_admin creates verified."""
    # Check if company name already exists
    if db.query(Company).filter(Company.name == company_data.name).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Company name already exists"
        )
    
    # Check if domain already exists (if provided)
    if company_data.domain and db.query(Company).filter(Company.domain == company_data.domain).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Company domain already exists"
        )
    
    db_company = Company(**company_data.dict())
    # Aegis admin creates verified companies, regular admin creates unverified
    if current_user.role.value != 'aegis_admin':
        db_company.is_verified = False
    db.add(db_company)
    db.commit()
    db.refresh(db_company)

    # Assign the creating admin to this company (if they don't have one)
    if not current_user.company_id and current_user.role.value == 'admin':
        current_user.company_id = db_company.id
        db.commit()

    # Log creation
    db.add(create_audit_log(
        request, current_user.id, "create_company", "company", db_company.id,
        {"name": db_company.name, "domain": db_company.domain}
    ))
    db.commit()

    return db_company


@router.put("/{company_id}", response_model=CompanyResponse)
async def update_company(
    request: Request,
    company_id: int,
    company_update: CompanyUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_aegis_admin)
):
    """Update company (Aegis AI admin only)."""
    company = db.query(Company).filter(Company.id == company_id).first()
    if not company:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Company not found"
        )
    
    # Update fields
    update_data = company_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(company, field, value)
    
    db.commit()
    db.refresh(company)
    
    # Log update
    db.add(create_audit_log(
        request, current_user.id, "update_company", "company", company_id,
        {"updated_fields": list(update_data.keys()), "company_name": company.name}
    ))
    db.commit()

    return company


@router.post("/{company_id}/verify", response_model=CompanyResponse)
async def verify_company(
    request: Request,
    company_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_aegis_admin)
):
    """Verify a company (Aegis AI admin only)."""
    company = db.query(Company).filter(Company.id == company_id).first()
    if not company:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Company not found"
        )
    
    company.is_verified = True
    db.commit()
    db.refresh(company)
    
    # Log verification
    db.add(create_audit_log(
        request, current_user.id, "verify_company", "company", company_id,
        {"company_name": company.name}
    ))
    db.commit()

    return company

