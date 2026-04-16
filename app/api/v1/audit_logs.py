"""
Audit log management endpoints.
"""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from app.database import get_db
from app.core.security import get_current_user
from app.models.audit_log import AuditLog
from app.models.user import User, Role
from app.schemas.audit_log import AuditLogResponse

router = APIRouter(prefix="/audit-logs", tags=["audit-logs"])


@router.get("", response_model=List[AuditLogResponse])
async def get_audit_logs(
    skip: int = 0,
    limit: int = 100,
    user_id: Optional[int] = None,
    action: Optional[str] = None,
    resource_type: Optional[str] = None,
    company_id: Optional[int] = Query(None, description="Filter by company (aegis_admin only)"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get audit logs.
    - Aegis Admin: Can see all logs, optionally filtered by company_id.
    - Company Admin/User: Can only see logs for users in their company.
    """
    query = db.query(AuditLog)
    _user_joined = False

    # RBAC Filter
    if current_user.role != Role.AEGIS_ADMIN:
        if not current_user.company_id:
            query = query.filter(AuditLog.user_id == current_user.id)
        else:
            query = query.join(User, AuditLog.user_id == User.id).filter(User.company_id == current_user.company_id)
            _user_joined = True
    elif company_id:
        # Aegis admin filtering by a specific company
        query = query.join(User, AuditLog.user_id == User.id).filter(User.company_id == company_id)
        _user_joined = True

    # Apply filters
    if user_id:
        query = query.filter(AuditLog.user_id == user_id)
    if action:
        query = query.filter(AuditLog.action.ilike(f"%{action}%"))
    if resource_type:
        query = query.filter(AuditLog.resource_type == resource_type)

    query = query.order_by(AuditLog.created_at.desc())

    return query.offset(skip).limit(limit).all()


@router.get("/{log_id}", response_model=AuditLogResponse)
async def get_audit_log(
    log_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get a specific audit log entry."""
    query = db.query(AuditLog).filter(AuditLog.id == log_id)
    
    if current_user.role != Role.AEGIS_ADMIN:
        if not current_user.company_id:
             query = query.filter(AuditLog.user_id == current_user.id)
        else:
            query = query.join(User).filter(User.company_id == current_user.company_id)
            
    log = query.first()
    if not log:
        raise HTTPException(status_code=404, detail="Audit log not found")
    return log
