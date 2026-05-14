"""
Zone-to-officer assignment endpoints.

Admins can assign a security officer to a camera zone.
When a detection fires in that zone, the officer (and all company admins)
receive alert email notifications.
"""

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.core.security import require_security_officer
from app.database import get_db
from app.models.audit_log import create_audit_log
from app.models.user import Role, User
from app.models.zone_officer_assignment import ZoneOfficerAssignment
from app.schemas.zone_officer_assignment import (
    ZoneOfficerAssignmentCreate,
    ZoneOfficerAssignmentResponse,
    ZoneOfficerAssignmentUpdate,
)

router = APIRouter(prefix="/zone-assignments", tags=["zone-assignments"])


def _require_admin(current_user: User) -> None:
    if current_user.role not in (Role.ADMIN, Role.AEGIS_ADMIN):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admins can manage zone assignments",
        )


def _validate_officer(db: Session, officer_id: int, company_id: int) -> User:
    """Ensure the officer exists, belongs to the company, and has the right role."""
    officer = db.query(User).filter(User.id == officer_id).first()
    if not officer:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Officer not found")
    if officer.role != Role.SECURITY_OFFICER:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User must have the SECURITY_OFFICER role",
        )
    if officer.company_id != company_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Officer does not belong to your company",
        )
    return officer


@router.post("", response_model=ZoneOfficerAssignmentResponse, status_code=status.HTTP_201_CREATED)
async def create_zone_assignment(
    request: Request,
    data: ZoneOfficerAssignmentCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_security_officer),
):
    """Assign a security officer to a zone. One officer per zone per company."""
    _require_admin(current_user)

    company_id = current_user.company_id
    if not company_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Admin must belong to a company",
        )

    _validate_officer(db, data.officer_id, company_id)

    existing = (
        db.query(ZoneOfficerAssignment)
        .filter(
            ZoneOfficerAssignment.company_id == company_id,
            ZoneOfficerAssignment.zone_name == data.zone_name,
        )
        .first()
    )
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Zone '{data.zone_name}' already has an officer assigned. Use PUT to update.",
        )

    assignment = ZoneOfficerAssignment(
        company_id=company_id,
        zone_name=data.zone_name,
        officer_id=data.officer_id,
    )
    db.add(assignment)
    db.flush()

    db.add(
        create_audit_log(
            request,
            current_user.id,
            "create_zone_assignment",
            "zone_officer_assignment",
            assignment.id,
            {"zone_name": data.zone_name, "officer_id": data.officer_id},
        )
    )
    db.commit()
    db.refresh(assignment)
    return assignment


@router.get("", response_model=list[ZoneOfficerAssignmentResponse])
async def list_zone_assignments(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_security_officer),
):
    """List all zone assignments for the current user's company."""
    company_id = current_user.company_id
    if not company_id:
        return []
    return (
        db.query(ZoneOfficerAssignment)
        .filter(ZoneOfficerAssignment.company_id == company_id)
        .order_by(ZoneOfficerAssignment.zone_name)
        .all()
    )


@router.get("/{assignment_id}", response_model=ZoneOfficerAssignmentResponse)
async def get_zone_assignment(
    assignment_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_security_officer),
):
    """Get a specific zone assignment."""
    company_id = current_user.company_id
    assignment = (
        db.query(ZoneOfficerAssignment)
        .filter(
            ZoneOfficerAssignment.id == assignment_id,
            ZoneOfficerAssignment.company_id == company_id,
        )
        .first()
    )
    if not assignment:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Assignment not found")
    return assignment


@router.put("/{assignment_id}", response_model=ZoneOfficerAssignmentResponse)
async def update_zone_assignment(
    request: Request,
    assignment_id: int,
    data: ZoneOfficerAssignmentUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_security_officer),
):
    """Re-assign a different security officer to a zone."""
    _require_admin(current_user)

    company_id = current_user.company_id
    assignment = (
        db.query(ZoneOfficerAssignment)
        .filter(
            ZoneOfficerAssignment.id == assignment_id,
            ZoneOfficerAssignment.company_id == company_id,
        )
        .first()
    )
    if not assignment:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Assignment not found")

    _validate_officer(db, data.officer_id, company_id)

    assignment.officer_id = data.officer_id
    db.add(
        create_audit_log(
            request,
            current_user.id,
            "update_zone_assignment",
            "zone_officer_assignment",
            assignment.id,
            {"zone_name": assignment.zone_name, "new_officer_id": data.officer_id},
        )
    )
    db.commit()
    db.refresh(assignment)
    return assignment


@router.delete("/{assignment_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_zone_assignment(
    request: Request,
    assignment_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_security_officer),
):
    """Remove a zone-officer assignment."""
    _require_admin(current_user)

    company_id = current_user.company_id
    assignment = (
        db.query(ZoneOfficerAssignment)
        .filter(
            ZoneOfficerAssignment.id == assignment_id,
            ZoneOfficerAssignment.company_id == company_id,
        )
        .first()
    )
    if not assignment:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Assignment not found")

    db.add(
        create_audit_log(
            request,
            current_user.id,
            "delete_zone_assignment",
            "zone_officer_assignment",
            assignment.id,
            {"zone_name": assignment.zone_name},
        )
    )
    db.delete(assignment)
    db.commit()
