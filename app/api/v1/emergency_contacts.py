"""
Emergency contact directory endpoints.
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.security import (
    check_directory_company_access,
    get_directory_company_filter,
    require_admin,
    require_any_authenticated,
)
from app.database import get_db
from app.models.company import Company
from app.models.emergency_contact import EmergencyContact
from app.models.user import Role, User

router = APIRouter(prefix="/emergency-contacts", tags=["emergency-contacts"])


class EmergencyContactCreate(BaseModel):
    name: str
    phone_number: str
    role: Optional[str] = None
    is_primary: bool = False
    company_id: Optional[int] = None


class EmergencyContactUpdate(BaseModel):
    name: Optional[str] = None
    phone_number: Optional[str] = None
    role: Optional[str] = None
    is_primary: Optional[bool] = None


class EmergencyContactResponse(BaseModel):
    id: int
    company_id: int
    name: str
    phone_number: str
    role: Optional[str] = None
    is_primary: bool

    class Config:
        from_attributes = True


@router.get("", response_model=list[EmergencyContactResponse])
async def list_contacts(
    company_id: Optional[int] = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_any_authenticated),
):
    """List emergency contacts for the authenticated user's company."""
    company_filter = get_directory_company_filter(current_user, company_id)
    if company_filter is None:
        return []

    query = db.query(EmergencyContact).filter(EmergencyContact.company_id == company_filter)
    return query.order_by(EmergencyContact.is_primary.desc(), EmergencyContact.name).all()


@router.post("", response_model=EmergencyContactResponse, status_code=201)
async def create_contact(
    data: EmergencyContactCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """Add an emergency contact (admin only)."""
    if current_user.role == Role.AEGIS_ADMIN:
        target_company_id = data.company_id or current_user.company_id
        if target_company_id is None:
            raise HTTPException(status_code=400, detail="company_id is required")
        if not db.query(Company).filter(Company.id == target_company_id).first():
            raise HTTPException(status_code=400, detail="Invalid company_id")
    else:
        if not current_user.company_id:
            raise HTTPException(status_code=400, detail="No company assigned")
        target_company_id = current_user.company_id

    contact = EmergencyContact(
        company_id=target_company_id,
        name=data.name,
        phone_number=data.phone_number,
        role=data.role,
        is_primary=data.is_primary,
    )
    db.add(contact)
    db.commit()
    db.refresh(contact)
    return contact


@router.put("/{contact_id}", response_model=EmergencyContactResponse)
async def update_contact(
    contact_id: int,
    data: EmergencyContactUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """Update an emergency contact."""
    contact = db.query(EmergencyContact).filter(EmergencyContact.id == contact_id).first()
    if not contact:
        raise HTTPException(status_code=404, detail="Contact not found")
    if not check_directory_company_access(current_user, contact.company_id):
        raise HTTPException(status_code=403, detail="No access")

    for key, value in data.dict(exclude_unset=True).items():
        setattr(contact, key, value)
    db.commit()
    db.refresh(contact)
    return contact


@router.delete("/{contact_id}", status_code=204)
async def delete_contact(
    contact_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """Delete an emergency contact."""
    contact = db.query(EmergencyContact).filter(EmergencyContact.id == contact_id).first()
    if not contact:
        raise HTTPException(status_code=404, detail="Contact not found")
    if not check_directory_company_access(current_user, contact.company_id):
        raise HTTPException(status_code=403, detail="No access")

    db.delete(contact)
    db.commit()
