"""
User management endpoints.
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy.orm import Session

from app.core.security import (
    check_directory_company_access,
    get_directory_company_filter,
    get_password_hash,
    require_admin,
)
from app.database import get_db
from app.models.audit_log import create_audit_log
from app.models.user import Role, User
from app.schemas.user import UserCreate, UserResponse, UserUpdate

router = APIRouter(prefix="/users", tags=["users"])


@router.get("", response_model=list[UserResponse])
async def list_users(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
    company_id: Optional[int] = Query(
        None, description="Filter by company ID (aegis_admin only; omit for all companies)"
    ),
    limit: int = Query(default=200, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    """List users (admin only).

    - **Company admin**: always scoped to their company (optional ``company_id`` must match).
    - **Aegis admin**: pass ``company_id`` to list one tenant; omit it for a platform-wide list
      (e.g. global verification, dashboard pending counts).
    """
    if current_user.role == Role.AEGIS_ADMIN:
        if company_id is not None:
            query = db.query(User).filter(User.company_id == company_id)
        else:
            query = db.query(User)
    else:
        company_filter = get_directory_company_filter(current_user, company_id)
        if company_filter is None:
            return []

        query = db.query(User).filter(User.company_id == company_filter)

    users = query.order_by(User.id).offset(offset).limit(limit).all()
    return users


@router.get("/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: int, db: Session = Depends(get_db), current_user: User = Depends(require_admin)
):
    """Get user by ID (admin only). Company admins can only access their company's users."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    # Check company access
    if user.company_id and not check_directory_company_access(current_user, user.company_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to access this user",
        )

    return user


@router.put("/{user_id}", response_model=UserResponse)
async def update_user(
    request: Request,
    user_id: int,
    user_update: UserUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """Update user (admin only). Company admins can only update their company's users."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    # Check company access
    if user.company_id and not check_directory_company_access(current_user, user.company_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to update this user",
        )

    # Company admins cannot change roles to AEGIS_ADMIN
    if current_user.role != Role.AEGIS_ADMIN and user_update.role == Role.AEGIS_ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Cannot assign AEGIS_ADMIN role"
        )

    # Update fields (skip None values to avoid overwriting with null)
    update_data = {k: v for k, v in user_update.dict(exclude_unset=True).items() if v is not None}
    if "password" in update_data:
        update_data["hashed_password"] = get_password_hash(update_data.pop("password"))

    for field, value in update_data.items():
        setattr(user, field, value)

    db.commit()
    db.refresh(user)

    # Log update
    db.add(
        create_audit_log(
            request,
            current_user.id,
            "update_user",
            "user",
            user_id,
            {"updated_fields": list(update_data.keys()), "target_user": user.username},
        )
    )
    db.commit()

    return user


@router.post("", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(
    request: Request,
    user_data: UserCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """Create a new user (admin only). Company admins can only create users in their company."""
    # Company admins cannot create Aegis AI admins
    if current_user.role != Role.AEGIS_ADMIN and user_data.role == Role.AEGIS_ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Cannot create AEGIS_ADMIN user"
        )

    # Check if user exists
    if db.query(User).filter(User.username == user_data.username).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Username already registered"
        )
    if db.query(User).filter(User.email == user_data.email).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered"
        )

    # Determine company_id
    if current_user.role == Role.AEGIS_ADMIN:
        # Aegis AI admins can create users for any company or without company
        company_id = user_data.company_id
    else:
        # Company admins create users in their own company
        company_id = current_user.company_id

    # Create user
    hashed_password = get_password_hash(user_data.password)
    db_user = User(
        username=user_data.username,
        email=user_data.email,
        hashed_password=hashed_password,
        full_name=user_data.full_name,
        phone_number=user_data.phone_number,
        role=user_data.role,
        company_id=company_id,
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)

    # Log user creation
    db.add(
        create_audit_log(
            request,
            current_user.id,
            "create_user",
            "user",
            db_user.id,
            {
                "username": db_user.username,
                "role": db_user.role.value,
                "company_id": db_user.company_id,
            },
        )
    )
    db.commit()

    return db_user


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(
    request: Request,
    user_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """Delete user (admin only). Company admins can only delete their company's users."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    # Check company access
    if user.company_id and not check_directory_company_access(current_user, user.company_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to delete this user",
        )

    # Log deletion
    db.add(
        create_audit_log(
            request,
            current_user.id,
            "delete_user",
            "user",
            user_id,
            {"deleted_user": user.username, "email": user.email},
        )
    )
    db.delete(user)
    db.commit()

    return None
