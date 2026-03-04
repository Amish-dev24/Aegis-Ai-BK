"""
User management endpoints.
"""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from app.database import get_db
from app.core.security import require_admin, get_current_user, get_user_company_filter, check_company_access
from app.schemas.user import UserResponse, UserUpdate, UserCreate
from app.models.user import User, Role
from app.models.company import Company
from app.models.audit_log import AuditLog
from app.core.security import get_password_hash

router = APIRouter(prefix="/users", tags=["users"])


@router.get("", response_model=List[UserResponse])
async def list_users(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
    company_id: Optional[int] = Query(None, description="Filter by company ID (Aegis AI admin only)")
):
    """List users (admin only). Company admins see only their company's users."""
    query = db.query(User)
    
    # Aegis AI admins can filter by company or see all
    if current_user.role == Role.AEGIS_ADMIN:
        if company_id:
            query = query.filter(User.company_id == company_id)
    else:
        # Company admins can only see their own company's users
        query = query.filter(User.company_id == current_user.company_id)
    
    users = query.all()
    return users


@router.get("/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin)
):
    """Get user by ID (admin only). Company admins can only access their company's users."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Check company access
    if user.company_id and not check_company_access(current_user, user.company_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to access this user"
        )
    
    return user


@router.put("/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: int,
    user_update: UserUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin)
):
    """Update user (admin only). Company admins can only update their company's users."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Check company access
    if user.company_id and not check_company_access(current_user, user.company_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to update this user"
        )
    
    # Company admins cannot change roles to AEGIS_ADMIN
    if current_user.role != Role.AEGIS_ADMIN and user_update.role == Role.AEGIS_ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot assign AEGIS_ADMIN role"
        )
    
    # Update fields
    update_data = user_update.dict(exclude_unset=True)
    if "password" in update_data:
        update_data["hashed_password"] = get_password_hash(update_data.pop("password"))
    
    for field, value in update_data.items():
        setattr(user, field, value)
    
    db.commit()
    db.refresh(user)
    
    # Log update
    audit_log = AuditLog(
        user_id=current_user.id,
        action="update_user",
        resource_type="user",
        resource_id=user_id
    )
    db.add(audit_log)
    db.commit()
    
    return user


@router.post("", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(
    user_data: UserCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin)
):
    """Create a new user (admin only). Company admins can only create users in their company."""
    # Company admins cannot create Aegis AI admins
    if current_user.role != Role.AEGIS_ADMIN and user_data.role == Role.AEGIS_ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot create AEGIS_ADMIN user"
        )
    
    # Check if user exists
    if db.query(User).filter(User.username == user_data.username).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    if db.query(User).filter(User.email == user_data.email).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
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
        role=user_data.role,
        company_id=company_id
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    # Log user creation
    audit_log = AuditLog(
        user_id=current_user.id,
        action="create_user",
        resource_type="user",
        resource_id=db_user.id
    )
    db.add(audit_log)
    db.commit()
    
    return db_user


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin)
):
    """Delete user (admin only). Company admins can only delete their company's users."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Check company access
    if user.company_id and not check_company_access(current_user, user.company_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to delete this user"
        )
    
    # Log deletion
    audit_log = AuditLog(
        user_id=current_user.id,
        action="delete_user",
        resource_type="user",
        resource_id=user_id
    )
    db.add(audit_log)
    db.delete(user)
    db.commit()
    
    return None

