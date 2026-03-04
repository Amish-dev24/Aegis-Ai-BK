"""
Camera management endpoints.
"""
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.database import get_db
from app.core.security import require_security_officer, get_current_user, get_user_company_filter, check_company_access
from app.models.user import Role
from app.schemas.camera import CameraCreate, CameraResponse, CameraUpdate
from app.models.camera import Camera
from app.models.user import User
from app.models.audit_log import AuditLog

router = APIRouter(prefix="/cameras", tags=["cameras"])


@router.post("", response_model=CameraResponse, status_code=status.HTTP_201_CREATED)
async def create_camera(
    camera_data: CameraCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_security_officer)
):
    """Create a new camera. Company users can only create cameras for their company."""
    camera_dict = camera_data.dict()
    
    # Set company_id based on user role
    if current_user.role == Role.AEGIS_ADMIN:
        # Aegis AI admins can specify company_id, but it should be in the request
        if "company_id" not in camera_dict or camera_dict["company_id"] is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="company_id is required"
            )
    else:
        # Company users automatically get their company_id
        camera_dict["company_id"] = current_user.company_id
    
    db_camera = Camera(**camera_dict)
    db.add(db_camera)
    db.commit()
    db.refresh(db_camera)
    
    # Log creation
    audit_log = AuditLog(
        user_id=current_user.id,
        action="create_camera",
        resource_type="camera",
        resource_id=db_camera.id
    )
    db.add(audit_log)
    db.commit()
    
    return db_camera


@router.get("", response_model=List[CameraResponse])
async def list_cameras(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    active_only: bool = False
):
    """List cameras. Company users see only their company's cameras."""
    query = db.query(Camera)
    
    # Filter by company (unless Aegis AI admin)
    company_filter = get_user_company_filter(current_user)
    if company_filter is not None:
        query = query.filter(Camera.company_id == company_filter)
    
    if active_only:
        query = query.filter(Camera.is_active == True)
    
    cameras = query.all()
    return cameras


@router.get("/{camera_id}", response_model=CameraResponse)
async def get_camera(
    camera_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get camera by ID. Company users can only access their company's cameras."""
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Camera not found"
        )
    
    # Check company access
    if camera.company_id and not check_company_access(current_user, camera.company_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to access this camera"
        )
    
    return camera


@router.put("/{camera_id}", response_model=CameraResponse)
async def update_camera(
    camera_id: int,
    camera_update: CameraUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_security_officer)
):
    """Update camera. Company users can only update their company's cameras."""
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Camera not found"
        )
    
    # Check company access
    if camera.company_id and not check_company_access(current_user, camera.company_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to update this camera"
        )
    
    update_data = camera_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(camera, field, value)
    
    db.commit()
    db.refresh(camera)
    
    # Log update
    audit_log = AuditLog(
        user_id=current_user.id,
        action="update_camera",
        resource_type="camera",
        resource_id=camera_id
    )
    db.add(audit_log)
    db.commit()
    
    return camera


@router.delete("/{camera_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_camera(
    camera_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_security_officer)
):
    """Delete camera. Company users can only delete their company's cameras."""
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Camera not found"
        )
    
    # Check company access
    if camera.company_id and not check_company_access(current_user, camera.company_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to delete this camera"
        )
    
    # Log deletion
    audit_log = AuditLog(
        user_id=current_user.id,
        action="delete_camera",
        resource_type="camera",
        resource_id=camera_id
    )
    db.add(audit_log)
    db.delete(camera)
    db.commit()
    
    return None

