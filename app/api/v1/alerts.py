"""
Alert management endpoints.
"""
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import and_
from app.database import get_db
from app.core.security import require_any_authenticated, require_security_officer, get_current_user, get_user_company_filter, check_company_access
from app.models.user import Role
from app.schemas.alert import AlertCreate, AlertResponse, AlertUpdate
from app.models.alert import Alert, AlertStatus
from app.models.detection import Detection
from app.models.user import User
from app.services.email_service import email_service
from app.models.evidence import Evidence
from app.models.audit_log import AuditLog
from datetime import datetime

router = APIRouter(prefix="/alerts", tags=["alerts"])


@router.post("", response_model=AlertResponse, status_code=status.HTTP_201_CREATED)
async def create_alert(
    alert_data: AlertCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_security_officer)
):
    """Create an alert. Company users can only create alerts for their company's detections."""
    # Verify detection exists
    detection = db.query(Detection).filter(Detection.id == alert_data.detection_id).first()
    if not detection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Detection not found"
        )
    
    # Check company access
    if detection.company_id and not check_company_access(current_user, detection.company_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to create alert for this detection"
        )
    
    alert_dict = alert_data.dict()
    alert_dict["company_id"] = detection.company_id
    
    db_alert = Alert(**alert_dict)
    db.add(db_alert)
    db.commit()
    db.refresh(db_alert)
    
    # Send email if high/critical threat
    if detection.threat_level.value in ["high", "critical"]:
        # Get evidence snapshot if available
        evidence = db.query(Evidence).filter(Evidence.detection_id == detection.id).first()
        snapshot_path = evidence.image_path if evidence else None
        
        # Send email (in background task ideally)
        await email_service.send_alert_email(
            to_emails=[current_user.email],  # In production, get from config
            subject=db_alert.title,
            message=db_alert.message or f"Alert for detection {detection.id}",
            snapshot_path=snapshot_path,
            metadata={
                "Detection Type": detection.detection_type.value,
                "Threat Level": detection.threat_level.value,
                "Confidence": f"{detection.confidence:.2%}",
                "Timestamp": detection.frame_timestamp.isoformat()
            }
        )
        
        db_alert.email_sent = True
        db_alert.email_sent_at = datetime.utcnow()
        db.commit()

    # Audit log
    db.add(AuditLog(
        user_id=current_user.id,
        action="create_alert",
        resource_type="alert",
        resource_id=db_alert.id,
        details={"detection_id": detection.id, "title": db_alert.title},
    ))
    db.commit()

    return db_alert


@router.get("", response_model=List[AlertResponse])
async def list_alerts(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_any_authenticated),
    status_filter: AlertStatus = None,
    limit: int = 100,
    offset: int = 0
):
    """List alerts with optional status filter. Company users see only their company's alerts."""
    query = db.query(Alert)
    
    # Filter by company (unless Aegis AI admin)
    company_filter = get_user_company_filter(current_user)
    if company_filter is not None:
        query = query.filter(Alert.company_id == company_filter)
    
    if status_filter:
        query = query.filter(Alert.status == status_filter)
    
    alerts = query.order_by(Alert.created_at.desc()).offset(offset).limit(limit).all()
    return alerts


@router.get("/{alert_id}", response_model=AlertResponse)
async def get_alert(
    alert_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_any_authenticated)
):
    """Get alert by ID. Company users can only access their company's alerts."""
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Alert not found"
        )
    
    # Check company access
    if alert.company_id and not check_company_access(current_user, alert.company_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to access this alert"
        )
    
    return alert


@router.put("/{alert_id}", response_model=AlertResponse)
async def update_alert(
    alert_id: int,
    alert_update: AlertUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_security_officer)
):
    """Update alert status. Company users can only update their company's alerts."""
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Alert not found"
        )
    
    # Check company access
    if alert.company_id and not check_company_access(current_user, alert.company_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to update this alert"
        )
    
    update_data = alert_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(alert, field, value)
    
    # Update acknowledgment/resolution timestamps
    if alert.status == AlertStatus.ACKNOWLEDGED and not alert.acknowledged_at:
        alert.acknowledged_by = current_user.username
        alert.acknowledged_at = datetime.utcnow()
    elif alert.status == AlertStatus.RESOLVED and not alert.resolved_at:
        alert.resolved_at = datetime.utcnow()

    # Audit log
    db.add(AuditLog(
        user_id=current_user.id,
        action="update_alert",
        resource_type="alert",
        resource_id=alert.id,
        details={"new_status": alert.status.value},
    ))
    db.commit()
    db.refresh(alert)

    return alert


@router.post("/{alert_id}/acknowledge", response_model=AlertResponse)
async def acknowledge_alert(
    alert_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_security_officer)
):
    """Acknowledge an alert. Company users can only acknowledge their company's alerts."""
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Alert not found"
        )
    
    # Check company access
    if alert.company_id and not check_company_access(current_user, alert.company_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to acknowledge this alert"
        )
    
    alert.status = AlertStatus.ACKNOWLEDGED
    alert.acknowledged_by = current_user.username
    alert.acknowledged_at = datetime.utcnow()

    # Audit log
    db.add(AuditLog(
        user_id=current_user.id,
        action="acknowledge_alert",
        resource_type="alert",
        resource_id=alert.id,
    ))
    db.commit()
    db.refresh(alert)

    return alert

