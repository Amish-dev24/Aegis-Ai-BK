"""
Alert management endpoints with enriched views, filters, and evidence images.
"""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Request, Query
from sqlalchemy.orm import Session
from app.database import get_db
from app.core.security import require_any_authenticated, require_security_officer, get_user_company_filter, check_company_access
from app.schemas.alert import AlertCreate, AlertResponse, AlertDetailResponse, AlertUpdate
from app.models.alert import Alert, AlertStatus
from app.models.detection import Detection, DetectionType, ThreatLevel
from app.models.camera import Camera
from app.models.evidence import Evidence
from app.models.user import User
from app.services.email_service import email_service
from app.models.audit_log import create_audit_log
from datetime import datetime
from pathlib import Path

router = APIRouter(prefix="/alerts", tags=["alerts"])


def _enrich_alert(alert: Alert, detection: Detection = None, camera: Camera = None, evidence: Evidence = None) -> dict:
    """Build enriched alert dict with detection context, camera info, evidence image."""
    image_url = None
    if evidence and evidence.image_path:
        filename = Path(evidence.image_path).name
        image_url = f"/evidence/{filename}"

    return AlertDetailResponse(
        id=alert.id,
        detection_id=alert.detection_id,
        company_id=alert.company_id,
        title=alert.title,
        message=alert.message,
        status=alert.status,
        email_sent=alert.email_sent,
        email_sent_at=alert.email_sent_at,
        acknowledged_by=alert.acknowledged_by,
        acknowledged_at=alert.acknowledged_at,
        resolved_at=alert.resolved_at,
        created_at=alert.created_at,
        # Detection context
        detection_type=detection.detection_type.value if detection else None,
        threat_level=detection.threat_level.value if detection else None,
        confidence=detection.confidence if detection else None,
        detection_timestamp=detection.frame_timestamp if detection else None,
        # Camera
        camera_id=detection.camera_id if detection else None,
        camera_name=camera.name if camera else None,
        camera_location=camera.location if camera else None,
        # Evidence
        evidence_id=evidence.id if evidence else None,
        evidence_image_url=image_url,
    )


# ---------------------------------------------------------------------------
# POST /alerts  —  create alert
# ---------------------------------------------------------------------------
@router.post("", response_model=AlertResponse, status_code=status.HTTP_201_CREATED)
async def create_alert(
    request: Request,
    alert_data: AlertCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_security_officer)
):
    """Create an alert. Company users can only create alerts for their company's detections."""
    detection = db.query(Detection).filter(Detection.id == alert_data.detection_id).first()
    if not detection:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Detection not found")

    if detection.company_id and not check_company_access(current_user, detection.company_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions")

    alert_dict = alert_data.dict()
    alert_dict["company_id"] = detection.company_id

    db_alert = Alert(**alert_dict)
    db.add(db_alert)
    db.commit()
    db.refresh(db_alert)

    # Send email for high/critical
    if detection.threat_level.value in ["high", "critical"]:
        evidence = db.query(Evidence).filter(Evidence.detection_id == detection.id).first()
        snapshot_path = evidence.image_path if evidence else None

        await email_service.send_alert_email(
            to_emails=[current_user.email],
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
        db_alert.email_sent_to = current_user.email
        db_alert.email_sent_at = datetime.utcnow()
        db.commit()

    db.add(create_audit_log(
        request, current_user.id, "create_alert", "alert", db_alert.id,
        {"detection_id": detection.id, "title": db_alert.title}
    ))
    db.commit()

    return db_alert


# ---------------------------------------------------------------------------
# GET /alerts  —  list with filters + enriched response
# ---------------------------------------------------------------------------
@router.get("", response_model=List[AlertDetailResponse])
async def list_alerts(
    status_filter: Optional[AlertStatus] = Query(None, alias="status", description="Filter by alert status"),
    threat_level: Optional[ThreatLevel] = Query(None, description="Filter by threat level"),
    detection_type: Optional[DetectionType] = Query(None, description="Filter by detection type"),
    camera_id: Optional[int] = Query(None, description="Filter by camera"),
    start_date: Optional[datetime] = Query(None, description="Filter from date"),
    end_date: Optional[datetime] = Query(None, description="Filter to date"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_any_authenticated),
):
    """List alerts with filters. Includes detection context, camera info, and evidence image."""
    company_filter = get_user_company_filter(current_user)

    # Join alert → detection → camera, outer-join evidence (first per detection)
    query = db.query(Alert, Detection, Camera).join(
        Detection, Alert.detection_id == Detection.id
    ).join(
        Camera, Detection.camera_id == Camera.id
    )

    # Company isolation
    if company_filter is not None:
        query = query.filter(Alert.company_id == company_filter)

    # Filters
    if status_filter:
        query = query.filter(Alert.status == status_filter)
    if threat_level:
        query = query.filter(Detection.threat_level == threat_level)
    if detection_type:
        query = query.filter(Detection.detection_type == detection_type)
    if camera_id:
        query = query.filter(Detection.camera_id == camera_id)
    if start_date:
        query = query.filter(Alert.created_at >= start_date)
    if end_date:
        query = query.filter(Alert.created_at <= end_date)

    results = query.order_by(Alert.created_at.desc()).offset(offset).limit(limit).all()

    # Fetch evidence for each detection (first snapshot)
    detection_ids = [det.id for _, det, _ in results]
    evidence_map = {}
    if detection_ids:
        evidences = db.query(Evidence).filter(Evidence.detection_id.in_(detection_ids)).all()
        for ev in evidences:
            if ev.detection_id not in evidence_map:
                evidence_map[ev.detection_id] = ev

    return [
        _enrich_alert(alert, detection, camera, evidence_map.get(detection.id))
        for alert, detection, camera in results
    ]


# ---------------------------------------------------------------------------
# GET /alerts/{alert_id}  —  single alert with full detail
# ---------------------------------------------------------------------------
@router.get("/{alert_id}", response_model=AlertDetailResponse)
async def get_alert(
    alert_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_any_authenticated)
):
    """Get alert by ID with detection context, camera info, and evidence image."""
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Alert not found")

    if alert.company_id and not check_company_access(current_user, alert.company_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions")

    detection = db.query(Detection).filter(Detection.id == alert.detection_id).first()
    camera = db.query(Camera).filter(Camera.id == detection.camera_id).first() if detection else None
    evidence = db.query(Evidence).filter(Evidence.detection_id == alert.detection_id).first()

    return _enrich_alert(alert, detection, camera, evidence)


# ---------------------------------------------------------------------------
# PUT /alerts/{alert_id}  —  update alert
# ---------------------------------------------------------------------------
@router.put("/{alert_id}", response_model=AlertResponse)
async def update_alert(
    request: Request,
    alert_id: int,
    alert_update: AlertUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_security_officer)
):
    """Update alert status."""
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Alert not found")

    if alert.company_id and not check_company_access(current_user, alert.company_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions")

    update_data = alert_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(alert, field, value)

    if alert.status == AlertStatus.ACKNOWLEDGED and not alert.acknowledged_at:
        alert.acknowledged_by = current_user.username
        alert.acknowledged_at = datetime.utcnow()
    elif alert.status == AlertStatus.RESOLVED and not alert.resolved_at:
        alert.resolved_at = datetime.utcnow()

    db.add(create_audit_log(
        request, current_user.id, "update_alert", "alert", alert.id,
        {"new_status": alert.status.value}
    ))
    db.commit()
    db.refresh(alert)

    return alert


# ---------------------------------------------------------------------------
# POST /alerts/{alert_id}/acknowledge
# ---------------------------------------------------------------------------
@router.post("/{alert_id}/acknowledge", response_model=AlertResponse)
async def acknowledge_alert(
    request: Request,
    alert_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_security_officer)
):
    """Acknowledge an alert."""
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Alert not found")

    if alert.company_id and not check_company_access(current_user, alert.company_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions")

    alert.status = AlertStatus.ACKNOWLEDGED
    alert.acknowledged_by = current_user.username
    alert.acknowledged_at = datetime.utcnow()

    db.add(create_audit_log(
        request, current_user.id, "acknowledge_alert", "alert", alert.id,
        {"title": alert.title, "acknowledged_by": current_user.username}
    ))
    db.commit()
    db.refresh(alert)

    return alert
