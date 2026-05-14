"""
Alert management endpoints with enriched views, filters, evidence images, and incident logs.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.security import (
    check_company_access,
    check_directory_company_access,
    get_directory_company_filter,
    require_any_authenticated,
    require_security_officer,
)
from app.database import get_db
from app.models.alert import Alert, AlertStatus
from app.models.alert_log import AlertLog
from app.models.audit_log import create_audit_log
from app.models.camera import Camera
from app.models.detection import Detection, DetectionType, ThreatLevel
from app.models.evidence import Evidence
from app.models.user import User
from app.schemas.alert import (
    AlertCreate,
    AlertDetailResponse,
    AlertLogCreate,
    AlertLogResponse,
    AlertResponse,
    AlertUpdate,
)
from app.services.email_service import email_service
from app.services.zone_notification_service import get_alert_emails_for_camera

router = APIRouter(prefix="/alerts", tags=["alerts"])


def _build_log_responses(alert: Alert, db: Session) -> list:
    """Build log responses with usernames (single batched query — no N+1)."""
    logs = (
        db.query(AlertLog).filter(AlertLog.alert_id == alert.id).order_by(AlertLog.created_at).all()
    )
    if not logs:
        return []
    # Batch-load all referenced users in one query
    user_ids = list({log.user_id for log in logs})
    username_map: dict[int, str] = {
        u.id: u.username
        for u in db.query(User.id, User.username).filter(User.id.in_(user_ids)).all()
    }
    return [
        AlertLogResponse(
            id=log.id,
            alert_id=log.alert_id,
            user_id=log.user_id,
            username=username_map.get(log.user_id),
            action=log.action,
            message=log.message,
            created_at=log.created_at,
        )
        for log in logs
    ]


def _latest_alert_notes(
    db: Session, alert_ids: list[int]
) -> dict[int, tuple[Optional[str], Optional[datetime]]]:
    """One row per alert: message and time of most recent AlertLog."""
    if not alert_ids:
        return {}
    subq = (
        db.query(
            AlertLog.alert_id,
            func.max(AlertLog.created_at).label("mx"),
        )
        .filter(AlertLog.alert_id.in_(alert_ids))
        .group_by(AlertLog.alert_id)
        .subquery()
    )
    rows = (
        db.query(AlertLog)
        .join(
            subq,
            (AlertLog.alert_id == subq.c.alert_id) & (AlertLog.created_at == subq.c.mx),
        )
        .all()
    )
    return {r.alert_id: (r.message, r.created_at) for r in rows}


def _enrich_alert(
    alert: Alert,
    detection: Detection = None,
    camera: Camera = None,
    evidence: Evidence = None,
    logs: list = None,
    *,
    latest_note: Optional[str] = None,
    latest_note_at: Optional[datetime] = None,
) -> AlertDetailResponse:
    """Build enriched alert with detection context, camera info, evidence image, and logs."""
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
        email_sent_to=alert.email_sent_to,
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
        # Logs
        logs=logs or [],
        latest_note=latest_note,
        latest_note_at=latest_note_at,
    )


# ---------------------------------------------------------------------------
# POST /alerts  —  create alert
# ---------------------------------------------------------------------------
@router.post("", response_model=AlertResponse, status_code=status.HTTP_201_CREATED)
async def create_alert(
    request: Request,
    alert_data: AlertCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_security_officer),
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

        # Resolve camera for zone lookup
        camera = db.query(Camera).filter(Camera.id == detection.camera_id).first()

        # All recipients: current user + company admins + zone officer
        alert_emails = get_alert_emails_for_camera(db, camera, current_user.email) if camera else [current_user.email]

        await email_service.send_alert_email(
            to_emails=alert_emails,
            subject=db_alert.title,
            message=db_alert.message or f"Alert for detection {detection.id}",
            snapshot_path=snapshot_path,
            metadata={
                "Detection Type": detection.detection_type.value,
                "Threat Level": detection.threat_level.value,
                "Confidence": f"{detection.confidence:.2%}",
                "Camera": camera.name if camera else "—",
                "Zone": (camera.zone or "—") if camera else "—",
                "Timestamp": detection.frame_timestamp.isoformat(),
            },
        )
        db_alert.email_sent = True
        db_alert.email_sent_to = ", ".join(alert_emails)
        db_alert.email_sent_at = datetime.utcnow()
        db.commit()

    db.add(
        create_audit_log(
            request,
            current_user.id,
            "create_alert",
            "alert",
            db_alert.id,
            {"detection_id": detection.id, "title": db_alert.title},
        )
    )
    db.commit()

    return db_alert


# ---------------------------------------------------------------------------
# GET /alerts  —  list with filters + enriched response
# ---------------------------------------------------------------------------
@router.get("", response_model=list[AlertDetailResponse])
async def list_alerts(
    request: Request,
    status_filter: Optional[AlertStatus] = Query(
        None, alias="status", description="Filter by alert status"
    ),
    threat_level: Optional[ThreatLevel] = Query(None, description="Filter by threat level"),
    detection_type: Optional[DetectionType] = Query(None, description="Filter by detection type"),
    camera_id: Optional[int] = Query(None, description="Filter by camera"),
    start_date: Optional[datetime] = Query(None, description="Filter from date"),
    end_date: Optional[datetime] = Query(None, description="Filter to date"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    company_id: Optional[int] = Query(
        None,
        description="Company scope (aegis_admin: any company; company staff: must match tenant)",
    ),
    include_latest_note: bool = Query(
        False,
        alias="include_latest_note",
        description="Include latest AlertLog text per alert (header / incident feed)",
    ),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_any_authenticated),
):
    """List alerts with filters. Includes detection context, camera info, and evidence image."""
    company_filter = get_directory_company_filter(current_user, company_id)
    if company_filter is None:
        return []

    query = (
        db.query(Alert, Detection, Camera)
        .join(Detection, Alert.detection_id == Detection.id)
        .join(Camera, Detection.camera_id == Camera.id)
        .filter(Alert.company_id == company_filter)
    )

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

    # Batch-fetch evidence
    detection_ids = [det.id for _, det, _ in results]
    evidence_map = {}
    if detection_ids:
        evidences = db.query(Evidence).filter(Evidence.detection_id.in_(detection_ids)).all()
        for ev in evidences:
            if ev.detection_id not in evidence_map:
                evidence_map[ev.detection_id] = ev

    note_map: dict[int, tuple[Optional[str], Optional[datetime]]] = {}
    if include_latest_note:
        aids = [a.id for a, _, _ in results]
        note_map = _latest_alert_notes(db, aids)

    out: list[AlertDetailResponse] = []
    for alert, detection, camera in results:
        pair = note_map.get(alert.id) if include_latest_note else None
        ln, ln_at = (pair[0], pair[1]) if pair else (None, None)
        out.append(
            _enrich_alert(
                alert,
                detection,
                camera,
                evidence_map.get(detection.id),
                latest_note=ln,
                latest_note_at=ln_at,
            )
        )
    return out


# ---------------------------------------------------------------------------
# GET /alerts/{alert_id}  —  single alert with full detail + logs
# ---------------------------------------------------------------------------
@router.get("/{alert_id}", response_model=AlertDetailResponse)
async def get_alert(
    alert_id: int,
    company_id: Optional[int] = Query(None, description="Filter by company (aegis admin)"),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_any_authenticated),
):
    """Get alert by ID with detection context, camera info, evidence image, and incident logs."""
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Alert not found")

    if alert.company_id and not check_directory_company_access(current_user, alert.company_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions")

    detection = db.query(Detection).filter(Detection.id == alert.detection_id).first()
    camera = (
        db.query(Camera).filter(Camera.id == detection.camera_id).first() if detection else None
    )
    evidence = db.query(Evidence).filter(Evidence.detection_id == alert.detection_id).first()
    logs = _build_log_responses(alert, db)

    return _enrich_alert(alert, detection, camera, evidence, logs)


# ---------------------------------------------------------------------------
# POST /alerts/{alert_id}/log  —  add incident log entry
# ---------------------------------------------------------------------------
@router.post(
    "/{alert_id}/log", response_model=AlertLogResponse, status_code=status.HTTP_201_CREATED
)
async def add_alert_log(
    request: Request,
    alert_id: int,
    log_data: AlertLogCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_security_officer),
):
    """Add an incident log entry to an alert (notes, actions taken, dispatch info)."""
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Alert not found")

    if alert.company_id and not check_company_access(current_user, alert.company_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions")

    log_entry = AlertLog(
        alert_id=alert_id,
        user_id=current_user.id,
        action=log_data.action,
        message=log_data.message,
    )
    db.add(log_entry)

    db.add(
        create_audit_log(
            request,
            current_user.id,
            "add_alert_log",
            "alert",
            alert_id,
            {"action": log_data.action, "message": log_data.message},
        )
    )
    db.commit()
    db.refresh(log_entry)

    return AlertLogResponse(
        id=log_entry.id,
        alert_id=log_entry.alert_id,
        user_id=log_entry.user_id,
        username=current_user.username,
        action=log_entry.action,
        message=log_entry.message,
        created_at=log_entry.created_at,
    )


# ---------------------------------------------------------------------------
# GET /alerts/{alert_id}/logs  —  list all logs for an alert
# ---------------------------------------------------------------------------
@router.get("/{alert_id}/logs", response_model=list[AlertLogResponse])
async def list_alert_logs(
    alert_id: int,
    company_id: Optional[int] = Query(None, description="Filter by company (aegis admin)"),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_any_authenticated),
):
    """List all incident log entries for an alert."""
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Alert not found")

    if alert.company_id and not check_directory_company_access(current_user, alert.company_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions")

    return _build_log_responses(alert, db)


# ---------------------------------------------------------------------------
# PUT /alerts/{alert_id}  —  update alert
# ---------------------------------------------------------------------------
@router.put("/{alert_id}", response_model=AlertResponse)
async def update_alert(
    request: Request,
    alert_id: int,
    alert_update: AlertUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_security_officer),
):
    """Update alert status."""
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Alert not found")

    if alert.company_id and not check_company_access(current_user, alert.company_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions")

    update_data = alert_update.dict(exclude_unset=True)
    old_status = alert.status.value

    for field, value in update_data.items():
        setattr(alert, field, value)

    if alert.status == AlertStatus.ACKNOWLEDGED and not alert.acknowledged_at:
        alert.acknowledged_by = current_user.username
        alert.acknowledged_at = datetime.utcnow()
    elif alert.status == AlertStatus.RESOLVED and not alert.resolved_at:
        alert.resolved_at = datetime.utcnow()

    # Auto-log status changes
    if "status" in update_data and update_data["status"].value != old_status:
        auto_log = AlertLog(
            alert_id=alert.id,
            user_id=current_user.id,
            action="status_change",
            message=f"Status changed from {old_status} to {alert.status.value}",
        )
        db.add(auto_log)

    db.add(
        create_audit_log(
            request,
            current_user.id,
            "update_alert",
            "alert",
            alert.id,
            {"new_status": alert.status.value},
        )
    )
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
    current_user: User = Depends(require_security_officer),
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

    # Auto-log
    auto_log = AlertLog(
        alert_id=alert.id,
        user_id=current_user.id,
        action="acknowledged",
        message=f"Alert acknowledged by {current_user.username}",
    )
    db.add(auto_log)

    db.add(
        create_audit_log(
            request,
            current_user.id,
            "acknowledge_alert",
            "alert",
            alert.id,
            {"title": alert.title, "acknowledged_by": current_user.username},
        )
    )
    db.commit()
    db.refresh(alert)

    return alert
