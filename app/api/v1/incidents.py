"""
Incident management endpoints — group alerts into security incidents.
Includes enriched views with detection context, camera info, and evidence images.
"""
from typing import List, Optional
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status, Request, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.core.security import require_any_authenticated, require_security_officer, get_user_company_filter, check_company_access
from app.models.user import User
from app.models.alert import Alert
from app.models.detection import Detection, DetectionType, ThreatLevel
from app.models.camera import Camera
from app.models.evidence import Evidence
from app.models.incident import Incident, IncidentStatus, IncidentSeverity
from app.models.audit_log import create_audit_log
from app.schemas.incident import (
    IncidentCreate, IncidentUpdate, IncidentAddAlerts,
    IncidentResponse, IncidentDetailResponse, IncidentAlertSummary,
)

router = APIRouter(prefix="/incidents", tags=["incidents"])


def _build_alert_summary(alert: Alert, detection: Detection, camera: Camera, evidence: Evidence = None) -> IncidentAlertSummary:
    image_url = None
    if evidence and evidence.image_path:
        image_url = f"/evidence/{Path(evidence.image_path).name}"

    return IncidentAlertSummary(
        id=alert.id,
        title=alert.title,
        status=alert.status.value,
        detection_type=detection.detection_type.value if detection else None,
        threat_level=detection.threat_level.value if detection else None,
        confidence=detection.confidence if detection else None,
        camera_name=camera.name if camera else None,
        evidence_image_url=image_url,
        created_at=alert.created_at,
    )


def _build_response(incident: Incident, include_alerts: bool = False, db: Session = None) -> dict:
    """Build incident response, optionally with enriched alert details."""
    creator_username = None
    if incident.creator:
        creator_username = incident.creator.username

    resp = IncidentResponse(
        id=incident.id,
        company_id=incident.company_id,
        title=incident.title,
        description=incident.description,
        status=incident.status,
        severity=incident.severity,
        assigned_to=incident.assigned_to,
        created_by=incident.created_by,
        created_by_username=creator_username,
        alert_count=len(incident.alerts),
        resolved_at=incident.resolved_at,
        created_at=incident.created_at,
        updated_at=incident.updated_at,
    )

    if not include_alerts or not db:
        return resp

    # Enrich alerts with detection/camera/evidence
    alert_summaries = []
    for alert in incident.alerts:
        detection = db.query(Detection).filter(Detection.id == alert.detection_id).first()
        camera = db.query(Camera).filter(Camera.id == detection.camera_id).first() if detection else None
        evidence = db.query(Evidence).filter(Evidence.detection_id == alert.detection_id).first()
        alert_summaries.append(_build_alert_summary(alert, detection, camera, evidence))

    return IncidentDetailResponse(
        **resp.model_dump(),
        alerts=alert_summaries,
    )


# ---------------------------------------------------------------------------
# POST /incidents
# ---------------------------------------------------------------------------
@router.post("", response_model=IncidentDetailResponse, status_code=status.HTTP_201_CREATED)
async def create_incident(
    request: Request,
    data: IncidentCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_security_officer),
):
    """Create an incident, optionally linking existing alerts."""
    company_filter = get_user_company_filter(current_user)

    # Determine company_id from linked alerts or user's company
    company_id = current_user.company_id
    linked_alerts = []

    if data.alert_ids:
        linked_alerts = db.query(Alert).filter(Alert.id.in_(data.alert_ids)).all()
        if not linked_alerts:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No matching alerts found")

        # Use company from first alert
        company_id = linked_alerts[0].company_id

        # Verify access to all alerts
        for alert in linked_alerts:
            if alert.company_id and not check_company_access(current_user, alert.company_id):
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f"No access to alert {alert.id}")

    if not company_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot determine company for incident")

    incident = Incident(
        company_id=company_id,
        title=data.title,
        description=data.description,
        severity=data.severity,
        assigned_to=data.assigned_to,
        created_by=current_user.id,
    )
    incident.alerts = linked_alerts
    db.add(incident)
    db.commit()
    db.refresh(incident)

    db.add(create_audit_log(
        request, current_user.id, "create_incident", "incident", incident.id,
        {"title": incident.title, "alert_count": len(linked_alerts)}
    ))
    db.commit()

    return _build_response(incident, include_alerts=True, db=db)


# ---------------------------------------------------------------------------
# GET /incidents  —  list with filters
# ---------------------------------------------------------------------------
@router.get("", response_model=List[IncidentResponse])
async def list_incidents(
    status_filter: Optional[IncidentStatus] = Query(None, alias="status"),
    severity: Optional[IncidentSeverity] = Query(None),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_any_authenticated),
):
    """List incidents with filters."""
    query = db.query(Incident)

    company_filter = get_user_company_filter(current_user)
    if company_filter is not None:
        query = query.filter(Incident.company_id == company_filter)

    if status_filter:
        query = query.filter(Incident.status == status_filter)
    if severity:
        query = query.filter(Incident.severity == severity)
    if start_date:
        query = query.filter(Incident.created_at >= start_date)
    if end_date:
        query = query.filter(Incident.created_at <= end_date)

    incidents = query.order_by(Incident.created_at.desc()).offset(offset).limit(limit).all()

    return [_build_response(inc) for inc in incidents]


# ---------------------------------------------------------------------------
# GET /incidents/{id}  —  detail with alerts + evidence images
# ---------------------------------------------------------------------------
@router.get("/{incident_id}", response_model=IncidentDetailResponse)
async def get_incident(
    incident_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_any_authenticated),
):
    """Get incident detail with linked alerts, detection info, and evidence images."""
    incident = db.query(Incident).filter(Incident.id == incident_id).first()
    if not incident:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Incident not found")

    if incident.company_id and not check_company_access(current_user, incident.company_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions")

    return _build_response(incident, include_alerts=True, db=db)


# ---------------------------------------------------------------------------
# PUT /incidents/{id}  —  update
# ---------------------------------------------------------------------------
@router.put("/{incident_id}", response_model=IncidentResponse)
async def update_incident(
    request: Request,
    incident_id: int,
    data: IncidentUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_security_officer),
):
    """Update incident status, severity, assignment, etc."""
    incident = db.query(Incident).filter(Incident.id == incident_id).first()
    if not incident:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Incident not found")

    if incident.company_id and not check_company_access(current_user, incident.company_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions")

    update_data = data.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(incident, field, value)

    if incident.status == IncidentStatus.RESOLVED and not incident.resolved_at:
        incident.resolved_at = datetime.utcnow()

    db.add(create_audit_log(
        request, current_user.id, "update_incident", "incident", incident.id,
        {"updates": update_data}
    ))
    db.commit()
    db.refresh(incident)

    return _build_response(incident)


# ---------------------------------------------------------------------------
# POST /incidents/{id}/alerts  —  link more alerts
# ---------------------------------------------------------------------------
@router.post("/{incident_id}/alerts", response_model=IncidentDetailResponse)
async def add_alerts_to_incident(
    request: Request,
    incident_id: int,
    data: IncidentAddAlerts,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_security_officer),
):
    """Link additional alerts to an existing incident."""
    incident = db.query(Incident).filter(Incident.id == incident_id).first()
    if not incident:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Incident not found")

    if incident.company_id and not check_company_access(current_user, incident.company_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions")

    new_alerts = db.query(Alert).filter(Alert.id.in_(data.alert_ids)).all()
    if not new_alerts:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No matching alerts found")

    existing_ids = {a.id for a in incident.alerts}
    added = 0
    for alert in new_alerts:
        if alert.company_id and not check_company_access(current_user, alert.company_id):
            continue
        if alert.id not in existing_ids:
            incident.alerts.append(alert)
            added += 1

    db.add(create_audit_log(
        request, current_user.id, "add_alerts_to_incident", "incident", incident.id,
        {"alert_ids": data.alert_ids, "added": added}
    ))
    db.commit()
    db.refresh(incident)

    return _build_response(incident, include_alerts=True, db=db)


# ---------------------------------------------------------------------------
# DELETE /incidents/{id}/alerts/{alert_id}  —  unlink an alert
# ---------------------------------------------------------------------------
@router.delete("/{incident_id}/alerts/{alert_id}", response_model=IncidentDetailResponse)
async def remove_alert_from_incident(
    request: Request,
    incident_id: int,
    alert_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_security_officer),
):
    """Remove an alert from an incident."""
    incident = db.query(Incident).filter(Incident.id == incident_id).first()
    if not incident:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Incident not found")

    if incident.company_id and not check_company_access(current_user, incident.company_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions")

    incident.alerts = [a for a in incident.alerts if a.id != alert_id]

    db.add(create_audit_log(
        request, current_user.id, "remove_alert_from_incident", "incident", incident.id,
        {"removed_alert_id": alert_id}
    ))
    db.commit()
    db.refresh(incident)

    return _build_response(incident, include_alerts=True, db=db)
