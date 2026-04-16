"""
Analytics endpoints for dashboard visualizations.
All endpoints respect multi-tenant isolation via company filtering.
"""
from fastapi import Request, Query, APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import func, and_
from app.database import get_db
from app.core.security import require_any_authenticated, get_current_user, get_user_company_filter
from app.models.user import Role
from app.models.detection import Detection, DetectionType, ThreatLevel
from app.models.camera import Camera
from app.models.user import User
from datetime import datetime, timedelta
from typing import Optional, Dict, List

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/heatmap")
async def get_heatmap_data(
    request: Request,
    company_id: Optional[int] = Query(None, description="Filter by company (aegis admin)"),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_any_authenticated),
    days: int = 7
):
    """Get heatmap data for incident locations."""
    company_filter = get_user_company_filter(current_user, company_id)
    if company_filter is None and current_user.role != Role.AEGIS_ADMIN:
        return {"heatmap": []}

    start_date = datetime.utcnow() - timedelta(days=days)

    # Select only the 4 columns needed — avoid loading full Detection ORM objects
    base_filter = [
        Detection.detected_at >= start_date,
        Camera.latitude.isnot(None),
        Camera.longitude.isnot(None),
    ]
    if company_filter is not None:
        base_filter.append(Detection.company_id == company_filter)

    rows = (
        db.query(
            Camera.latitude,
            Camera.longitude,
            Camera.zone,
            Detection.threat_level,
        )
        .join(Camera, Detection.camera_id == Camera.id)
        .filter(*base_filter)
        .all()
    )

    heatmap_data: Dict[str, dict] = {}
    for lat, lon, zone, threat_level in rows:
        key = f"{lat},{lon}"
        if key not in heatmap_data:
            heatmap_data[key] = {
                "latitude": lat,
                "longitude": lon,
                "zone": zone,
                "count": 0,
                "high_threat_count": 0,
            }
        heatmap_data[key]["count"] += 1
        if threat_level in (ThreatLevel.HIGH, ThreatLevel.CRITICAL):
            heatmap_data[key]["high_threat_count"] += 1

    return {"heatmap": list(heatmap_data.values())}


@router.get("/timeline")
async def get_timeline_data(
    request: Request,
    company_id: Optional[int] = Query(None, description="Filter by company (aegis admin)"),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_any_authenticated),
    days: int = 7
):
    """Get timeline data for detections over time."""
    company_filter = get_user_company_filter(current_user, company_id)
    if company_filter is None and current_user.role != Role.AEGIS_ADMIN:
        return {"timeline": []}

    start_date = datetime.utcnow() - timedelta(days=days)

    query = db.query(
        func.date_trunc('hour', Detection.detected_at).label('hour'),
        func.count(Detection.id).label('count')
    ).filter(Detection.detected_at >= start_date)
    if company_filter is not None:
        query = query.filter(Detection.company_id == company_filter)

    detections = query.group_by(
        func.date_trunc('hour', Detection.detected_at)
    ).order_by('hour').all()

    timeline = [{"timestamp": str(hour), "count": count} for hour, count in detections]

    return {"timeline": timeline}


@router.get("/by-zone")
async def get_detections_by_zone(
    request: Request,
    company_id: Optional[int] = Query(None, description="Filter by company (aegis admin)"),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_any_authenticated),
    days: int = 7
):
    """Get detection counts grouped by zone."""
    company_filter = get_user_company_filter(current_user, company_id)
    if company_filter is None and current_user.role != Role.AEGIS_ADMIN:
        return {"by_zone": []}

    start_date = datetime.utcnow() - timedelta(days=days)

    query = db.query(
        Camera.zone,
        func.count(Detection.id).label('count')
    ).join(
        Detection, Camera.id == Detection.camera_id
    ).filter(Detection.detected_at >= start_date)
    if company_filter is not None:
        query = query.filter(Detection.company_id == company_filter)

    results = query.group_by(Camera.zone).all()

    zone_data = [{"zone": zone or "Unknown", "count": count} for zone, count in results]

    return {"by_zone": zone_data}


@router.get("/threat-distribution")
async def get_threat_distribution(
    request: Request,
    company_id: Optional[int] = Query(None, description="Filter by company (aegis admin)"),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_any_authenticated),
    days: int = 7
):
    """Get threat level distribution."""
    company_filter = get_user_company_filter(current_user, company_id)
    if company_filter is None and current_user.role != Role.AEGIS_ADMIN:
        return {"distribution": {}}

    start_date = datetime.utcnow() - timedelta(days=days)

    query = db.query(
        Detection.threat_level,
        func.count(Detection.id).label('count')
    ).filter(Detection.detected_at >= start_date)
    if company_filter is not None:
        query = query.filter(Detection.company_id == company_filter)

    results = query.group_by(Detection.threat_level).all()

    distribution = {threat.value: count for threat, count in results}

    return {"distribution": distribution}


@router.get("/top-cameras")
async def get_top_cameras(
    request: Request,
    company_id: Optional[int] = Query(None, description="Filter by company (aegis admin)"),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_any_authenticated),
    days: int = 7,
    limit: int = 10
):
    """Get top cameras by detection count."""
    company_filter = get_user_company_filter(current_user, company_id)
    if company_filter is None and current_user.role != Role.AEGIS_ADMIN:
        return {"top_cameras": []}

    start_date = datetime.utcnow() - timedelta(days=days)

    query = db.query(
        Camera.name,
        Camera.location,
        func.count(Detection.id).label('count')
    ).join(
        Detection, Camera.id == Detection.camera_id
    ).filter(Detection.detected_at >= start_date)
    if company_filter is not None:
        query = query.filter(Detection.company_id == company_filter)

    results = query.group_by(
        Camera.id, Camera.name, Camera.location
    ).order_by(
        func.count(Detection.id).desc()
    ).limit(limit).all()

    top_cameras = [
        {"name": name, "location": location, "count": count}
        for name, location, count in results
    ]

    return {"top_cameras": top_cameras}

