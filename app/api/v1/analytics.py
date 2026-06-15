"""
Analytics endpoints for dashboard visualizations.
All endpoints respect multi-tenant isolation via company filtering.
"""

from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy import case, func
from sqlalchemy.orm import Session

from app.core.security import get_user_company_filter, require_any_authenticated
from app.database import get_db
from app.models.camera import Camera
from app.models.detection import Detection, ThreatLevel
from app.models.user import User

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/heatmap")
async def get_heatmap_data(
    request: Request,
    company_id: Optional[int] = Query(None, description="Filter by company (aegis admin)"),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_any_authenticated),
    days: int = 7,
):
    """Get heatmap data for incident locations."""
    company_filter = get_user_company_filter(current_user, company_id)
    if company_filter is None:
        return {"heatmap": []}

    start_date = datetime.utcnow() - timedelta(days=days)

    rows = (
        db.query(
            Camera.latitude,
            Camera.longitude,
            Camera.zone,
            func.count(Detection.id).label("count"),
            func.sum(
                case(
                    (Detection.threat_level.in_([ThreatLevel.HIGH, ThreatLevel.CRITICAL]), 1),
                    else_=0,
                )
            ).label("high_threat_count"),
        )
        .join(Camera, Detection.camera_id == Camera.id)
        .filter(
            Detection.detected_at >= start_date,
            Camera.latitude.isnot(None),
            Camera.longitude.isnot(None),
            Detection.company_id == company_filter,
        )
        .group_by(Camera.latitude, Camera.longitude, Camera.zone)
        .all()
    )

    return {
        "heatmap": [
            {
                "latitude": lat,
                "longitude": lon,
                "zone": zone,
                "count": int(count or 0),
                "high_threat_count": int(high or 0),
            }
            for lat, lon, zone, count, high in rows
        ]
    }


@router.get("/timeline")
async def get_timeline_data(
    request: Request,
    company_id: Optional[int] = Query(None, description="Filter by company (aegis admin)"),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_any_authenticated),
    days: int = 7,
):
    """Get timeline data for detections over time."""
    company_filter = get_user_company_filter(current_user, company_id)
    if company_filter is None:
        return {"timeline": []}

    start_date = datetime.utcnow() - timedelta(days=days)

    query = db.query(
        func.date_trunc("hour", Detection.detected_at).label("hour"),
        func.count(Detection.id).label("count"),
    ).filter(Detection.detected_at >= start_date, Detection.company_id == company_filter)

    detections = (
        query.group_by(func.date_trunc("hour", Detection.detected_at)).order_by("hour").all()
    )

    timeline = [{"timestamp": str(hour), "count": count} for hour, count in detections]

    return {"timeline": timeline}


@router.get("/by-zone")
async def get_detections_by_zone(
    request: Request,
    company_id: Optional[int] = Query(None, description="Filter by company (aegis admin)"),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_any_authenticated),
    days: int = 7,
):
    """Get detection counts grouped by zone."""
    company_filter = get_user_company_filter(current_user, company_id)
    if company_filter is None:
        return {"by_zone": []}

    start_date = datetime.utcnow() - timedelta(days=days)

    query = (
        db.query(Camera.zone, func.count(Detection.id).label("count"))
        .join(Detection, Camera.id == Detection.camera_id)
        .filter(Detection.detected_at >= start_date, Detection.company_id == company_filter)
    )

    results = query.group_by(Camera.zone).all()

    zone_data = [{"zone": zone or "Unknown", "count": count} for zone, count in results]

    return {"by_zone": zone_data}


@router.get("/threat-distribution")
async def get_threat_distribution(
    request: Request,
    company_id: Optional[int] = Query(None, description="Filter by company (aegis admin)"),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_any_authenticated),
    days: int = 7,
):
    """Get threat level distribution."""
    company_filter = get_user_company_filter(current_user, company_id)
    if company_filter is None:
        return {"distribution": {}}

    start_date = datetime.utcnow() - timedelta(days=days)

    query = db.query(Detection.threat_level, func.count(Detection.id).label("count")).filter(
        Detection.detected_at >= start_date,
        Detection.company_id == company_filter,
    )

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
    limit: int = 10,
):
    """Get top cameras by detection count."""
    company_filter = get_user_company_filter(current_user, company_id)
    if company_filter is None:
        return {"top_cameras": []}

    start_date = datetime.utcnow() - timedelta(days=days)

    query = (
        db.query(
            Camera.id,
            Camera.name,
            Camera.location,
            func.count(Detection.id).label("count"),
        )
        .join(Detection, Camera.id == Detection.camera_id)
        .filter(Detection.detected_at >= start_date, Detection.company_id == company_filter)
    )

    results = (
        query.group_by(Camera.id, Camera.name, Camera.location)
        .order_by(func.count(Detection.id).desc())
        .limit(limit)
        .all()
    )

    top_cameras = [
        {
            "camera_id": cam_id,
            "camera_name": name or "",
            "location": location or "",
            "detection_count": int(count or 0),
        }
        for cam_id, name, location, count in results
    ]

    return {"top_cameras": top_cameras}


@router.get("/all")
async def get_all_analytics(
    request: Request,
    company_id: Optional[int] = Query(None),
    days: int = Query(7, ge=1, le=365),
    top_cameras_limit: int = Query(5, ge=1, le=20),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_any_authenticated),
):
    """
    Single endpoint that returns all analytics charts in one DB round-trip.
    Replaces the 5 individual calls (/timeline, /threat-distribution, /by-zone,
    /top-cameras, /heatmap) with a single request.
    """
    company_filter = get_user_company_filter(current_user, company_id)
    if company_filter is None:
        return {
            "timeline": [],
            "distribution": {},
            "by_zone": [],
            "top_cameras": [],
            "heatmap": [],
        }

    start_date = datetime.utcnow() - timedelta(days=days)

    # 1. Timeline — detections per hour
    timeline_rows = (
        db.query(
            func.date_trunc("hour", Detection.detected_at).label("hour"),
            func.count(Detection.id).label("count"),
        )
        .filter(Detection.detected_at >= start_date, Detection.company_id == company_filter)
        .group_by(func.date_trunc("hour", Detection.detected_at))
        .order_by("hour")
        .all()
    )
    timeline = [{"timestamp": str(hour), "count": count} for hour, count in timeline_rows]

    # 2. Threat distribution
    dist_rows = (
        db.query(Detection.threat_level, func.count(Detection.id).label("count"))
        .filter(Detection.detected_at >= start_date, Detection.company_id == company_filter)
        .group_by(Detection.threat_level)
        .all()
    )
    distribution = {threat.value: count for threat, count in dist_rows}

    # 3. By zone
    zone_rows = (
        db.query(Camera.zone, func.count(Detection.id).label("count"))
        .join(Detection, Camera.id == Detection.camera_id)
        .filter(Detection.detected_at >= start_date, Detection.company_id == company_filter)
        .group_by(Camera.zone)
        .all()
    )
    by_zone = [{"zone": zone or "Unknown", "count": count} for zone, count in zone_rows]

    # 4. Top cameras
    cam_rows = (
        db.query(
            Camera.id,
            Camera.name,
            Camera.location,
            func.count(Detection.id).label("count"),
        )
        .join(Detection, Camera.id == Detection.camera_id)
        .filter(Detection.detected_at >= start_date, Detection.company_id == company_filter)
        .group_by(Camera.id, Camera.name, Camera.location)
        .order_by(func.count(Detection.id).desc())
        .limit(top_cameras_limit)
        .all()
    )
    top_cameras = [
        {
            "camera_id": cid,
            "camera_name": name or "",
            "location": loc or "",
            "detection_count": int(cnt or 0),
        }
        for cid, name, loc, cnt in cam_rows
    ]

    # 5. Heatmap — aggregated in SQL (no Python row-by-row loop)
    heatmap_rows = (
        db.query(
            Camera.latitude,
            Camera.longitude,
            Camera.zone,
            func.count(Detection.id).label("count"),
            func.sum(
                case(
                    (Detection.threat_level.in_([ThreatLevel.HIGH, ThreatLevel.CRITICAL]), 1),
                    else_=0,
                )
            ).label("high_threat_count"),
        )
        .join(Camera, Detection.camera_id == Camera.id)
        .filter(
            Detection.detected_at >= start_date,
            Camera.latitude.isnot(None),
            Camera.longitude.isnot(None),
            Detection.company_id == company_filter,
        )
        .group_by(Camera.latitude, Camera.longitude, Camera.zone)
        .all()
    )
    heatmap = [
        {
            "latitude": lat,
            "longitude": lon,
            "zone": zone,
            "count": int(cnt or 0),
            "high_threat_count": int(high or 0),
        }
        for lat, lon, zone, cnt, high in heatmap_rows
    ]

    return {
        "timeline": timeline,
        "distribution": distribution,
        "by_zone": by_zone,
        "top_cameras": top_cameras,
        "heatmap": heatmap,
    }
