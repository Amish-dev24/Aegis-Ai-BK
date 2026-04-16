"""
Evidence management endpoints.
All endpoints respect multi-tenant isolation via company access checks.
"""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, File, UploadFile, Request, Query
from fastapi.responses import FileResponse, Response
from sqlalchemy.orm import Session
from app.database import get_db
from app.core.security import require_any_authenticated, get_current_user, check_company_access, get_user_company_filter
from app.schemas.evidence import EvidenceCreate, EvidenceResponse
from app.models.evidence import Evidence
from app.models.detection import Detection, DetectionType, ThreatLevel
from app.models.camera import Camera
from app.models.user import User, Role
from app.models.audit_log import create_audit_log
from pathlib import Path
from datetime import datetime
import aiofiles
import json

router = APIRouter(prefix="/evidence", tags=["evidence"])


def _check_detection_access(detection: Detection, current_user: User):
    """Verify the user has company-level access to the parent detection."""
    if detection.company_id and not check_company_access(current_user, detection.company_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to access this evidence"
        )


def _enrich_evidence(evidence: Evidence, detection: Detection, camera: Camera = None) -> EvidenceResponse:
    """Add detection context to evidence response."""
    # Build image URL from file path: ./evidence/file.jpg -> /evidence/file.jpg
    image_url = None
    if evidence.image_path:
        filename = Path(evidence.image_path).name
        image_url = f"/evidence/{filename}"

    return EvidenceResponse(
        id=evidence.id,
        detection_id=evidence.detection_id,
        image_path=evidence.image_path,
        image_url=image_url,
        video_path=evidence.video_path,
        metadata_json=evidence.metadata_json,
        created_at=evidence.created_at,
        detection_type=detection.detection_type.value if detection else None,
        threat_level=detection.threat_level.value if detection else None,
        confidence=detection.confidence if detection else None,
        camera_id=detection.camera_id if detection else None,
        camera_name=camera.name if camera else None,
        detection_timestamp=detection.frame_timestamp if detection else None,
    )


@router.post("", response_model=EvidenceResponse, status_code=status.HTTP_201_CREATED)
async def create_evidence(
    detection_id: int,
    image_file: UploadFile = File(...),
    metadata_json: str = None,
    company_id: Optional[int] = Query(None, description="Filter by company (aegis admin)"),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_any_authenticated)
):
    """Upload evidence (snapshot/image) for a detection."""
    # Verify detection exists
    detection = db.query(Detection).filter(Detection.id == detection_id).first()
    if not detection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Detection not found"
        )

    _check_detection_access(detection, current_user)

    # Save uploaded file
    from app.config import settings
    evidence_dir = Path(settings.EVIDENCE_DIR)
    file_path = evidence_dir / f"evidence_{detection_id}_{image_file.filename}"

    async with aiofiles.open(file_path, 'wb') as f:
        content = await image_file.read()
        await f.write(content)

    db_evidence = Evidence(
        detection_id=detection_id,
        image_path=str(file_path),
        metadata_json=metadata_json
    )
    db.add(db_evidence)
    db.commit()
    db.refresh(db_evidence)

    return db_evidence


@router.get("", response_model=List[EvidenceResponse])
async def list_evidence(
    request: Request,
    detection_id: Optional[int] = None,
    camera_id: Optional[int] = Query(None, description="Filter by camera"),
    detection_type: Optional[DetectionType] = Query(None, description="Filter by detection type"),
    threat_level: Optional[ThreatLevel] = Query(None, description="Filter by threat level"),
    min_confidence: Optional[float] = Query(None, ge=0.0, le=1.0, description="Minimum confidence threshold"),
    start_date: Optional[datetime] = Query(None, description="Filter from date"),
    end_date: Optional[datetime] = Query(None, description="Filter to date"),
    limit: int = Query(20, ge=1, le=200, description="Max results"),
    offset: int = Query(0, ge=0, description="Skip results"),
    company_id: Optional[int] = Query(None, description="Filter by company (aegis admin)"),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_any_authenticated)
):
    """List evidence with filters and pagination. Includes detection context."""
    company_filter = get_user_company_filter(current_user, company_id)
    if company_filter is None and current_user.role != Role.AEGIS_ADMIN:
        return []

    query = db.query(Evidence, Detection, Camera).join(
        Detection, Evidence.detection_id == Detection.id
    ).join(
        Camera, Detection.camera_id == Camera.id
    )
    if company_filter is not None:
        query = query.filter(Detection.company_id == company_filter)
    if detection_id:
        query = query.filter(Evidence.detection_id == detection_id)
    if camera_id:
        query = query.filter(Detection.camera_id == camera_id)
    if detection_type:
        query = query.filter(Detection.detection_type == detection_type)
    if threat_level:
        query = query.filter(Detection.threat_level == threat_level)
    if min_confidence is not None:
        query = query.filter(Detection.confidence >= min_confidence)
    if start_date:
        query = query.filter(Detection.frame_timestamp >= start_date)
    if end_date:
        query = query.filter(Detection.frame_timestamp <= end_date)

    results = query.order_by(Evidence.created_at.desc()).offset(offset).limit(limit).all()

    return [_enrich_evidence(ev, det, cam) for ev, det, cam in results]


def _load_evidence_with_detection(evidence_id: int, current_user: User, db: Session):
    """Single JOIN query to load evidence + detection + camera, then access-check.

    Replaces the previous pattern of 2–3 individual queries per request.
    """
    row = (
        db.query(Evidence, Detection, Camera)
        .join(Detection, Evidence.detection_id == Detection.id)
        .join(Camera, Detection.camera_id == Camera.id)
        .filter(Evidence.id == evidence_id)
        .first()
    )
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Evidence not found")
    evidence, detection, camera = row
    _check_detection_access(detection, current_user)
    return evidence, detection, camera


@router.get("/{evidence_id}", response_model=EvidenceResponse)
async def get_evidence(
    evidence_id: int,
    company_id: Optional[int] = Query(None, description="Filter by company (aegis admin)"),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_any_authenticated)
):
    """Get evidence by ID with detection context."""
    evidence, detection, camera = _load_evidence_with_detection(evidence_id, current_user, db)
    return _enrich_evidence(evidence, detection, camera)


@router.get("/{evidence_id}/image")
async def view_evidence_image(
    evidence_id: int,
    company_id: Optional[int] = Query(None, description="Filter by company (aegis admin)"),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_any_authenticated)
):
    """View evidence image inline (for displaying in frontend).

    Returns Cache-Control headers so the browser caches the image for 24 hours
    and does not re-fetch it every time the evidence tab is opened.
    """
    evidence, _det, _cam = _load_evidence_with_detection(evidence_id, current_user, db)

    file_path = Path(evidence.image_path)
    if not file_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Evidence file not found")

    return FileResponse(
        path=str(file_path),
        media_type="image/jpeg",
        headers={
            "Cache-Control": "private, max-age=86400",  # browser caches for 24 h
            "ETag": str(evidence_id),
        },
    )


@router.get("/{evidence_id}/download")
async def download_evidence(
    evidence_id: int,
    company_id: Optional[int] = Query(None, description="Filter by company (aegis admin)"),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_any_authenticated)
):
    """Download evidence image file."""
    evidence, _det, _cam = _load_evidence_with_detection(evidence_id, current_user, db)

    file_path = Path(evidence.image_path)
    if not file_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Evidence file not found")

    return FileResponse(
        path=str(file_path),
        filename=file_path.name,
        media_type="image/jpeg",
        headers={"Cache-Control": "private, max-age=86400"},
    )


@router.post("/export")
async def export_evidence(
    request: Request,
    detection_ids: List[int],
    company_id: Optional[int] = Query(None, description="Filter by company (aegis admin)"),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_any_authenticated)
):
    """Export evidence as CSV with metadata. Respects company isolation."""
    import pandas as pd
    from io import StringIO
    from fastapi.responses import StreamingResponse

    company_filter = get_user_company_filter(current_user, company_id)

    det_query = db.query(Detection).filter(Detection.id.in_(detection_ids))
    if company_filter is not None:
        det_query = det_query.filter(Detection.company_id == company_filter)
    elif current_user.role != Role.AEGIS_ADMIN:
        det_query = det_query.filter(False)

    detections = det_query.all()
    accessible_ids = [d.id for d in detections]

    # Audit log
    db.add(create_audit_log(
        request, current_user.id, "export_evidence", "evidence", None,
        {"detection_ids": accessible_ids}
    ))
    db.commit()

    evidence_list = db.query(Evidence).filter(Evidence.detection_id.in_(accessible_ids)).all()

    # Create export data
    export_data = []
    for evidence in evidence_list:
        detection = next((d for d in detections if d.id == evidence.detection_id), None)
        if detection:
            export_data.append({
                "detection_id": detection.id,
                "detection_type": detection.detection_type.value,
                "threat_level": detection.threat_level.value,
                "confidence": detection.confidence,
                "timestamp": detection.frame_timestamp.isoformat(),
                "evidence_id": evidence.id,
                "image_path": evidence.image_path,
                "metadata": evidence.metadata_json
            })

    df = pd.DataFrame(export_data)
    output = StringIO()
    df.to_csv(output, index=False)
    output.seek(0)

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=evidence_export.csv"}
    )
