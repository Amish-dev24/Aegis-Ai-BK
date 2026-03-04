"""
Evidence management endpoints.
All endpoints respect multi-tenant isolation via company access checks.
"""
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status, File, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from app.database import get_db
from app.core.security import require_any_authenticated, get_current_user, check_company_access, get_user_company_filter
from app.schemas.evidence import EvidenceCreate, EvidenceResponse
from app.models.evidence import Evidence
from app.models.detection import Detection
from app.models.user import User
from app.models.audit_log import AuditLog
from pathlib import Path
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


@router.post("", response_model=EvidenceResponse, status_code=status.HTTP_201_CREATED)
async def create_evidence(
    detection_id: int,
    image_file: UploadFile = File(...),
    metadata_json: str = None,
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
    detection_id: int = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_any_authenticated)
):
    """List evidence records. Company users see only their company's evidence."""
    company_filter = get_user_company_filter(current_user)

    query = db.query(Evidence).join(Detection, Evidence.detection_id == Detection.id)

    if company_filter is not None:
        query = query.filter(Detection.company_id == company_filter)

    if detection_id:
        query = query.filter(Evidence.detection_id == detection_id)

    evidence_list = query.all()
    return evidence_list


@router.get("/{evidence_id}", response_model=EvidenceResponse)
async def get_evidence(
    evidence_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_any_authenticated)
):
    """Get evidence by ID."""
    evidence = db.query(Evidence).filter(Evidence.id == evidence_id).first()
    if not evidence:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Evidence not found"
        )

    detection = db.query(Detection).filter(Detection.id == evidence.detection_id).first()
    if detection:
        _check_detection_access(detection, current_user)

    return evidence


@router.get("/{evidence_id}/download")
async def download_evidence(
    evidence_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_any_authenticated)
):
    """Download evidence image file."""
    evidence = db.query(Evidence).filter(Evidence.id == evidence_id).first()
    if not evidence:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Evidence not found"
        )

    detection = db.query(Detection).filter(Detection.id == evidence.detection_id).first()
    if detection:
        _check_detection_access(detection, current_user)

    file_path = Path(evidence.image_path)
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Evidence file not found"
        )

    return FileResponse(
        path=str(file_path),
        filename=file_path.name,
        media_type="image/jpeg"
    )


@router.post("/export")
async def export_evidence(
    detection_ids: List[int],
    db: Session = Depends(get_db),
    current_user: User = Depends(require_any_authenticated)
):
    """Export evidence as CSV with metadata. Respects company isolation."""
    import pandas as pd
    from io import StringIO
    from fastapi.responses import StreamingResponse

    company_filter = get_user_company_filter(current_user)

    det_query = db.query(Detection).filter(Detection.id.in_(detection_ids))
    if company_filter is not None:
        det_query = det_query.filter(Detection.company_id == company_filter)

    detections = det_query.all()
    accessible_ids = [d.id for d in detections]

    # Audit log
    db.add(AuditLog(
        user_id=current_user.id,
        action="export_evidence",
        resource_type="evidence",
        details={"detection_ids": accessible_ids},
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
