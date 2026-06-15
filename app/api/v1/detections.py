"""
Detection endpoints for managing AI detections.
"""

from datetime import datetime
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile, status
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.config import settings
from app.core.security import (
    check_company_access,
    get_user_company_filter,
    require_any_authenticated,
    require_security_officer,
)
from app.database import get_db
from app.models.alert import Alert, AlertStatus
from app.models.audit_log import create_audit_log
from app.models.camera import Camera
from app.models.detection import Detection, DetectionType, ThreatLevel
from app.models.detection_settings import CompanyDetectionSettings, GlobalModuleSettings
from app.models.evidence import Evidence
from app.models.user import User
from app.schemas.detection import DetectionCreate, DetectionFilter, DetectionResponse
from app.services.detection_service import detection_service
from app.services.email_service import email_service
from app.services.video_service import video_service
from app.services.violence_model import ViolenceClassifier
from app.services.zone_notification_service import get_alert_emails_for_camera

router = APIRouter(prefix="/detections", tags=["detections"])


@router.post("", response_model=DetectionResponse, status_code=status.HTTP_201_CREATED)
async def create_detection(
    request: Request,
    detection_data: DetectionCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_any_authenticated),
):
    """Create a detection record (typically called by detection service)."""
    # Verify camera exists
    camera = db.query(Camera).filter(Camera.id == detection_data.camera_id).first()
    if not camera:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Camera not found")

    # Check company access to camera
    if camera.company_id and not check_company_access(current_user, camera.company_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to create detection for this camera",
        )

    detection_dict = detection_data.dict()
    # Set company_id from camera
    detection_dict["company_id"] = camera.company_id

    db_detection = Detection(**detection_dict)
    db.add(db_detection)
    db.commit()
    db.refresh(db_detection)

    # Audit log
    db.add(
        create_audit_log(
            request,
            current_user.id,
            "create_detection",
            "detection",
            db_detection.id,
            {
                "detection_type": db_detection.detection_type.value,
                "threat_level": db_detection.threat_level.value,
            },
        )
    )
    db.commit()

    # Create alert + send email for medium/high/critical threats
    if db_detection.threat_level in [ThreatLevel.MEDIUM, ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
        alert = Alert(
            detection_id=db_detection.id,
            company_id=camera.company_id,
            title=f"{db_detection.detection_type.value.replace('_', ' ').title()} – {db_detection.threat_level.value.upper()}",
            message=f"Detection of {db_detection.detection_type.value} with {db_detection.confidence:.2%} confidence",
            status=AlertStatus.PENDING,
        )
        db.add(alert)
        db.flush()

        # Fetch evidence snapshot if available
        evidence = db.query(Evidence).filter(Evidence.detection_id == db_detection.id).first()
        snapshot_path = evidence.image_path if evidence else None

        # Resolve all recipients: current user + company admins + zone officer
        alert_emails = get_alert_emails_for_camera(db, camera, current_user.email)

        email_sent = await email_service.send_alert_email(
            to_emails=alert_emails,
            subject=alert.title,
            message=alert.message or f"Alert for detection {db_detection.id}",
            snapshot_path=snapshot_path,
            metadata={
                "Detection Type": db_detection.detection_type.value,
                "Threat Level": db_detection.threat_level.value,
                "Confidence": f"{db_detection.confidence:.2%}",
                "Camera": camera.name,
                "Zone": camera.zone or "—",
                "Timestamp": db_detection.frame_timestamp.isoformat(),
            },
        )
        if email_sent:
            alert.email_sent = True
            alert.email_sent_to = ", ".join(alert_emails)
            alert.email_sent_at = datetime.utcnow()

        db.commit()

    return db_detection


@router.get("")
async def list_detections(
    request: Request,
    filter_params: DetectionFilter = Depends(),
    company_id: Optional[int] = Query(None, description="Filter by company (aegis_admin only)"),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_any_authenticated),
):
    """List detections with filtering. Company users see only their company's detections."""
    company_filter = get_user_company_filter(current_user, company_id)
    if company_filter is None:
        return []

    query = db.query(Detection).filter(Detection.company_id == company_filter)

    if filter_params.camera_id:
        query = query.filter(Detection.camera_id == filter_params.camera_id)
    if filter_params.detection_type:
        query = query.filter(Detection.detection_type == filter_params.detection_type)
    if filter_params.threat_level:
        query = query.filter(Detection.threat_level == filter_params.threat_level)
    if filter_params.start_date:
        query = query.filter(Detection.frame_timestamp >= filter_params.start_date)
    if filter_params.end_date:
        query = query.filter(Detection.frame_timestamp <= filter_params.end_date)
    if filter_params.min_confidence:
        query = query.filter(Detection.confidence >= filter_params.min_confidence)

    detections = (
        query.order_by(Detection.detected_at.desc())
        .offset(filter_params.offset)
        .limit(filter_params.limit)
        .all()
    )

    # Batch-load evidence in one query instead of one query per detection (N+1 fix)
    det_ids = [d.id for d in detections]
    evidence_map: dict[int, int] = {}
    if det_ids:
        evidence_map = {
            e.detection_id: e.id
            for e in db.query(Evidence.detection_id, Evidence.id)
            .filter(Evidence.detection_id.in_(det_ids))
            .all()
        }

    result = []
    for det in detections:
        result.append(
            {
                "id": det.id,
                "camera_id": det.camera_id,
                "company_id": det.company_id,
                "detection_type": det.detection_type.value if det.detection_type else None,
                "threat_level": det.threat_level.value if det.threat_level else None,
                "confidence": det.confidence,
                "frame_timestamp": det.frame_timestamp,
                "detected_at": det.detected_at,
                "bbox_x": det.bbox_x,
                "bbox_y": det.bbox_y,
                "bbox_width": det.bbox_width,
                "bbox_height": det.bbox_height,
                "detection_metadata": det.detection_metadata,
                "evidence_id": evidence_map.get(det.id),
            }
        )
    return result


@router.get("/{detection_id}", response_model=DetectionResponse)
async def get_detection(
    detection_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_any_authenticated),
):
    """Get detection by ID. Company users can only access their company's detections."""
    detection = db.query(Detection).filter(Detection.id == detection_id).first()
    if not detection:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Detection not found")

    # Check company access
    if detection.company_id and not check_company_access(current_user, detection.company_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to access this detection",
        )

    return detection


@router.post("/process-video", deprecated=True)
async def process_video(
    request: Request,
    camera_id: int,
    video_file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_security_officer),
):
    """Process a video file through all detection modules.
    DEPRECATED: Use POST /video/process instead for non-blocking processing with progress tracking.
    """
    # Verify camera exists
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Camera not found")

    # Check company access to camera
    if camera.company_id and not check_company_access(current_user, camera.company_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to process video for this camera",
        )

    # Save uploaded file temporarily
    import os
    from pathlib import Path

    import aiofiles

    upload_path = Path(f"./uploads/{video_file.filename}")
    async with aiofiles.open(upload_path, "wb") as f:
        content = await video_file.read()
        await f.write(content)

    try:
        detections_created = 0
        alerts_created = 0
        detection_results = []  # Store results to return
        previous_frames: list = []
        object_history: dict = {}
        frame_index = 0
        process_every_n = 10  # Process every 10th frame (~3 fps for 30fps video)
        last_detection_time: dict = {}  # Track last detection time per type to avoid duplicates
        dedup_interval = 2.0  # Minimum seconds between same detection type

        # Get enabled modules (respects global + company settings)
        enabled_modules = detection_service.get_enabled_modules(db, camera.company_id)
        prev_buf_max = (
            max(0, ViolenceClassifier.NUM_FRAMES - 1) if "violence" in enabled_modules else 10
        )

        for frame, timestamp in video_service.read_video_file(str(upload_path)):
            frame_index += 1
            if frame_index % process_every_n != 0:
                continue
            all_raw_detections: list[tuple[DetectionType, dict]] = []

            # --- 1. Weapon detection (YOLOv8) ---
            if "weapon" in enabled_modules:
                w_minc = enabled_modules["weapon"].get("min_confidence")
                for det in detection_service.detect_weapons(
                    frame, timestamp, min_confidence=w_minc
                ):
                    all_raw_detections.append((DetectionType.WEAPON, det))

            # --- 2. Violence / aggression detection (MediaPipe + motion) ---
            if "violence" in enabled_modules:
                v_minc = enabled_modules["violence"].get("min_confidence")
                violence = detection_service.detect_violence(
                    frame, previous_frames, timestamp, prob_threshold=v_minc
                )
                if violence["is_violent"]:
                    all_raw_detections.append((DetectionType.VIOLENCE, violence))

            # --- 3. Abandoned object detection (MOG2) ---
            if "abandoned_object" in enabled_modules:
                for det in detection_service.detect_abandoned_object(
                    frame, None, timestamp, object_history
                ):
                    all_raw_detections.append((DetectionType.ABANDONED_OBJECT, det))

            # --- 4. Mask / face obfuscation detection ---
            if "mask_face" in enabled_modules:
                for det in detection_service.detect_mask_face(frame, timestamp):
                    all_raw_detections.append((DetectionType.MASK_FACE, det))

            # --- 5. Crowd density monitoring ---
            if "crowd_density" in enabled_modules:
                crowd = detection_service.calculate_crowd_density(frame, timestamp)
                if int(crowd.get("count", 0) or 0) > 0:
                    all_raw_detections.append((DetectionType.CROWD_DENSITY, crowd))

            # Persist detections (deduplicated — skip if same type detected within 2 sec)
            for det_type, det in all_raw_detections:
                elapsed = (
                    timestamp - last_detection_time.get(det_type, datetime.min)
                ).total_seconds()
                if elapsed < dedup_interval:
                    continue
                last_detection_time[det_type] = timestamp

                confidence = det.get("confidence", 0.0)
                module_settings = enabled_modules.get(det_type.value, {})

                # Skip if below company's custom min_confidence
                min_conf = module_settings.get("min_confidence")
                if (
                    det_type != DetectionType.CROWD_DENSITY
                    and min_conf is not None
                    and confidence < min_conf
                ):
                    continue

                threat_level = detection_service.classify_threat_level(
                    det_type, confidence, det, module_settings
                )
                bbox = det.get("bbox", [0, 0, 0, 0])

                db_detection = Detection(
                    camera_id=camera_id,
                    company_id=camera.company_id,
                    detection_type=det_type,
                    threat_level=threat_level,
                    confidence=confidence,
                    frame_timestamp=timestamp,
                    bbox_x=bbox[0] if len(bbox) > 0 else None,
                    bbox_y=bbox[1] if len(bbox) > 1 else None,
                    bbox_width=bbox[2] if len(bbox) > 2 else None,
                    bbox_height=bbox[3] if len(bbox) > 3 else None,
                    detection_metadata=det,
                )
                db.add(db_detection)
                db.flush()  # get db_detection.id

                # Save evidence snapshot with bounding box drawn
                label = f"{det.get('class', det_type.value)} {confidence:.0%}"
                face_boxes = (
                    detection_service.collect_face_bboxes_normalized(frame)
                    if getattr(settings, "PRIVACY_BLUR_NON_SUBJECT_FACES", True)
                    else []
                )
                snap_frame = video_service.privacy_blur_for_snapshot(
                    frame, det_type.value, bbox, face_boxes
                )
                snapshot_path = video_service.save_snapshot(
                    snap_frame,
                    db_detection.id,
                    prefix=det_type.value,
                    bbox=bbox,
                    label=label,
                )
                db_evidence = Evidence(
                    detection_id=db_detection.id,
                    image_path=snapshot_path,
                    metadata_json=str(det),
                )
                db.add(db_evidence)
                db.flush()  # get db_evidence.id

                # Auto-create alert + send email for MEDIUM / HIGH / CRITICAL threats
                if threat_level in (ThreatLevel.MEDIUM, ThreatLevel.HIGH, ThreatLevel.CRITICAL):
                    alert = Alert(
                        detection_id=db_detection.id,
                        company_id=camera.company_id,
                        title=f"{det_type.value.replace('_', ' ').title()} – {threat_level.value.upper()}",
                        message=f"{det_type.value} detected with {confidence:.0%} confidence",
                        status=AlertStatus.PENDING,
                    )
                    db.add(alert)
                    db.flush()

                    await email_service.send_alert_email(
                        to_emails=[current_user.email],
                        subject=alert.title,
                        message=alert.message,
                        snapshot_path=snapshot_path,
                        metadata={
                            "Detection Type": det_type.value,
                            "Threat Level": threat_level.value,
                            "Confidence": f"{confidence:.2%}",
                            "Camera": camera.name,
                            "Timestamp": timestamp.isoformat(),
                        },
                    )
                    alert.email_sent = True
                    alert.email_sent_to = current_user.email
                    alert.email_sent_at = datetime.utcnow()
                    alerts_created += 1

                detection_results.append(
                    {
                        "id": db_detection.id,
                        "detection_type": det_type.value,
                        "threat_level": threat_level.value,
                        "confidence": round(confidence, 4),
                        "class": det.get("class", det_type.value),
                        "evidence_id": db_evidence.id if db_evidence else None,
                        "timestamp": timestamp.isoformat(),
                    }
                )
                detections_created += 1

            # Sliding window: Conv3D violence needs 15 prior frames (+ current passed above).
            previous_frames.append(frame)
            if len(previous_frames) > prev_buf_max:
                previous_frames.pop(0)

        # Audit log for video processing
        db.add(
            create_audit_log(
                request,
                current_user.id,
                "process_video",
                "camera",
                camera_id,
                {
                    "filename": video_file.filename,
                    "detections": detections_created,
                    "alerts": alerts_created,
                },
            )
        )
        db.commit()

        return {
            "message": f"Processed video, created {detections_created} detections and {alerts_created} alerts",
            "total_detections": detections_created,
            "total_alerts": alerts_created,
            "detections": detection_results,
        }
    finally:
        # Clean up temporary uploaded file
        if upload_path.exists():
            os.remove(upload_path)


@router.post("/process-image")
async def process_image(
    request: Request,
    camera_id: int,
    image_file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_security_officer),
):
    """Process a single image through all detection modules."""
    # Verify camera exists
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Camera not found")

    if camera.company_id and not check_company_access(current_user, camera.company_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to process image for this camera",
        )

    # Read image into OpenCV frame
    content = await image_file.read()
    nparr = np.frombuffer(content, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid image file")

    timestamp = datetime.utcnow()
    detections_created = 0
    alerts_created = 0
    detection_results = []
    all_raw_detections: list[tuple] = []

    # Get enabled modules (respects global + company settings)
    enabled_modules = detection_service.get_enabled_modules(db, camera.company_id)

    # Run all detection modules on the single frame
    if "weapon" in enabled_modules:
        w_minc = enabled_modules["weapon"].get("min_confidence")
        for det in detection_service.detect_weapons(frame, timestamp, min_confidence=w_minc):
            all_raw_detections.append((DetectionType.WEAPON, det))

    if "violence" in enabled_modules:
        v_minc = enabled_modules.get("violence", {}).get("min_confidence")
        violence = detection_service.detect_violence(frame, [], timestamp, prob_threshold=v_minc)
        if violence["is_violent"]:
            all_raw_detections.append((DetectionType.VIOLENCE, violence))

    if "mask_face" in enabled_modules:
        for det in detection_service.detect_mask_face(frame, timestamp):
            all_raw_detections.append((DetectionType.MASK_FACE, det))

    if "crowd_density" in enabled_modules:
        crowd = detection_service.calculate_crowd_density(frame, timestamp)
        if int(crowd.get("count", 0) or 0) > 0:
            all_raw_detections.append((DetectionType.CROWD_DENSITY, crowd))

    # Save detections
    for det_type, det in all_raw_detections:
        confidence = det.get("confidence", 0.0)
        module_settings = enabled_modules.get(det_type.value, {})

        # Skip if below company's custom min_confidence
        min_conf = module_settings.get("min_confidence")
        if (
            det_type != DetectionType.CROWD_DENSITY
            and min_conf is not None
            and confidence < min_conf
        ):
            continue

        threat_level = detection_service.classify_threat_level(
            det_type, confidence, det, module_settings
        )
        bbox = det.get("bbox", [0, 0, 0, 0])

        db_detection = Detection(
            camera_id=camera_id,
            company_id=camera.company_id,
            detection_type=det_type,
            threat_level=threat_level,
            confidence=confidence,
            frame_timestamp=timestamp,
            bbox_x=bbox[0] if len(bbox) > 0 else None,
            bbox_y=bbox[1] if len(bbox) > 1 else None,
            bbox_width=bbox[2] if len(bbox) > 2 else None,
            bbox_height=bbox[3] if len(bbox) > 3 else None,
            detection_metadata=det,
        )
        db.add(db_detection)
        db.flush()

        label = f"{det.get('class', det_type.value)} {confidence:.0%}"
        face_boxes = (
            detection_service.collect_face_bboxes_normalized(frame)
            if getattr(settings, "PRIVACY_BLUR_NON_SUBJECT_FACES", True)
            else []
        )
        snap_frame = video_service.privacy_blur_for_snapshot(
            frame, det_type.value, bbox, face_boxes
        )
        snapshot_path = video_service.save_snapshot(
            snap_frame,
            db_detection.id,
            prefix=det_type.value,
            bbox=bbox,
            label=label,
        )
        db_evidence = Evidence(
            detection_id=db_detection.id,
            image_path=snapshot_path,
            metadata_json=str(det),
        )
        db.add(db_evidence)
        db.flush()  # get db_evidence.id

        if threat_level in (ThreatLevel.MEDIUM, ThreatLevel.HIGH, ThreatLevel.CRITICAL):
            alert = Alert(
                detection_id=db_detection.id,
                company_id=camera.company_id,
                title=f"{det_type.value.replace('_', ' ').title()} – {threat_level.value.upper()}",
                message=f"{det_type.value} detected with {confidence:.0%} confidence",
                status=AlertStatus.PENDING,
            )
            db.add(alert)
            db.flush()
            await email_service.send_alert_email(
                to_emails=[current_user.email],
                subject=alert.title,
                message=alert.message,
                snapshot_path=snapshot_path,
                metadata={
                    "Detection Type": det_type.value,
                    "Threat Level": threat_level.value,
                    "Confidence": f"{confidence:.2%}",
                    "Camera": camera.name,
                    "Timestamp": timestamp.isoformat(),
                },
            )
            alert.email_sent = True
            alert.email_sent_to = current_user.email
            alert.email_sent_at = datetime.utcnow()
            alerts_created += 1

        detection_results.append(
            {
                "id": db_detection.id,
                "detection_type": det_type.value,
                "threat_level": threat_level.value,
                "confidence": round(confidence, 4),
                "class": det.get("class", det_type.value),
                "evidence_id": db_evidence.id if db_evidence else None,
                "timestamp": timestamp.isoformat(),
            }
        )
        detections_created += 1

    db.add(
        create_audit_log(
            request,
            current_user.id,
            "process_image",
            "camera",
            camera_id,
            {
                "filename": image_file.filename,
                "detections": detections_created,
                "alerts": alerts_created,
            },
        )
    )
    db.commit()

    return {
        "message": f"Processed image, created {detections_created} detections and {alerts_created} alerts",
        "total_detections": detections_created,
        "total_alerts": alerts_created,
        "detections": detection_results,
    }


@router.get("/stats/summary")
async def get_detection_stats(
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_any_authenticated),
    days: int = 7,
    company_id: Optional[int] = Query(None, description="Filter by company (aegis_admin only)"),
):
    """Dashboard summary statistics.

    Returns detection counts, evidence total, active cameras, critical threats,
    and enabled AI module count — all in a single request so the dashboard
    does not need to call multiple list endpoints and guess totals from page sizes.

    company_id behaviour:
    - User with ``company_id`` → stats for that company only
    - Unassigned user → empty summary
    """
    from datetime import timedelta

    company_filter = get_user_company_filter(current_user, company_id)
    if company_filter is None:
        empty = {t.value: 0 for t in DetectionType}
        return {
            "total_detections": 0,
            "by_type": empty,
            "by_threat_level": {t.value: 0 for t in ThreatLevel},
            "period_days": days,
            "critical_threats": 0,
            "evidence_count": 0,
            "active_cameras": 0,
            "ai_modules": 0,
        }

    start_date = datetime.utcnow() - timedelta(days=days)

    # --- Detection counts (2 GROUP BY queries) ---
    det_base = db.query(Detection).filter(Detection.detected_at >= start_date)
    det_base = det_base.filter(Detection.company_id == company_filter)

    by_type = {t.value: 0 for t in DetectionType}
    for det_type_val, count in (
        det_base.with_entities(Detection.detection_type, func.count(Detection.id))
        .group_by(Detection.detection_type)
        .all()
    ):
        by_type[det_type_val.value] = count

    by_threat = {t.value: 0 for t in ThreatLevel}
    for threat_val, count in (
        det_base.with_entities(Detection.threat_level, func.count(Detection.id))
        .group_by(Detection.threat_level)
        .all()
    ):
        by_threat[threat_val.value] = count

    total_detections = sum(by_type.values())
    critical_threats = by_threat.get(ThreatLevel.CRITICAL.value, 0)

    # --- Evidence total count (real total, not page size) ---
    ev_query = db.query(func.count(Evidence.id)).join(
        Detection, Evidence.detection_id == Detection.id
    )
    ev_query = ev_query.filter(Detection.company_id == company_filter)
    evidence_count = ev_query.scalar() or 0

    # --- Active cameras count ---
    cam_query = db.query(func.count(Camera.id)).filter(Camera.is_active.is_(True))
    cam_query = cam_query.filter(Camera.company_id == company_filter)
    active_cameras = cam_query.scalar() or 0

    # --- AI modules count (this company only) ---
    company_enabled = (
        db.query(func.count(CompanyDetectionSettings.id))
        .filter(
            CompanyDetectionSettings.company_id == company_filter,
            CompanyDetectionSettings.is_enabled.is_(True),
        )
        .scalar()
        or 0
    )

    overridden_modules = [
        r.module_name
        for r in db.query(CompanyDetectionSettings.module_name)
        .filter(CompanyDetectionSettings.company_id == company_filter)
        .all()
    ]
    global_fallback_q = db.query(func.count(GlobalModuleSettings.id)).filter(
        GlobalModuleSettings.is_enabled.is_(True)
    )
    if overridden_modules:
        global_fallback_q = global_fallback_q.filter(
            GlobalModuleSettings.module_name.notin_(overridden_modules)
        )
    global_fallback = global_fallback_q.scalar() or 0
    ai_modules = company_enabled + global_fallback

    return {
        "total_detections": total_detections,
        "by_type": by_type,
        "by_threat_level": by_threat,
        "period_days": days,
        "critical_threats": critical_threats,
        "evidence_count": evidence_count,
        "active_cameras": active_cameras,
        "ai_modules": ai_modules,
    }
