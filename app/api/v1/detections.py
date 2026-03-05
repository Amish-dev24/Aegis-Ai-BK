"""
Detection endpoints for managing AI detections.
"""
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_
from app.database import get_db
from app.core.security import require_any_authenticated, require_security_officer, get_current_user, get_user_company_filter, check_company_access
from app.models.user import Role
from app.schemas.detection import DetectionCreate, DetectionResponse, DetectionFilter
from app.models.detection import Detection, DetectionType, ThreatLevel
from app.models.camera import Camera
from app.models.user import User
from app.services.detection_service import detection_service
from app.services.video_service import video_service
from app.services.email_service import email_service
from app.models.alert import Alert, AlertStatus
from app.models.evidence import Evidence
from app.models.audit_log import AuditLog
import cv2
import numpy as np
from datetime import datetime

router = APIRouter(prefix="/detections", tags=["detections"])


@router.post("", response_model=DetectionResponse, status_code=status.HTTP_201_CREATED)
async def create_detection(
    detection_data: DetectionCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_any_authenticated)
):
    """Create a detection record (typically called by detection service)."""
    # Verify camera exists
    camera = db.query(Camera).filter(Camera.id == detection_data.camera_id).first()
    if not camera:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Camera not found"
        )
    
    # Check company access to camera
    if camera.company_id and not check_company_access(current_user, camera.company_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to create detection for this camera"
        )
    
    detection_dict = detection_data.dict()
    # Set company_id from camera
    detection_dict["company_id"] = camera.company_id
    
    db_detection = Detection(**detection_dict)
    db.add(db_detection)
    db.commit()
    db.refresh(db_detection)

    # Audit log
    db.add(AuditLog(
        user_id=current_user.id,
        action="create_detection",
        resource_type="detection",
        resource_id=db_detection.id,
        details={"detection_type": db_detection.detection_type.value, "threat_level": db_detection.threat_level.value},
    ))
    db.commit()

    # Create alert + send email for medium/high/critical threats
    if db_detection.threat_level in [ThreatLevel.MEDIUM, ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
        alert = Alert(
            detection_id=db_detection.id,
            company_id=camera.company_id,
            title=f"{db_detection.detection_type.value.replace('_', ' ').title()} – {db_detection.threat_level.value.upper()}",
            message=f"Detection of {db_detection.detection_type.value} with {db_detection.confidence:.2%} confidence",
            status=AlertStatus.PENDING
        )
        db.add(alert)
        db.flush()

        # Fetch evidence snapshot if available
        evidence = db.query(Evidence).filter(Evidence.detection_id == db_detection.id).first()
        snapshot_path = evidence.image_path if evidence else None

        email_sent = await email_service.send_alert_email(
            to_emails=[current_user.email],
            subject=alert.title,
            message=alert.message or f"Alert for detection {db_detection.id}",
            snapshot_path=snapshot_path,
            metadata={
                "Detection Type": db_detection.detection_type.value,
                "Threat Level": db_detection.threat_level.value,
                "Confidence": f"{db_detection.confidence:.2%}",
                "Camera": camera.name,
                "Timestamp": db_detection.frame_timestamp.isoformat(),
            },
        )
        if email_sent:
            alert.email_sent = True
            alert.email_sent_at = datetime.utcnow()

        db.commit()

    return db_detection


@router.get("", response_model=List[DetectionResponse])
async def list_detections(
    filter_params: DetectionFilter = Depends(),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_any_authenticated)
):
    """List detections with filtering. Company users see only their company's detections."""
    query = db.query(Detection)
    
    # Filter by company (unless Aegis AI admin)
    company_filter = get_user_company_filter(current_user)
    if company_filter is not None:
        query = query.filter(Detection.company_id == company_filter)
    
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
    
    detections = query.order_by(Detection.detected_at.desc()).offset(filter_params.offset).limit(filter_params.limit).all()
    return detections


@router.get("/{detection_id}", response_model=DetectionResponse)
async def get_detection(
    detection_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_any_authenticated)
):
    """Get detection by ID. Company users can only access their company's detections."""
    detection = db.query(Detection).filter(Detection.id == detection_id).first()
    if not detection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Detection not found"
        )
    
    # Check company access
    if detection.company_id and not check_company_access(current_user, detection.company_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to access this detection"
        )
    
    return detection


@router.post("/process-video")
async def process_video(
    camera_id: int,
    video_file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_security_officer)
):
    """Process a video file through all detection modules."""
    # Verify camera exists
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Camera not found"
        )

    # Check company access to camera
    if camera.company_id and not check_company_access(current_user, camera.company_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to process video for this camera"
        )

    # Save uploaded file temporarily
    from pathlib import Path
    import aiofiles
    import os
    upload_path = Path(f"./uploads/{video_file.filename}")
    async with aiofiles.open(upload_path, 'wb') as f:
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

        for frame, timestamp in video_service.read_video_file(str(upload_path)):
            frame_index += 1
            if frame_index % process_every_n != 0:
                continue
            all_raw_detections: list[tuple[DetectionType, dict]] = []

            # --- 1. Weapon detection (YOLOv8) ---
            if "weapon" in enabled_modules:
                for det in detection_service.detect_weapons(frame, timestamp):
                    all_raw_detections.append((DetectionType.WEAPON, det))

            # --- 2. Violence / aggression detection (MediaPipe + motion) ---
            if "violence" in enabled_modules:
                violence = detection_service.detect_violence(frame, previous_frames, timestamp)
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
                if crowd["count"] > 0:
                    all_raw_detections.append((DetectionType.CROWD_DENSITY, crowd))

            # Persist detections (deduplicated — skip if same type detected within 2 sec)
            for det_type, det in all_raw_detections:
                elapsed = (timestamp - last_detection_time.get(det_type, datetime.min)).total_seconds()
                if elapsed < dedup_interval:
                    continue
                last_detection_time[det_type] = timestamp

                confidence = det.get("confidence", 0.0)
                module_settings = enabled_modules.get(det_type.value, {})

                # Skip if below company's custom min_confidence
                min_conf = module_settings.get("min_confidence")
                if min_conf and confidence < min_conf:
                    continue

                threat_level = detection_service.classify_threat_level(det_type, confidence, det, module_settings)
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
                snapshot_path = video_service.save_snapshot(
                    frame, db_detection.id, prefix=det_type.value,
                    bbox=bbox, label=label,
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
                    alert.email_sent_at = datetime.utcnow()
                    alerts_created += 1

                detection_results.append({
                    "id": db_detection.id,
                    "detection_type": det_type.value,
                    "threat_level": threat_level.value,
                    "confidence": round(confidence, 4),
                    "class": det.get("class", det_type.value),
                    "evidence_id": db_evidence.id if db_evidence else None,
                    "timestamp": timestamp.isoformat(),
                })
                detections_created += 1

            # Keep a sliding window of previous frames for temporal analysis
            previous_frames.append(frame)
            if len(previous_frames) > 10:
                previous_frames.pop(0)

        # Audit log for video processing
        db.add(AuditLog(
            user_id=current_user.id,
            action="process_video",
            resource_type="camera",
            resource_id=camera_id,
            details={"filename": video_file.filename, "detections": detections_created, "alerts": alerts_created},
        ))
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
    camera_id: int,
    image_file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_security_officer)
):
    """Process a single image through all detection modules."""
    # Verify camera exists
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Camera not found"
        )

    if camera.company_id and not check_company_access(current_user, camera.company_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to process image for this camera"
        )

    # Read image into OpenCV frame
    content = await image_file.read()
    nparr = np.frombuffer(content, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image file"
        )

    timestamp = datetime.utcnow()
    detections_created = 0
    alerts_created = 0
    detection_results = []
    all_raw_detections: list[tuple] = []

    # Get enabled modules (respects global + company settings)
    enabled_modules = detection_service.get_enabled_modules(db, camera.company_id)

    # Run all detection modules on the single frame
    if "weapon" in enabled_modules:
        for det in detection_service.detect_weapons(frame, timestamp):
            all_raw_detections.append((DetectionType.WEAPON, det))

    if "violence" in enabled_modules:
        violence = detection_service.detect_violence(frame, [], timestamp)
        if violence["is_violent"]:
            all_raw_detections.append((DetectionType.VIOLENCE, violence))

    if "mask_face" in enabled_modules:
        for det in detection_service.detect_mask_face(frame, timestamp):
            all_raw_detections.append((DetectionType.MASK_FACE, det))

    if "crowd_density" in enabled_modules:
        crowd = detection_service.calculate_crowd_density(frame, timestamp)
        if crowd["count"] > 0:
            all_raw_detections.append((DetectionType.CROWD_DENSITY, crowd))

    # Save detections
    for det_type, det in all_raw_detections:
        confidence = det.get("confidence", 0.0)
        module_settings = enabled_modules.get(det_type.value, {})

        # Skip if below company's custom min_confidence
        min_conf = module_settings.get("min_confidence")
        if min_conf and confidence < min_conf:
            continue

        threat_level = detection_service.classify_threat_level(det_type, confidence, det, module_settings)
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
        snapshot_path = video_service.save_snapshot(
            frame, db_detection.id, prefix=det_type.value,
            bbox=bbox, label=label,
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
            alert.email_sent_at = datetime.utcnow()
            alerts_created += 1

        detection_results.append({
            "id": db_detection.id,
            "detection_type": det_type.value,
            "threat_level": threat_level.value,
            "confidence": round(confidence, 4),
            "class": det.get("class", det_type.value),
            "evidence_id": db_evidence.id if db_evidence else None,
            "timestamp": timestamp.isoformat(),
        })
        detections_created += 1

    db.add(AuditLog(
        user_id=current_user.id,
        action="process_image",
        resource_type="camera",
        resource_id=camera_id,
        details={"filename": image_file.filename, "detections": detections_created, "alerts": alerts_created},
    ))
    db.commit()

    return {
        "message": f"Processed image, created {detections_created} detections and {alerts_created} alerts",
        "total_detections": detections_created,
        "total_alerts": alerts_created,
        "detections": detection_results,
    }


@router.get("/stats/summary")
async def get_detection_stats(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_any_authenticated),
    days: int = 7
):
    """Get detection statistics summary. Company users see only their company's stats."""
    from datetime import timedelta
    start_date = datetime.utcnow() - timedelta(days=days)
    
    # Filter by company (unless Aegis AI admin)
    company_filter = get_user_company_filter(current_user)
    
    base_query = db.query(Detection).filter(Detection.detected_at >= start_date)
    if company_filter is not None:
        base_query = base_query.filter(Detection.company_id == company_filter)
    
    total = base_query.count()
    by_type = {}
    by_threat = {}
    
    for det_type in DetectionType:
        query = base_query.filter(Detection.detection_type == det_type)
        count = query.count()
        by_type[det_type.value] = count
    
    for threat in ThreatLevel:
        query = base_query.filter(Detection.threat_level == threat)
        count = query.count()
        by_threat[threat.value] = count
    
    return {
        "total_detections": total,
        "by_type": by_type,
        "by_threat_level": by_threat,
        "period_days": days
    }

