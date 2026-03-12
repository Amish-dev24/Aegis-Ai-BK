"""
Video processing endpoints with progress tracking and annotated video output.

The heavy video/AI work runs in a thread pool so it never blocks
the async event loop — other API endpoints stay responsive.
"""
import asyncio
import uuid
import os
import logging
from typing import Optional
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Request, Query
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.orm import Session

from app.database import get_db, SessionLocal
from app.config import settings
from app.core.security import require_security_officer, require_any_authenticated, check_company_access, get_user_company_filter
from app.models.user import User
from app.models.camera import Camera
from app.models.detection import Detection, DetectionType, ThreatLevel
from app.models.alert import Alert, AlertStatus
from app.models.evidence import Evidence
from app.models.audit_log import create_audit_log
from app.services.detection_service import detection_service
from app.services.video_service import video_service
from app.services.email_service import email_service

import cv2

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/video", tags=["video-processing"])

# Thread pool for CPU-heavy video/AI work (keeps event loop free)
_thread_pool = ThreadPoolExecutor(max_workers=2)

# ---------------------------------------------------------------------------
# In-memory job store
# ---------------------------------------------------------------------------
_jobs: dict[str, dict] = {}


def _get_job(job_id: str) -> dict:
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    return job


# ---------------------------------------------------------------------------
# POST /video/process  —  start background processing, return job_id
# ---------------------------------------------------------------------------
@router.post("/process")
async def start_video_processing(
    request: Request,
    camera_id: int,
    video_file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_security_officer),
):
    """
    Upload a video for AI processing.
    Returns a job_id immediately — poll /video/jobs/{job_id} or
    stream /video/jobs/{job_id}/progress for real-time updates.
    """
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Camera not found")
    if camera.company_id and not check_company_access(current_user, camera.company_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions")

    # Save uploaded file
    import aiofiles
    upload_path = Path(settings.UPLOAD_DIR) / f"{uuid.uuid4().hex}_{video_file.filename}"
    async with aiofiles.open(upload_path, "wb") as f:
        content = await video_file.read()
        await f.write(content)

    # Get video info
    info = video_service.get_video_info(str(upload_path))

    # Create job
    job_id = uuid.uuid4().hex
    output_filename = f"processed_{job_id}.mp4"
    output_path = Path(settings.PROCESSED_VIDEO_DIR) / output_filename

    _jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0,
        "current_frame": 0,
        "total_frames": info["total_frames"],
        "fps": info["fps"],
        "duration_seconds": info["duration_seconds"],
        "total_detections": 0,
        "total_alerts": 0,
        "detections": [],
        "output_video": str(output_path),
        "output_video_url": None,
        "started_at": datetime.utcnow().isoformat(),
        "completed_at": None,
        "error": None,
        "camera_id": camera_id,
        "company_id": camera.company_id,
        "user_id": current_user.id,
        "user_email": current_user.email,
        "camera_name": camera.name,
        "filename": video_file.filename,
        "upload_path": str(upload_path),
    }

    # Run heavy processing in thread pool — does NOT block the event loop
    loop = asyncio.get_event_loop()
    loop.run_in_executor(_thread_pool, _process_video_sync, job_id)

    return {
        "job_id": job_id,
        "status": "queued",
        "total_frames": info["total_frames"],
        "duration_seconds": round(info["duration_seconds"], 2),
        "message": "Video processing started. Use /video/jobs/{job_id} to check status or /video/jobs/{job_id}/progress for real-time updates.",
    }


# ---------------------------------------------------------------------------
# Synchronous processing — runs in thread pool, never touches event loop
# ---------------------------------------------------------------------------
def _process_video_sync(job_id: str):
    """Run detection + annotation. Runs in a worker thread."""
    job = _jobs[job_id]
    db = SessionLocal()

    try:
        job["status"] = "processing"
        upload_path = job["upload_path"]
        output_path = job["output_video"]

        info = video_service.get_video_info(upload_path)
        fps = info["fps"]
        width = info["width"]
        height = info["height"]
        total_frames = info["total_frames"]

        cap = cv2.VideoCapture(upload_path)
        writer = video_service.create_video_writer(output_path, fps, width, height)

        process_every_n = 10
        previous_frames: list = []
        object_history: dict = {}
        last_detection_time: dict = {}
        dedup_interval = 2.0
        frame_index = 0

        camera_id = job["camera_id"]
        company_id = job["company_id"]

        enabled_modules = detection_service.get_enabled_modules(db, company_id)
        active_detections: list = []
        video_start = datetime.now()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_index += 1
                elapsed = timedelta(seconds=frame_index / fps)
                timestamp = video_start + elapsed

                # Run AI on every Nth frame
                if frame_index % process_every_n == 0:
                    all_raw: list[tuple] = []
                    active_detections = []

                    if "weapon" in enabled_modules:
                        for det in detection_service.detect_weapons(frame, timestamp):
                            all_raw.append((DetectionType.WEAPON, det))

                    if "violence" in enabled_modules:
                        violence = detection_service.detect_violence(frame, previous_frames, timestamp)
                        if violence["is_violent"]:
                            all_raw.append((DetectionType.VIOLENCE, violence))

                    if "abandoned_object" in enabled_modules:
                        for det in detection_service.detect_abandoned_object(frame, None, timestamp, object_history):
                            all_raw.append((DetectionType.ABANDONED_OBJECT, det))

                    if "mask_face" in enabled_modules:
                        for det in detection_service.detect_mask_face(frame, timestamp):
                            all_raw.append((DetectionType.MASK_FACE, det))

                    if "crowd_density" in enabled_modules:
                        crowd = detection_service.calculate_crowd_density(frame, timestamp)
                        if crowd["count"] > 0:
                            all_raw.append((DetectionType.CROWD_DENSITY, crowd))

                    # Persist detections (deduplicated)
                    for det_type, det in all_raw:
                        dt_elapsed = (timestamp - last_detection_time.get(det_type, datetime.min)).total_seconds()
                        if dt_elapsed < dedup_interval:
                            continue
                        last_detection_time[det_type] = timestamp

                        confidence = det.get("confidence", 0.0)
                        module_settings = enabled_modules.get(det_type.value, {})
                        min_conf = module_settings.get("min_confidence")
                        if min_conf and confidence < min_conf:
                            continue

                        threat_level = detection_service.classify_threat_level(det_type, confidence, det, module_settings)
                        bbox = det.get("bbox", [0, 0, 0, 0])

                        db_detection = Detection(
                            camera_id=camera_id,
                            company_id=company_id,
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
                        db.flush()

                        # Alert for MEDIUM+
                        if threat_level in (ThreatLevel.MEDIUM, ThreatLevel.HIGH, ThreatLevel.CRITICAL):
                            alert = Alert(
                                detection_id=db_detection.id,
                                company_id=company_id,
                                title=f"{det_type.value.replace('_', ' ').title()} – {threat_level.value.upper()}",
                                message=f"{det_type.value} detected with {confidence:.0%} confidence",
                                status=AlertStatus.PENDING,
                            )
                            alert.email_sent_to = job["user_email"]
                            db.add(alert)
                            db.flush()
                            job["total_alerts"] += 1

                        active_detections.append({
                            "det_type": det_type.value,
                            "confidence": confidence,
                            "bbox": bbox,
                            "class_name": det.get("class", det_type.value),
                        })

                        job["detections"].append({
                            "id": db_detection.id,
                            "detection_type": det_type.value,
                            "threat_level": threat_level.value,
                            "confidence": round(confidence, 4),
                            "class": det.get("class", det_type.value),
                            "evidence_id": db_evidence.id,
                            "timestamp": timestamp.isoformat(),
                            "frame_number": frame_index,
                        })
                        job["total_detections"] += 1

                    previous_frames.append(frame)
                    if len(previous_frames) > 10:
                        previous_frames.pop(0)

                # Draw annotations on every frame
                if active_detections:
                    annotated = video_service.draw_detections_on_frame(frame, active_detections)
                else:
                    annotated = frame

                writer.write(annotated)

                # Update progress
                job["current_frame"] = frame_index
                job["progress"] = min(int((frame_index / total_frames) * 100), 100) if total_frames > 0 else 0

        finally:
            cap.release()
            writer.release()

        db.commit()

        job["status"] = "completed"
        job["progress"] = 100
        job["completed_at"] = datetime.utcnow().isoformat()
        job["output_video_url"] = f"/api/v1/video/jobs/{job_id}/result"
        logger.info("Job %s completed: %d detections, %d alerts", job_id, job["total_detections"], job["total_alerts"])

    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)
        logger.exception("Job %s failed: %s", job_id, e)
        db.rollback()
    finally:
        db.close()
        try:
            if os.path.exists(job["upload_path"]):
                os.remove(job["upload_path"])
        except Exception:
            pass


# ---------------------------------------------------------------------------
# GET /video/jobs/{job_id}  —  poll status + progress
# ---------------------------------------------------------------------------
@router.get("/jobs/{job_id}")
async def get_job_status(
    job_id: str,
    current_user: User = Depends(require_any_authenticated),
):
    """Get processing job status, progress, and results."""
    job = _get_job(job_id)

    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "progress": job["progress"],
        "current_frame": job["current_frame"],
        "total_frames": job["total_frames"],
        "duration_seconds": job["duration_seconds"],
        "total_detections": job["total_detections"],
        "total_alerts": job["total_alerts"],
        "detections": job["detections"],
        "output_video_url": job["output_video_url"],
        "started_at": job["started_at"],
        "completed_at": job["completed_at"],
        "error": job["error"],
    }


# ---------------------------------------------------------------------------
# GET /video/jobs/{job_id}/progress  —  SSE stream for real-time progress
# ---------------------------------------------------------------------------
@router.get("/jobs/{job_id}/progress")
async def stream_progress(
    job_id: str,
    current_user: User = Depends(require_any_authenticated),
):
    """
    Server-Sent Events stream for real-time progress.
    Frontend: const es = new EventSource('/api/v1/video/jobs/{job_id}/progress');
    """
    _get_job(job_id)

    async def event_generator():
        import json
        while True:
            job = _jobs.get(job_id)
            if not job:
                break

            payload = {
                "status": job["status"],
                "progress": job["progress"],
                "current_frame": job["current_frame"],
                "total_frames": job["total_frames"],
                "total_detections": job["total_detections"],
                "total_alerts": job["total_alerts"],
            }

            if job["status"] in ("completed", "failed"):
                if job["status"] == "completed":
                    payload["output_video_url"] = job["output_video_url"]
                else:
                    payload["error"] = job["error"]
                yield f"data: {json.dumps(payload)}\n\n"
                break

            yield f"data: {json.dumps(payload)}\n\n"
            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# GET /video/jobs/{job_id}/result  —  stream/play processed video
# ---------------------------------------------------------------------------
@router.get("/jobs/{job_id}/result")
async def get_processed_video(
    job_id: str,
    current_user: User = Depends(require_any_authenticated),
):
    """Stream the processed video with bounding boxes for playback."""
    job = _get_job(job_id)

    if job["status"] != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Job is {job['status']}, not completed yet",
        )

    file_path = Path(job["output_video"])
    if not file_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Processed video file not found")

    return FileResponse(
        path=str(file_path),
        media_type="video/mp4",
        headers={"Content-Disposition": f'inline; filename="processed_{job_id}.mp4"'},
    )


# ---------------------------------------------------------------------------
# GET /video/jobs/{job_id}/download  —  download processed video
# ---------------------------------------------------------------------------
@router.get("/jobs/{job_id}/download")
async def download_processed_video(
    job_id: str,
    current_user: User = Depends(require_any_authenticated),
):
    """Download the processed video with bounding boxes."""
    job = _get_job(job_id)

    if job["status"] != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Job is {job['status']}, not completed yet",
        )

    file_path = Path(job["output_video"])
    if not file_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Processed video file not found")

    return FileResponse(
        path=str(file_path),
        filename=f"processed_{job['filename']}",
        media_type="video/mp4",
    )


# ---------------------------------------------------------------------------
# GET /video/jobs  —  list all jobs for current user
# ---------------------------------------------------------------------------
@router.get("/jobs")
async def list_jobs(
    status_filter: Optional[str] = Query(None, alias="status", description="Filter by status"),
    current_user: User = Depends(require_any_authenticated),
):
    """List all processing jobs for the current user."""
    company_filter = get_user_company_filter(current_user)

    results = []
    for job in _jobs.values():
        if company_filter is not None and job.get("company_id") != company_filter:
            if job.get("user_id") != current_user.id:
                continue

        if status_filter and job["status"] != status_filter:
            continue

        results.append({
            "job_id": job["job_id"],
            "status": job["status"],
            "progress": job["progress"],
            "total_frames": job["total_frames"],
            "total_detections": job["total_detections"],
            "total_alerts": job["total_alerts"],
            "filename": job["filename"],
            "camera_id": job["camera_id"],
            "started_at": job["started_at"],
            "completed_at": job["completed_at"],
            "output_video_url": job["output_video_url"],
        })

    return results
