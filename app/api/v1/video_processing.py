"""
Video processing endpoints with progress tracking and annotated video output.

Optimizations:
- Downscale frames before AI inference (YOLO works at 640px)
- Skip more frames (process every 30th = 1fps for 30fps video)
- Skip annotated video output by default (optional)
- Batch DB commits
- Thread pool so event loop stays free
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
import numpy as np

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/video", tags=["video-processing"])

_thread_pool = ThreadPoolExecutor(max_workers=2)
_jobs: dict[str, dict] = {}


def _get_job(job_id: str) -> dict:
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    return job


# ---------------------------------------------------------------------------
# POST /video/process
# ---------------------------------------------------------------------------
@router.post("/process")
async def start_video_processing(
    request: Request,
    camera_id: int,
    video_file: UploadFile = File(...),
    generate_video: bool = Query(False, description="Generate annotated output video (slower)"),
    process_fps: int = Query(1, ge=1, le=10, description="Frames per second to analyze (1=fast, 10=thorough)"),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_security_officer),
):
    """
    Upload a video for AI processing.

    Speed options:
    - process_fps=1 (default): ~1 min for a 2-min video (fast, detection-only)
    - process_fps=3: ~3 min (more thorough)
    - generate_video=true: adds ~1-2 min (writes annotated MP4 with boxes)
    """
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Camera not found")
    if camera.company_id and not check_company_access(current_user, camera.company_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions")

    import aiofiles
    upload_path = Path(settings.UPLOAD_DIR) / f"{uuid.uuid4().hex}_{video_file.filename}"
    async with aiofiles.open(upload_path, "wb") as f:
        content = await video_file.read()
        await f.write(content)

    info = video_service.get_video_info(str(upload_path))

    job_id = uuid.uuid4().hex
    output_filename = f"processed_{job_id}.mp4"
    output_path = Path(settings.PROCESSED_VIDEO_DIR) / output_filename

    # Calculate process_every_n from desired fps
    video_fps = info["fps"] or 30.0
    process_every_n = max(1, int(video_fps / process_fps))

    _jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0,
        "current_frame": 0,
        "total_frames": info["total_frames"],
        "fps": video_fps,
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
        # Processing options
        "generate_video": generate_video,
        "process_every_n": process_every_n,
        "process_fps": process_fps,
    }

    loop = asyncio.get_event_loop()
    loop.run_in_executor(_thread_pool, _process_video_sync, job_id)

    analyzed_frames = info["total_frames"] // process_every_n
    return {
        "job_id": job_id,
        "status": "queued",
        "total_frames": info["total_frames"],
        "frames_to_analyze": analyzed_frames,
        "process_every_n": process_every_n,
        "generate_video": generate_video,
        "duration_seconds": round(info["duration_seconds"], 2),
        "message": f"Processing started. Analyzing {analyzed_frames} frames ({process_fps} fps). Poll /video/jobs/{job_id} for status.",
    }


# ---------------------------------------------------------------------------
# Sync processing in thread pool
# ---------------------------------------------------------------------------
def _process_video_sync(job_id: str):
    """Run detection + optional annotation. Runs in a worker thread."""
    job = _jobs[job_id]
    db = SessionLocal()

    try:
        job["status"] = "processing"
        upload_path = job["upload_path"]
        output_path = job["output_video"]
        generate_video = job["generate_video"]
        process_every_n = job["process_every_n"]

        info = video_service.get_video_info(upload_path)
        fps = info["fps"]
        width = info["width"]
        height = info["height"]
        total_frames = info["total_frames"]

        cap = cv2.VideoCapture(upload_path)

        # Only create video writer if requested
        writer = None
        if generate_video:
            writer = video_service.create_video_writer(output_path, fps, width, height)

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
        pending_db_count = 0

        # Pre-calculate resize for AI inference (640px max side)
        ai_max_size = 640
        scale = min(ai_max_size / width, ai_max_size / height, 1.0)
        ai_width = int(width * scale)
        ai_height = int(height * scale)
        needs_resize = scale < 1.0

        logger.info("Job %s: %d frames, process every %d (analyze %d), video=%s, resize=%s (%.0f%%)",
                     job_id, total_frames, process_every_n,
                     total_frames // process_every_n, generate_video,
                     needs_resize, scale * 100)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_index += 1
                elapsed = timedelta(seconds=frame_index / fps)
                timestamp = video_start + elapsed

                is_analysis_frame = (frame_index % process_every_n == 0)

                # Run AI on analysis frames
                if is_analysis_frame:
                    # Downscale for AI (faster inference)
                    ai_frame = cv2.resize(frame, (ai_width, ai_height)) if needs_resize else frame

                    all_raw: list[tuple] = []
                    active_detections = []

                    if "weapon" in enabled_modules:
                        for det in detection_service.detect_weapons(ai_frame, timestamp):
                            all_raw.append((DetectionType.WEAPON, det))

                    if "violence" in enabled_modules:
                        # Violence needs previous frames at same scale
                        resized_prev = [cv2.resize(f, (ai_width, ai_height)) for f in previous_frames[-3:]] if needs_resize and previous_frames else previous_frames[-3:]
                        violence = detection_service.detect_violence(ai_frame, resized_prev, timestamp)
                        if violence["is_violent"]:
                            all_raw.append((DetectionType.VIOLENCE, violence))

                    if "abandoned_object" in enabled_modules:
                        for det in detection_service.detect_abandoned_object(ai_frame, None, timestamp, object_history):
                            all_raw.append((DetectionType.ABANDONED_OBJECT, det))

                    if "mask_face" in enabled_modules:
                        for det in detection_service.detect_mask_face(ai_frame, timestamp):
                            all_raw.append((DetectionType.MASK_FACE, det))

                    if "crowd_density" in enabled_modules:
                        crowd = detection_service.calculate_crowd_density(ai_frame, timestamp)
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

                        # Save snapshot using original full-res frame
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

                            # Send email from sync thread
                            try:
                                import asyncio as _aio
                                sent = _aio.run(email_service.send_alert_email(
                                    to_emails=[job["user_email"]],
                                    subject=alert.title,
                                    message=alert.message,
                                    snapshot_path=snapshot_path,
                                    metadata={
                                        "Detection Type": det_type.value,
                                        "Threat Level": threat_level.value,
                                        "Confidence": f"{confidence:.2%}",
                                        "Camera": job["camera_name"],
                                        "Timestamp": timestamp.isoformat(),
                                    },
                                ))
                                if sent:
                                    alert.email_sent = True
                                    alert.email_sent_at = datetime.utcnow()
                            except Exception as email_err:
                                logger.warning("Email failed for alert %s: %s", alert.id, email_err)

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
                        pending_db_count += 1

                    # Keep sliding window (only store every Nth for memory)
                    previous_frames.append(frame)
                    if len(previous_frames) > 5:
                        previous_frames.pop(0)

                    # Batch commit every 10 detections
                    if pending_db_count >= 10:
                        db.commit()
                        pending_db_count = 0

                # Write annotated video frame (only if requested)
                if writer:
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
            if writer:
                writer.release()

        # Final commit
        if pending_db_count > 0:
            db.commit()

        # Re-encode only if video was generated
        if generate_video and Path(output_path).exists():
            video_service.reencode_to_h264(output_path)
            job["output_video_url"] = f"/api/v1/video/jobs/{job_id}/result"

        job["status"] = "completed"
        job["progress"] = 100
        job["completed_at"] = datetime.utcnow().isoformat()
        logger.info("Job %s completed: %d detections, %d alerts",
                     job_id, job["total_detections"], job["total_alerts"])

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
# GET /video/jobs/{job_id}
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
# GET /video/jobs/{job_id}/progress  — SSE
# ---------------------------------------------------------------------------
@router.get("/jobs/{job_id}/progress")
async def stream_progress(
    job_id: str,
    current_user: User = Depends(require_any_authenticated),
):
    """Server-Sent Events stream for real-time progress."""
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
# GET /video/jobs/{job_id}/result
# ---------------------------------------------------------------------------
@router.get("/jobs/{job_id}/result")
async def get_processed_video(
    job_id: str,
    current_user: User = Depends(require_any_authenticated),
):
    """Stream the processed video with bounding boxes for playback."""
    job = _get_job(job_id)

    if job["status"] != "completed":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Job is {job['status']}")

    if not job.get("output_video_url"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No video generated. Use generate_video=true when processing.")

    file_path = Path(job["output_video"])
    if not file_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Processed video file not found")

    return FileResponse(
        path=str(file_path),
        media_type="video/mp4",
        headers={"Content-Disposition": f'inline; filename="processed_{job_id}.mp4"'},
    )


# ---------------------------------------------------------------------------
# GET /video/jobs/{job_id}/download
# ---------------------------------------------------------------------------
@router.get("/jobs/{job_id}/download")
async def download_processed_video(
    job_id: str,
    current_user: User = Depends(require_any_authenticated),
):
    """Download the processed video."""
    job = _get_job(job_id)

    if job["status"] != "completed":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Job is {job['status']}")

    if not job.get("output_video_url"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No video generated.")

    file_path = Path(job["output_video"])
    if not file_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Processed video file not found")

    return FileResponse(
        path=str(file_path),
        filename=f"processed_{job['filename']}",
        media_type="video/mp4",
    )


# ---------------------------------------------------------------------------
# GET /video/jobs
# ---------------------------------------------------------------------------
@router.get("/jobs")
async def list_jobs(
    status_filter: Optional[str] = Query(None, alias="status"),
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
