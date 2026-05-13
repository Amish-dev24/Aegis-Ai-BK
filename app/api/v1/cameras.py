"""
Camera management endpoints.
"""

from typing import Optional

import asyncio
import time

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request, status
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.security import (
    check_company_access,
    get_current_user,
    get_user_company_filter,
    oauth2_optional,
    require_security_officer,
    user_from_access_token,
)
from app.database import get_db
from app.models.audit_log import create_audit_log
from app.models.camera import Camera
from app.models.user import User
from app.schemas.camera import CameraCreate, CameraResponse, CameraUpdate
from app.services.frame_detection_pipeline import encode_jpeg_bytes, grab_jpeg_snapshot, open_stream_capture
from app.services import live_camera_runtime

router = APIRouter(prefix="/cameras", tags=["cameras"])


class LiveDetectionStartBody(BaseModel):
    process_fps: float = Field(default=1.0, ge=0.25, le=15.0, description="Rough target FPS for inference")


def _get_camera_or_404(db: Session, camera_id: int) -> Camera:
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Camera not found")
    return camera


def _ensure_camera_company_access(camera: Camera, user: User) -> None:
    if camera.company_id and not check_company_access(user, camera.company_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to access this camera",
        )


@router.post("", response_model=CameraResponse, status_code=status.HTTP_201_CREATED)
async def create_camera(
    request: Request,
    camera_data: CameraCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_security_officer),
):
    """Create a new camera. Company users can only create cameras for their company."""
    camera_dict = camera_data.dict()

    # Tenant: cameras always belong to the user's company (aegis_admin included)
    cid = current_user.company_id
    if cid is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User must be assigned to a company to manage cameras",
        )
    req_cid = camera_dict.get("company_id")
    if req_cid is not None and req_cid != cid:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot assign a camera to another company",
        )
    camera_dict["company_id"] = cid

    db_camera = Camera(**camera_dict)
    db.add(db_camera)
    db.commit()
    db.refresh(db_camera)

    # Log creation
    db.add(
        create_audit_log(
            request,
            current_user.id,
            "create_camera",
            "camera",
            db_camera.id,
            {
                "name": db_camera.name,
                "location": db_camera.location,
                "company_id": db_camera.company_id,
            },
        )
    )
    db.commit()

    return db_camera


@router.get("", response_model=list[CameraResponse])
async def list_cameras(
    request: Request,
    company_id: Optional[int] = Query(None, description="Filter by company"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    active_only: bool = Query(default=False, description="Return only active cameras"),
    limit: int = Query(default=200, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    """List cameras for the authenticated user's company only."""
    query = db.query(Camera)

    company_filter = get_user_company_filter(current_user, company_id)
    if company_filter is None:
        return []

    query = query.filter(Camera.company_id == company_filter)

    if active_only:
        query = query.filter(Camera.is_active.is_(True))

    cameras = query.order_by(Camera.id).offset(offset).limit(limit).all()
    return cameras


@router.get("/{camera_id}", response_model=CameraResponse)
async def get_camera(
    camera_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)
):
    """Get camera by ID. Company users can only access their company's cameras."""
    camera = _get_camera_or_404(db, camera_id)

    # Check company access
    _ensure_camera_company_access(camera, current_user)

    return camera


@router.get("/{camera_id}/snapshot", response_class=Response)
async def camera_snapshot(
    camera_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Single JPEG frame from ``stream_url`` (RTSP/MJPEG/HTTP); for UI preview."""

    camera = _get_camera_or_404(db, camera_id)
    _ensure_camera_company_access(camera, current_user)

    if not camera.stream_url or not str(camera.stream_url).strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Camera has no stream_url configured",
        )

    jpeg, grab_err = await asyncio.to_thread(
        grab_jpeg_snapshot, str(camera.stream_url).strip(), 12.0
    )
    if not jpeg:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=grab_err or "Could not grab frame from stream",
        )

    return Response(content=jpeg, media_type="image/jpeg")


@router.get("/{camera_id}/preview")
async def camera_preview(
    request: Request,
    camera_id: int,
    db: Session = Depends(get_db),
    token_header: Optional[str] = Depends(oauth2_optional),
    access_token: Optional[str] = Query(
        default=None,
        description="Same JWT as login when the client cannot send Authorization (e.g. HTML img); prefer header when possible",
    ),
    fps: float = Query(default=12.0, ge=4.0, le=30.0, description="Target output FPS (limits JPEG rate)"),
):
    """
    Low-latency MJPEG preview: one long-lived RTSP read loop.

    Prefer this in the UI (e.g. ``<img src=".../preview?access_token=...">`` or any
    player that supports multipart MJPEG with Bearer auth on the request) instead of
    polling ``/snapshot``, which reconnects to RTSP on every frame and looks choppy.
    """

    token = token_header or access_token
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    current_user = user_from_access_token(db, token)

    camera = _get_camera_or_404(db, camera_id)
    _ensure_camera_company_access(camera, current_user)

    if not camera.stream_url or not str(camera.stream_url).strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Camera has no stream_url configured",
        )

    stream_url = str(camera.stream_url).strip()
    frame_interval = 1.0 / fps

    async def mjpeg_stream():
        # When live detection is running, reuse its decoder so we do not open a second RTSP client
        # (many devices allow only one stream or drop frames under dual read).
        cap = None
        try:
            while True:
                if await request.is_disconnected():
                    break
                loop_start = time.monotonic()

                shared = await asyncio.to_thread(
                    live_camera_runtime.get_live_preview_frame, camera_id, 3.0
                )
                if shared is not None:
                    if cap is not None:
                        await asyncio.to_thread(cap.release)
                        cap = None
                    frame = shared
                else:
                    if cap is None:
                        cap = await asyncio.to_thread(open_stream_capture, stream_url)
                        if not cap.isOpened():
                            return
                    ok, frame = await asyncio.to_thread(cap.read)
                    if (
                        not ok
                        or frame is None
                        or getattr(frame, "size", 0) == 0
                    ):
                        await asyncio.sleep(0.05)
                        continue

                ok_j, jpeg = await asyncio.to_thread(encode_jpeg_bytes, frame, 82)
                if not ok_j or not jpeg:
                    continue
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"\r\n" + jpeg + b"\r\n"
                )
                elapsed = time.monotonic() - loop_start
                wait = frame_interval - elapsed
                if wait > 0:
                    await asyncio.sleep(wait)
        finally:
            if cap is not None:
                await asyncio.to_thread(cap.release)

    return StreamingResponse(
        mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.post("/{camera_id}/live-detection/start")
async def start_live_detection(
    request: Request,
    camera_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_security_officer),
    body: LiveDetectionStartBody = Body(),
):
    """Begin background inference on live ``stream_url`` for this camera."""

    camera = _get_camera_or_404(db, camera_id)
    _ensure_camera_company_access(camera, current_user)

    if not camera.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Camera must be active to start live detection",
        )

    ok, msg = live_camera_runtime.start_live(
        camera.id,
        camera.stream_url or "",
        body.process_fps,
        notify_user_email=current_user.email,
        notify_alert_preference=getattr(current_user, "alert_preference", None) or "email",
        notify_user_phone=getattr(current_user, "phone_number", None),
        camera_name=camera.name or "",
    )
    if not ok:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=msg)

    db.add(
        create_audit_log(
            request,
            current_user.id,
            "start_live_detection",
            "camera",
            camera_id,
            {"camera_name": camera.name, "process_fps": body.process_fps},
        )
    )
    db.commit()
    return {"success": True, "message": msg}


@router.post("/{camera_id}/live-detection/stop")
async def stop_live_detection(
    request: Request,
    camera_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_security_officer),
):
    """Stop background live inference if running."""

    camera = _get_camera_or_404(db, camera_id)
    _ensure_camera_company_access(camera, current_user)

    ok, msg = live_camera_runtime.stop_live(camera_id)
    if ok:
        db.add(
            create_audit_log(
                request,
                current_user.id,
                "stop_live_detection",
                "camera",
                camera_id,
                {"camera_name": camera.name},
            )
        )
        db.commit()
    return {"success": ok, "message": msg}


@router.get("/{camera_id}/live-detection/status")
async def live_detection_status(
    camera_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Whether live detection is running and basic counters."""

    camera = _get_camera_or_404(db, camera_id)
    _ensure_camera_company_access(camera, current_user)

    st = live_camera_runtime.status(camera_id)
    if st is None:
        return {"running": False, "camera_id": camera_id}
    return st


@router.put("/{camera_id}", response_model=CameraResponse)
async def update_camera(
    request: Request,
    camera_id: int,
    camera_update: CameraUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_security_officer),
):
    """Update camera. Company users can only update their company's cameras."""
    camera = _get_camera_or_404(db, camera_id)
    _ensure_camera_company_access(camera, current_user)

    update_data = camera_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(camera, field, value)

    db.commit()
    db.refresh(camera)

    # Log update
    db.add(
        create_audit_log(
            request,
            current_user.id,
            "update_camera",
            "camera",
            camera_id,
            {"updated_fields": list(update_data.keys()), "camera_name": camera.name},
        )
    )
    db.commit()

    return camera


@router.delete("/{camera_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_camera(
    request: Request,
    camera_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_security_officer),
):
    """Delete camera. Company users can only delete their company's cameras."""
    camera = _get_camera_or_404(db, camera_id)
    _ensure_camera_company_access(camera, current_user)

    live_camera_runtime.stop_live(camera_id)

    # Log deletion
    db.add(
        create_audit_log(
            request,
            current_user.id,
            "delete_camera",
            "camera",
            camera_id,
            {"camera_name": camera.name, "location": camera.location},
        )
    )
    db.delete(camera)
    db.commit()

    return None
