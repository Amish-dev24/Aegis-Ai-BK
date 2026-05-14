"""
Background live-detection workers (one thread per camera_id).

Architecture (two-thread design):
  ┌─────────────────────┐     latest frame      ┌─────────────────────┐
  │  _FrameReaderThread  │ ──────────────────▶  │  _InferenceThread    │
  │  reads at camera FPS │                       │  runs AI at proc_fps │
  │  publishes preview   │                       │  persists detections │
  └─────────────────────┘                       └─────────────────────┘

This keeps the MJPEG preview at full camera FPS (≈ real CCTV) while AI
inference runs at the configurable process_fps without blocking the reader.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import numpy as np

from app.database import SessionLocal
from app.models.camera import Camera
from app.services.detection_service import detection_service
from app.services.frame_detection_pipeline import (
    live_prior_frames_capacity,
    open_stream_capture,
    persist_detections_for_live_frame,
)

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_sessions: dict[int, "_LiveSession"] = {}

# Latest decoded BGR frame per camera — shared between reader thread and preview endpoint.
_preview_lock = threading.Lock()
_preview_last: dict[int, tuple[float, np.ndarray]] = {}


class _LatestFrame:
    """Thread-safe single-slot buffer: reader writes at camera FPS, inference reads when ready."""

    def __init__(self):
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._ts: Optional[datetime] = None
        self._event = threading.Event()

    def put(self, frame: np.ndarray, ts: datetime) -> None:
        with self._lock:
            self._frame = frame
            self._ts = ts
        self._event.set()

    def get(self, timeout: float = 1.0) -> tuple[Optional[np.ndarray], Optional[datetime]]:
        self._event.wait(timeout=timeout)
        with self._lock:
            f, t = self._frame, self._ts
            self._frame = None
            self._ts = None
            self._event.clear()
        return f, t


def _publish_live_preview_frame(camera_id: int, frame: np.ndarray) -> None:
    if frame is None or getattr(frame, "size", 0) == 0:
        return
    h, w = int(frame.shape[0]), int(frame.shape[1])
    if h < 8 or w < 8:
        return
    with _preview_lock:
        _preview_last[camera_id] = (time.monotonic(), frame.copy())


def get_live_preview_frame(camera_id: int, max_age_sec: float = 3.0) -> Optional[np.ndarray]:
    """BGR frame from the live-detection reader, or None if not live or stale."""
    with _lock:
        if camera_id not in _sessions:
            return None
    with _preview_lock:
        item = _preview_last.get(camera_id)
        if not item:
            return None
        t0, fr = item
        if time.monotonic() - t0 > float(max_age_sec):
            return None
        return fr.copy()


def _clear_live_preview_frame(camera_id: int) -> None:
    with _preview_lock:
        _preview_last.pop(camera_id, None)


@dataclass
class _LiveSession:
    camera_id: int
    thread: threading.Thread
    stop: threading.Event
    process_fps: float
    started_at: datetime
    frames_processed: int = 0
    last_detection_at: Optional[datetime] = None
    last_frame_at: Optional[datetime] = None
    detections_total: int = 0
    alerts_total: int = 0
    last_error: Optional[str] = None


def _frame_reader(
    camera_id: int,
    stream_url: str,
    stop: threading.Event,
    slot: _LatestFrame,
) -> None:
    """
    Dedicated reader thread: pulls frames from the RTSP/HTTP source at full
    camera FPS and writes each to *slot* + the shared preview buffer.
    Completely independent of AI inference speed.
    """
    cap = open_stream_capture(stream_url)
    if not cap.isOpened():
        with _lock:
            s = _sessions.get(camera_id)
            if s:
                s.last_error = "Failed to open stream"
        logger.warning("live reader: camera %s open failed", camera_id)
        return

    logger.info("live reader started: camera %s", camera_id)
    consecutive_failures = 0

    while not stop.is_set():
        ok, frame = cap.read()
        if not ok or frame is None or getattr(frame, "size", 0) == 0:
            consecutive_failures += 1
            if consecutive_failures > 30:
                logger.warning("live reader: too many failures for camera %s, stopping", camera_id)
                with _lock:
                    s = _sessions.get(camera_id)
                    if s:
                        s.last_error = "Stream read failed repeatedly"
                break
            time.sleep(0.05)
            continue

        fh, fw = int(frame.shape[0]), int(frame.shape[1])
        if fh < 32 or fw < 32:
            continue

        consecutive_failures = 0
        now = datetime.utcnow()

        with _lock:
            s = _sessions.get(camera_id)
            if s:
                s.last_frame_at = now

        # Publish for MJPEG preview at full camera FPS
        _publish_live_preview_frame(camera_id, frame)
        # Signal the inference thread
        slot.put(frame.copy(), now)

    cap.release()
    logger.info("live reader stopped: camera %s", camera_id)


def _inference_worker(
    camera_id: int,
    process_fps: float,
    stop: threading.Event,
    slot: _LatestFrame,
    notify_user_email: Optional[str],
    notify_alert_preference: str,
    notify_user_phone: Optional[str],
    camera_name: str,
    notify_extra_emails: Optional[list[str]] = None,
) -> None:
    """
    Inference thread: consumes frames from *slot* at *process_fps* rate and
    runs all enabled AI modules. Does NOT touch the RTSP stream directly.
    """
    min_interval = max(0.05, 1.0 / max(process_fps, 0.1))

    live_notify = None
    if notify_user_email:
        live_notify = {
            "user_email": notify_user_email,
            "user_alert_preference": notify_alert_preference or "email",
            "user_phone": notify_user_phone or "",
            "camera_name": camera_name or "",
            "extra_emails": list(notify_extra_emails) if notify_extra_emails else [],
        }

    buf_max = 0
    prev_buf: list = []
    object_history: dict[str, list[datetime]] = {}

    logger.info("live inference started: camera %s @ %.1f fps", camera_id, process_fps)

    while not stop.is_set():
        loop_start = time.monotonic()

        frame, now = slot.get(timeout=1.0)
        if frame is None or stop.is_set():
            continue

        db = SessionLocal()
        try:
            cam = db.query(Camera).filter(Camera.id == camera_id).first()
            if not cam or not cam.is_active:
                break

            enabled_now = detection_service.get_enabled_modules(db, cam.company_id)
            need_buf = live_prior_frames_capacity(process_fps, "violence" in enabled_now)
            if need_buf != buf_max:
                buf_max = need_buf
            if buf_max == 0:
                prev_buf.clear()
            else:
                while len(prev_buf) > buf_max:
                    prev_buf.pop(0)

            prior_snap = [x.copy() for x in prev_buf] if buf_max > 0 else []

            dc, ac, _ = persist_detections_for_live_frame(
                db,
                cam,
                frame,
                now,
                previous_frames=prior_snap if prior_snap else None,
                live_notify=live_notify,
                object_history=object_history,
            )
            db.commit()

            with _lock:
                ss = _sessions.get(camera_id)
                if ss:
                    ss.frames_processed += 1
                    ss.detections_total += dc
                    ss.alerts_total += ac
                    if dc > 0:
                        ss.last_detection_at = now
                    ss.last_error = None

        except Exception as e:
            logger.exception("live inference error camera=%s", camera_id)
            try:
                db.rollback()
            except Exception:
                pass
            with _lock:
                ss = _sessions.get(camera_id)
                if ss:
                    ss.last_error = str(e)[:240]
        finally:
            db.close()

        if buf_max > 0:
            prev_buf.append(frame.copy())
            while len(prev_buf) > buf_max:
                prev_buf.pop(0)

        # Throttle inference to process_fps; remaining time is spent waiting for next frame
        elapsed = time.monotonic() - loop_start
        remaining = min_interval - elapsed
        if remaining > 0:
            stop.wait(timeout=remaining)

    logger.info("live inference stopped: camera %s", camera_id)


def _worker(
    camera_id: int,
    stream_url: str,
    process_fps: float,
    stop: threading.Event,
    notify_user_email: Optional[str],
    notify_alert_preference: str,
    notify_user_phone: Optional[str],
    camera_name: str,
    notify_extra_emails: Optional[list[str]] = None,
) -> None:
    """
    Entry point called by the session thread. Spawns a reader sub-thread for
    full-FPS capture and runs inference in the current thread.
    """
    slot = _LatestFrame()

    reader = threading.Thread(
        target=_frame_reader,
        args=(camera_id, stream_url, stop, slot),
        name=f"aegis-reader-{camera_id}",
        daemon=True,
    )
    reader.start()

    try:
        _inference_worker(
            camera_id=camera_id,
            process_fps=process_fps,
            stop=stop,
            slot=slot,
            notify_user_email=notify_user_email,
            notify_alert_preference=notify_alert_preference,
            notify_user_phone=notify_user_phone,
            camera_name=camera_name,
            notify_extra_emails=notify_extra_emails,
        )
    finally:
        stop.set()          # ensure reader also exits
        reader.join(timeout=6.0)
        _clear_live_preview_frame(camera_id)
        with _lock:
            _sessions.pop(camera_id, None)


def start_live(
    camera_id: int,
    stream_url: str,
    process_fps: float = 1.0,
    *,
    notify_user_email: Optional[str] = None,
    notify_alert_preference: str = "email",
    notify_user_phone: Optional[str] = None,
    camera_name: str = "",
    notify_extra_emails: Optional[list[str]] = None,
) -> tuple[bool, str]:
    if not stream_url or not stream_url.strip():
        return False, "Camera has no stream_url configured"

    with _lock:
        if camera_id in _sessions:
            return False, "Live detection already running for this camera"
        stop = threading.Event()
        t = threading.Thread(
            target=_worker,
            args=(
                camera_id,
                stream_url.strip(),
                float(process_fps),
                stop,
                notify_user_email,
                notify_alert_preference,
                notify_user_phone,
                camera_name,
                list(notify_extra_emails) if notify_extra_emails else [],
            ),
            name=f"aegis-live-camera-{camera_id}",
            daemon=True,
        )
        sess = _LiveSession(
            camera_id=camera_id,
            thread=t,
            stop=stop,
            process_fps=process_fps,
            started_at=datetime.utcnow(),
        )
        _sessions[camera_id] = sess
        t.start()
    return True, "started"


def stop_live(camera_id: int) -> tuple[bool, str]:
    with _lock:
        sess = _sessions.get(camera_id)
        if not sess:
            return False, "Live detection not running"
        sess.stop.set()
        th = sess.thread
    th.join(timeout=12.0)
    with _lock:
        _sessions.pop(camera_id, None)
    return True, "stopped"


def status(camera_id: int) -> Optional[dict[str, Any]]:
    with _lock:
        sess = _sessions.get(camera_id)
        if not sess:
            return None
        return {
            "running": True,
            "camera_id": camera_id,
            "process_fps": sess.process_fps,
            "started_at": sess.started_at.isoformat() + "Z",
            "frames_processed": sess.frames_processed,
            "detections_total": sess.detections_total,
            "alerts_total": sess.alerts_total,
            "last_frame_at": sess.last_frame_at.isoformat() + "Z" if sess.last_frame_at else None,
            "last_detection_at": sess.last_detection_at.isoformat() + "Z" if sess.last_detection_at else None,
            "last_error": sess.last_error,
        }


def shutdown_all() -> None:
    with _lock:
        ids = list(_sessions.keys())
    for cid in ids:
        stop_live(cid)
