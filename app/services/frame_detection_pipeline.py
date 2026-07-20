"""
Frame→detection persistence for live camera worker (sync, thread-safe).

Alert rows respect company ``alert_on_levels``; optional ``live_notify`` queues email/SMS like video jobs.
"""

import logging
import os
import threading
import time
from datetime import datetime
from typing import Any, Optional

import cv2
import numpy as np
from sqlalchemy.orm import Session

from app.config import settings
from app.models.alert import Alert, AlertStatus
from app.models.camera import Camera
from app.models.detection import Detection, DetectionType
from app.models.evidence import Evidence
from app.services.detection_service import _normalize_model_confidence, detection_service
from app.services.video_service import video_service

logger = logging.getLogger(__name__)

# Wowza Cloud rate-limits rapid reconnects (playlist often returns 403 after the first hit).
# Cache recent JPEGs and serialize opens per URL so UI polling does not burn the stream.
_snapshot_cache_lock = threading.Lock()
_snapshot_jpeg_cache: dict[str, tuple[float, bytes]] = {}
_SNAPSHOT_CACHE_TTL_SEC = 12.0
_open_locks_guard = threading.Lock()
_open_locks: dict[str, threading.Lock] = {}


def _url_open_lock(url: str) -> threading.Lock:
    with _open_locks_guard:
        lock = _open_locks.get(url)
        if lock is None:
            lock = threading.Lock()
            _open_locks[url] = lock
        return lock


def _live_ai_frame_pair(full_bgr: np.ndarray, prior_full: list) -> tuple[np.ndarray, list]:
    """Match file-analysis geometry: max-side ``WEAPON_VIDEO_AI_MAX_SIDE`` for YOLO / violence / face."""
    ai_max = int(getattr(settings, "WEAPON_VIDEO_AI_MAX_SIDE", 640))
    h0, w0 = full_bgr.shape[:2]
    rscale = min(ai_max / float(w0), ai_max / float(h0), 1.0)
    ai_w, ai_h = int(round(w0 * rscale)), int(round(h0 * rscale))
    if rscale < 1.0:
        ai_curr = cv2.resize(full_bgr, (ai_w, ai_h), interpolation=cv2.INTER_LINEAR)
        ai_prior = [
            cv2.resize(f, (ai_w, ai_h), interpolation=cv2.INTER_LINEAR)
            for f in prior_full
            if f is not None and getattr(f, "size", 0) > 0
        ]
    else:
        ai_curr = full_bgr
        ai_prior = list(prior_full)
    return ai_curr, ai_prior


def live_prior_frames_capacity(process_fps: float, violence_enabled: bool) -> int:
    """Prior frames to retain for live Conv3D violence (matches video job buffer sizing)."""
    from app.services.violence_model import ViolenceClassifier

    if not violence_enabled:
        return 0
    file_win = float(getattr(settings, "VIOLENCE_TEMPORAL_WINDOW_SECONDS", 2.0))
    live_win = float(getattr(settings, "VIOLENCE_LIVE_TEMPORAL_WINDOW_SECONDS", 3.5))
    # Live: avoid capping the buffer at only ~2s when user selects high process_fps (e.g. 10).
    win = max(file_win, live_win)
    cap = int(getattr(settings, "VIOLENCE_HISTORY_MAX_FRAMES", 120))
    target = int(max(1.0, process_fps) * win + 0.999)
    prior_needed = max(0, ViolenceClassifier.NUM_FRAMES - 1)
    return max(prior_needed, min(cap, target))


def persist_detections_for_live_frame(
    db: Session,
    camera: Camera,
    frame,
    timestamp: datetime,
    previous_frames: Optional[list] = None,
    live_notify: Optional[dict[str, Any]] = None,
    object_history: Optional[dict[str, list[datetime]]] = None,
) -> tuple[int, int, list[dict[str, Any]]]:
    """Run enabled modules, persist detections/evidence/alerts for one frame."""
    camera_id = camera.id
    detections_created = 0
    alerts_created = 0
    detection_results: list[dict[str, Any]] = []
    all_raw_detections: list[tuple] = []

    enabled_modules = detection_service.get_enabled_modules(db, camera.company_id)

    prior = list(previous_frames) if previous_frames else []

    need_ai = (
        "weapon" in enabled_modules
        or "abandoned_object" in enabled_modules
        or "violence" in enabled_modules
        or "mask_face" in enabled_modules
    )
    if need_ai:
        ai_frame, ai_prior = _live_ai_frame_pair(frame, prior)
    else:
        ai_frame, ai_prior = frame, prior

    yolo_results = None
    weapon_conf_eff = float(detection_service.confidence_threshold)
    need_weapon_yolo = "weapon" in enabled_modules and detection_service.weapon_model is not None
    need_bag_box_yolo = (
        "abandoned_object" in enabled_modules and detection_service.bag_box_model is not None
    )
    if need_weapon_yolo or need_bag_box_yolo:
        w_minc = enabled_modules.get("weapon", {}).get("min_confidence")
        weapon_conf_eff = (
            float(w_minc) if w_minc is not None else float(detection_service.confidence_threshold)
        )
        yolo_conf = max(0.01, min(weapon_conf_eff, 0.25)) if "weapon" in enabled_modules else 0.25
        yolo_results = detection_service.run_yolo_shared(ai_frame, conf=yolo_conf)

    if "weapon" in enabled_modules and yolo_results:
        for det in detection_service._extract_weapons(yolo_results, weapon_conf_eff):
            all_raw_detections.append((DetectionType.WEAPON, det))

    if "abandoned_object" in enabled_modules and object_history is not None:
        ab_settings = enabled_modules.get("abandoned_object", {})
        saved_thr = detection_service.abandoned_threshold
        try:
            if ab_settings.get("abandoned_seconds"):
                detection_service.abandoned_threshold = float(ab_settings["abandoned_seconds"])
            if detection_service.bag_box_model is not None and yolo_results:
                for det in detection_service._extract_abandoned(
                    yolo_results, ai_frame, timestamp, object_history
                ):
                    all_raw_detections.append((DetectionType.ABANDONED_OBJECT, det))
            else:
                for det in detection_service.detect_abandoned_object(
                    frame, None, timestamp, object_history
                ):
                    all_raw_detections.append((DetectionType.ABANDONED_OBJECT, det))
        finally:
            detection_service.abandoned_threshold = saved_thr

    if "violence" in enabled_modules:
        vset = enabled_modules.get("violence", {})
        live_ovr = getattr(settings, "VIOLENCE_LIVE_PROB_THRESHOLD", None)
        v_min = vset.get("min_confidence")
        if live_ovr is not None:
            prob_thr = _normalize_model_confidence(float(live_ovr))
        elif v_min is not None:
            prob_thr = float(v_min)
        else:
            prob_thr = None
        violence = detection_service.detect_violence(
            ai_frame, ai_prior, timestamp, prob_threshold=prob_thr
        )
        if violence["is_violent"]:
            all_raw_detections.append((DetectionType.VIOLENCE, violence))

    if "mask_face" in enabled_modules:
        for det in detection_service.detect_mask_face(ai_frame, timestamp):
            all_raw_detections.append((DetectionType.MASK_FACE, det))

    if "crowd_density" in enabled_modules:
        crowd = detection_service.calculate_crowd_density(frame, timestamp)
        if int(crowd.get("count", 0) or 0) > 0:
            all_raw_detections.append((DetectionType.CROWD_DENSITY, crowd))

    for det_type, det in all_raw_detections:
        confidence = det.get("confidence", 0.0)
        module_settings = enabled_modules.get(det_type.value, {})

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
        db.flush()

        alert_levels_str = module_settings.get("alert_on_levels", "medium,high,critical")
        alert_levels = {s.strip().lower() for s in alert_levels_str.split(",") if s.strip()}
        if threat_level.value in alert_levels:
            alert = Alert(
                detection_id=db_detection.id,
                company_id=camera.company_id,
                title=f"{det_type.value.replace('_', ' ').title()} – {threat_level.value.upper()}",
                message=f"{det_type.value} detected with {confidence:.0%} confidence (live)",
                status=AlertStatus.PENDING,
            )
            if live_notify and live_notify.get("user_email"):
                alert.email_sent_to = live_notify["user_email"]
            db.add(alert)
            db.flush()
            alerts_created += 1

            if live_notify and live_notify.get("user_email"):
                try:
                    from app.api.v1.video_processing import _email_queue, _ensure_email_thread

                    # Merge primary user email with extra recipients (admins + zone officer)
                    primary_email = live_notify["user_email"]
                    extra_emails: list[str] = live_notify.get("extra_emails") or []
                    seen_emails: set[str] = set()
                    all_emails: list[str] = []
                    for addr in [primary_email, *extra_emails]:
                        if addr and addr not in seen_emails:
                            seen_emails.add(addr)
                            all_emails.append(addr)

                    _ensure_email_thread()
                    _email_queue.put(
                        {
                            "alert_id": alert.id,
                            "alert_preference": live_notify.get("user_alert_preference", "email"),
                            "sms_number": live_notify.get("user_phone", "") or "",
                            "kwargs": {
                                "to_emails": all_emails,
                                "subject": alert.title,
                                "message": alert.message,
                                "snapshot_path": snapshot_path,
                                "metadata": {
                                    "Detection Type": det_type.value,
                                    "Threat Level": threat_level.value,
                                    "Confidence": f"{confidence:.2%}",
                                    "Camera": live_notify.get("camera_name", ""),
                                    "Zone": camera.zone or "—",
                                    "Timestamp": timestamp.isoformat(),
                                },
                            },
                        }
                    )
                except Exception as e:
                    logger.warning("Could not queue live alert notification: %s", e)

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

    return detections_created, alerts_created, detection_results


def normalize_stream_url(stream_url: str) -> str:
    """
    Rewrite stream URLs that OpenCV/FFmpeg cannot play as-is.

    Wowza Cloud ``entrypoint.cloud.wowza.com`` RTSP URLs often accept DESCRIBE
    but return ``SETUP 403 Forbidden`` (ingest-style endpoint). The same
    app/stream path serves HLS playback that FFmpeg can open:

      rtsp://host:1935/app/stream
        -> http://host:1935/app/stream/playlist.m3u8
    """
    from urllib.parse import urlparse

    url = (stream_url or "").strip()
    if not url:
        return url

    lower = url.lower()
    if not lower.startswith("rtsp://"):
        return url
    if "entrypoint.cloud.wowza.com" not in lower:
        return url

    parsed = urlparse(url)
    path = (parsed.path or "").rstrip("/")
    if not path or path.endswith(".m3u8"):
        return url

    hls = f"http://{parsed.netloc}{path}/playlist.m3u8"
    logger.info("Rewrote Wowza RTSP entrypoint to HLS playback URL")
    return hls


def open_stream_capture(stream_url: str, low_latency: bool = True) -> cv2.VideoCapture:
    """
    Open a live URL with settings that minimise RTSP lag.

    ``low_latency=True`` (default for live streams):
    - Buffer size 1 — always decode the newest frame, never a stale one.
    - ``fflags nobuffer`` + ``flags low_delay`` — tell FFmpeg to skip
      input buffering and reduce its internal pipeline latency.
    - ``probesize`` / ``analyzeduration`` reduced so the stream opens
      faster (typically shaves 1-3 s off initial connection time).
    """
    url = normalize_stream_url(stream_url)
    if not url:
        return cv2.VideoCapture()

    if url.lower().startswith("rtsp://"):
        opts_parts = ["rtsp_transport;tcp"]

        if low_latency:
            # Minimise FFmpeg's internal buffering for real-time playback
            opts_parts += [
                "fflags;nobuffer",  # skip input buffering
                "flags;low_delay",  # enable low-delay decoding
                "probesize;32",  # tiny probe (bytes) — fast open
                "analyzeduration;0",  # no pre-analysis delay
                "reorder_queue_size;0",  # no packet reorder buffer
            ]

        extra = str(getattr(settings, "RTSP_FFMPEG_CAPTURE_OPTIONS", "") or "").strip()
        for segment in extra.split("|"):
            seg = segment.strip()
            if seg:
                opts_parts.append(seg)

        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "|".join(opts_parts)

    lock = _url_open_lock(url)
    with lock:
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if not cap.isOpened() and "entrypoint.cloud.wowza.com" in url.lower():
            # Wowza often 403s on rapid reconnect — brief pause then one retry
            time.sleep(1.25)
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if cap.isOpened():
            # Buffer=1: reader always gets the most recent frame, not a queued-up old one
            buf = 1 if low_latency else int(getattr(settings, "RTSP_CAPTURE_BUFFER_SIZE", 2))
            buf = max(1, min(buf, 16))
            cap.set(cv2.CAP_PROP_BUFFERSIZE, buf)
        return cap


def encode_jpeg_bytes(frame: np.ndarray, quality: int = 82) -> tuple[bool, Optional[bytes]]:
    if frame is None or frame.size == 0:
        return False, None
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return False, None
    return True, buf.tobytes()


def grab_jpeg_snapshot(
    stream_url: str, timeout_sec: float = 8.0
) -> tuple[Optional[bytes], Optional[str]]:
    """
    Blocking: open RTSP/HTTP URL, read one frame, encode as JPEG.
    Returns (jpeg_bytes_or_none, error_message_or_none).
    """
    url = normalize_stream_url(stream_url)
    now = time.monotonic()
    with _snapshot_cache_lock:
        cached = _snapshot_jpeg_cache.get(url)
        if cached and now - cached[0] <= _SNAPSHOT_CACHE_TTL_SEC:
            return cached[1], None

    cap = open_stream_capture(url)
    if not cap.isOpened():
        with _snapshot_cache_lock:
            cached = _snapshot_jpeg_cache.get(url)
            if cached:
                return cached[1], None
        return None, "Could not open stream URL"

    deadline = time.monotonic() + timeout_sec
    frame = None
    try:
        while time.monotonic() < deadline:
            ok, frm = cap.read()
            if ok and frm is not None and frm.size > 0:
                frame = frm
                break
            time.sleep(0.05)
        if frame is None:
            return None, "Timed out reading frame"

        ok_j, jpeg = encode_jpeg_bytes(frame, quality=82)
        if not ok_j or not jpeg:
            return None, "JPEG encode failed"
        with _snapshot_cache_lock:
            _snapshot_jpeg_cache[url] = (time.monotonic(), jpeg)
        return jpeg, None
    finally:
        cap.release()
