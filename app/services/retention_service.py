"""
Data retention service — auto-cleanup of old detections, evidence, alerts, and files.

Implements the data retention policy from Section 9 of the proposal:
"Define retention period (e.g., 30-90 days) after which evidence is archived or deleted."
"""
import os
import logging
from datetime import datetime, timedelta
from pathlib import Path
from sqlalchemy.orm import Session

from app.config import settings
from app.models.detection import Detection
from app.models.evidence import Evidence
from app.models.alert import Alert
from app.models.alert_log import AlertLog

logger = logging.getLogger(__name__)


def run_retention_cleanup(db: Session, retention_days: int = None) -> dict:
    """
    Delete detections, evidence, alerts, and associated files older than retention_days.

    Returns a summary dict with counts of deleted records and files.
    """
    if retention_days is None:
        retention_days = settings.DATA_RETENTION_DAYS

    if retention_days <= 0:
        return {"message": "Data retention disabled (0 days)", "deleted": {}}

    cutoff = datetime.utcnow() - timedelta(days=retention_days)
    logger.info("Retention cleanup: deleting data older than %s (%d days)", cutoff.isoformat(), retention_days)

    stats = {
        "retention_days": retention_days,
        "cutoff_date": cutoff.isoformat(),
        "alert_logs_deleted": 0,
        "alerts_deleted": 0,
        "evidence_deleted": 0,
        "evidence_files_deleted": 0,
        "detections_deleted": 0,
        "processed_videos_deleted": 0,
    }

    # 1. Find old detections
    old_detections = db.query(Detection).filter(Detection.detected_at < cutoff).all()
    old_detection_ids = [d.id for d in old_detections]

    if not old_detection_ids:
        logger.info("Retention cleanup: no data older than %d days", retention_days)
        return {"message": f"No data older than {retention_days} days", "deleted": stats}

    # 2. Delete alert logs for old alerts
    old_alerts = db.query(Alert).filter(Alert.detection_id.in_(old_detection_ids)).all()
    old_alert_ids = [a.id for a in old_alerts]

    if old_alert_ids:
        log_count = db.query(AlertLog).filter(AlertLog.alert_id.in_(old_alert_ids)).delete(synchronize_session=False)
        stats["alert_logs_deleted"] = log_count

    # 3. Delete old alerts
    if old_alert_ids:
        alert_count = db.query(Alert).filter(Alert.id.in_(old_alert_ids)).delete(synchronize_session=False)
        stats["alerts_deleted"] = alert_count

    # 4. Delete evidence records + files
    old_evidence = db.query(Evidence).filter(Evidence.detection_id.in_(old_detection_ids)).all()
    files_deleted = 0
    for ev in old_evidence:
        # Delete the actual image file
        if ev.image_path:
            try:
                file_path = Path(ev.image_path)
                if file_path.exists():
                    file_path.unlink()
                    files_deleted += 1
            except Exception as e:
                logger.warning("Failed to delete evidence file %s: %s", ev.image_path, e)

    evidence_count = db.query(Evidence).filter(Evidence.detection_id.in_(old_detection_ids)).delete(synchronize_session=False)
    stats["evidence_deleted"] = evidence_count
    stats["evidence_files_deleted"] = files_deleted

    # 5. Delete old detections
    detection_count = db.query(Detection).filter(Detection.id.in_(old_detection_ids)).delete(synchronize_session=False)
    stats["detections_deleted"] = detection_count

    # 6. Clean up old processed videos
    video_dir = Path(settings.PROCESSED_VIDEO_DIR)
    if video_dir.exists():
        for f in video_dir.iterdir():
            if f.is_file():
                try:
                    file_age = datetime.utcnow() - datetime.utcfromtimestamp(f.stat().st_mtime)
                    if file_age.days > retention_days:
                        f.unlink()
                        stats["processed_videos_deleted"] += 1
                except Exception as e:
                    logger.warning("Failed to delete video %s: %s", f, e)

    db.commit()

    total = (stats["detections_deleted"] + stats["alerts_deleted"] +
             stats["evidence_deleted"] + stats["alert_logs_deleted"])
    logger.info(
        "Retention cleanup complete: %d records deleted, %d files removed",
        total, stats["evidence_files_deleted"] + stats["processed_videos_deleted"]
    )

    return {
        "message": f"Cleaned up data older than {retention_days} days",
        "deleted": stats,
    }
