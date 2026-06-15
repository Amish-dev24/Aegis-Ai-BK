import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models.alert import Alert
from app.models.audit_log import AuditLog
from app.models.camera import Camera
from app.models.company import Company
from app.models.detection import Detection
from app.models.evidence import Evidence
from app.models.user import Role, User

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def reset_db():
    db: Session = SessionLocal()
    try:
        logger.info("Starting database reset...")

        # 1. Identify Aegis Admin
        admin = db.query(User).filter(User.role == Role.AEGIS_ADMIN).first()
        admin_id = admin.id if admin else None

        if not admin:
            logger.warning("No AEGIS_ADMIN found. Database will be completely wiped.")
        else:
            logger.info(f"Preserving AEGIS_ADMIN: {admin.username} (ID: {admin.id})")
            # Ensure admin is not linked to any company (to allow company deletion)
            if admin.company_id is not None:
                logger.info("Unlinking admin from company before deletion...")
                admin.company_id = None
                db.add(admin)
                db.commit()

        # 2. Delete Data in Order (Child -> Parent)

        # Evidence (links to Detection)
        deleted = db.query(Evidence).delete()
        logger.info(f"Deleted {deleted} Evidence records.")

        # Alerts (links to Detection, Company)
        deleted = db.query(Alert).delete()
        logger.info(f"Deleted {deleted} Alert records.")

        # Detections (links to Camera, Company)
        deleted = db.query(Detection).delete()
        logger.info(f"Deleted {deleted} Detection records.")

        # Cameras (links to Company)
        deleted = db.query(Camera).delete()
        logger.info(f"Deleted {deleted} Camera records.")

        # Audit Logs (links to User)
        if admin_id:
             deleted = db.query(AuditLog).filter(AuditLog.user_id != admin_id).delete()
             logger.info(f"Deleted {deleted} AuditLog records (kept admin's).")
        else:
             deleted = db.query(AuditLog).delete()
             logger.info(f"Deleted {deleted} AuditLog records.")

        # Users (links to Company)
        if admin_id:
            deleted = db.query(User).filter(User.id != admin_id).delete()
            logger.info(f"Deleted {deleted} User records (kept admin).")
        else:
            deleted = db.query(User).delete()
            logger.info(f"Deleted {deleted} User records.")

        # Companies
        deleted = db.query(Company).delete()
        logger.info(f"Deleted {deleted} Company records.")

        db.commit()
        logger.info("Database reset complete. Aegis Admin preserved.")

    except Exception as e:
        logger.error(f"Error resetting database: {e}")
        db.rollback()
        raise
    finally:
        db.close()

if __name__ == "__main__":
    reset_db()
