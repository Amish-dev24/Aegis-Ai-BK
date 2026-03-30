"""
Script to delete all data from database excluding users, companies, and cameras.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.database import SessionLocal
from app.models import (
    Detection,
    Alert,
    Evidence,
    AuditLog,
    GlobalModuleSettings,
    CompanyDetectionSettings,
)
from app.models.alert_log import AlertLog


def clear_all_data():
    """Delete all data except users, companies, and cameras."""
    db = SessionLocal()
    
    try:
        print("Starting data deletion...")
        
        # Delete in order of dependencies (foreign keys first)
        tables = [
            ("AlertLog", AlertLog),
            ("Alert", Alert),
            ("Evidence", Evidence),
            ("Detection", Detection),
            ("AuditLog", AuditLog),
            ("CompanyDetectionSettings", CompanyDetectionSettings),
            ("GlobalModuleSettings", GlobalModuleSettings),
        ]
        
        for table_name, model in tables:
            count = db.query(model).count()
            if count > 0:
                print(f"Deleting {count} records from {table_name}...")
                db.query(model).delete()
                print(f"  ✓ Deleted {count} records from {table_name}")
            else:
                print(f"  - {table_name} is already empty")
        
        db.commit()
        print("\n✓ All data deleted successfully!")
        print("Preserved tables: Users, Companies, Cameras")
        
    except Exception as e:
        db.rollback()
        print(f"\n✗ Error occurred: {str(e)}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    clear_all_data()
