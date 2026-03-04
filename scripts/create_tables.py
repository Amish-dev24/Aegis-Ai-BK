"""
Script to create database tables using SQLAlchemy (without Alembic).
Use this if you don't want to use migrations initially.
Run: python scripts/create_tables.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.database import Base, engine
from app.models import user, camera, detection, alert, evidence, audit_log
from app.config import settings


def create_tables():
    """Create all database tables."""
    try:
        print("Creating database tables...")
        print(f"Database: {settings.DATABASE_URL.split('@')[-1] if '@' in settings.DATABASE_URL else 'N/A'}")
        print()
        
        # Import all models to register them
        from app.models import (
            User, Camera, Detection, Alert, Evidence, AuditLog
        )
        
        # Create all tables
        Base.metadata.create_all(bind=engine)
        
        print("OK: All tables created successfully!")
        print("\nCreated tables:")
        for table in Base.metadata.tables:
            print(f"  - {table}")
        
        print("\nNext step: Create admin user")
        print("  python scripts/create_admin.py")
        
        return True
        
    except Exception as e:
        print(f"\nERROR: Error creating tables: {e}")
        print("\nPossible issues:")
        print("  1. Database doesn't exist - Run: python scripts/setup_database.py")
        print("  2. Wrong credentials - Check your .env file")
        print("  3. Database not running - Start PostgreSQL or use: docker-compose up -d db")
        return False


if __name__ == "__main__":
    success = create_tables()
    sys.exit(0 if success else 1)

