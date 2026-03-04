"""
Script to drop and recreate all database tables.
WARNING: This will delete all data in the database!
Run: python scripts/recreate_tables.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text, inspect
from app.database import engine, Base
from app.config import settings


def recreate_tables():
    """Drop and recreate all tables."""
    try:
        print("=" * 60)
        print("WARNING: This will DELETE ALL DATA in the database!")
        print("=" * 60)
        print(f"Database: {settings.DATABASE_URL.split('@')[-1] if '@' in settings.DATABASE_URL else 'N/A'}")
        
        response = input("\nAre you sure you want to continue? (yes/no): ")
        if response.lower() != 'yes':
            print("Cancelled.")
            return False
        
        print("\nDropping all tables...")
        
        # Import all models to register them
        from app.models import user, company, camera, detection, alert, evidence, audit_log
        
        # Drop all tables
        Base.metadata.drop_all(bind=engine)
        print("[OK] All tables dropped")
        
        # Create all tables
        print("Creating all tables...")
        Base.metadata.create_all(bind=engine)
        print("[OK] All tables created")
        
        # List created tables
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        print(f"\nCreated {len(tables)} tables:")
        for table in sorted(tables):
            print(f"  - {table}")
        
        print("\n[OK] Database schema recreated successfully!")
        print("\nNext step: Create admin user")
        print("  python scripts/create_admin.py")
        print("\nOr register via API:")
        print("  POST http://localhost:8000/api/v1/auth/register")
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Error recreating tables: {e}")
        print("\nPossible issues:")
        print("  1. Database doesn't exist - Run: python scripts/setup_database.py")
        print("  2. Wrong credentials - Check your .env file")
        print("  3. Database not running - Start PostgreSQL or use: docker-compose up -d db")
        return False


if __name__ == "__main__":
    success = recreate_tables()
    sys.exit(0 if success else 1)

