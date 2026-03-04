"""
Script to fix database schema by adding missing columns.
This script adds the company_id column to the users table if it doesn't exist.
Run: python scripts/fix_database_schema.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text, inspect
from app.database import engine, Base, SessionLocal
from app.config import settings


def fix_database_schema():
    """Add missing columns to existing tables."""
    try:
        print("Checking database schema...")
        print(f"Database: {settings.DATABASE_URL.split('@')[-1] if '@' in settings.DATABASE_URL else 'N/A'}")
        print()
        
        # Create all tables first (this will create companies table if it doesn't exist)
        print("Creating missing tables...")
        from app.models import user, company, camera, detection, alert, evidence, audit_log
        Base.metadata.create_all(bind=engine)
        print("[OK] Tables checked/created")
        
        # Check if company_id column exists in users table
        inspector = inspect(engine)
        users_columns = [col['name'] for col in inspector.get_columns('users')]
        
        if 'company_id' not in users_columns:
            print("\nAdding missing 'company_id' column to 'users' table...")
            with engine.connect() as conn:
                # Check if companies table exists first
                if 'companies' not in inspector.get_table_names():
                    print("  Creating companies table first...")
                    Base.metadata.tables['companies'].create(bind=engine)
                
                # Add company_id column (PostgreSQL doesn't support IF NOT EXISTS in ALTER TABLE)
                try:
                    conn.execute(text("""
                        ALTER TABLE users 
                        ADD COLUMN company_id INTEGER 
                        REFERENCES companies(id)
                    """))
                    conn.commit()
                    print("[OK] Added 'company_id' column to 'users' table")
                except Exception as e:
                    if 'already exists' in str(e).lower() or 'duplicate' in str(e).lower():
                        print("[OK] 'company_id' column already exists")
                    else:
                        raise
        else:
            print("[OK] 'company_id' column already exists in 'users' table")
        
        # Verify the fix
        print("\nVerifying schema...")
        inspector = inspect(engine)
        users_columns = [col['name'] for col in inspector.get_columns('users')]
        if 'company_id' in users_columns:
            print("[OK] Schema is correct!")
            print("\nYou can now register users via the API.")
            return True
        else:
            print("[ERROR] Schema fix failed")
            return False
            
    except Exception as e:
        print(f"\n[ERROR] Error fixing schema: {e}")
        print("\nPossible issues:")
        print("  1. Database doesn't exist - Run: python scripts/setup_database.py")
        print("  2. Wrong credentials - Check your .env file")
        print("  3. Database not running - Start PostgreSQL or use: docker-compose up -d db")
        print("\nAlternative: Drop and recreate tables (WARNING: This will delete all data!)")
        print("  python scripts/recreate_tables.py")
        return False


if __name__ == "__main__":
    success = fix_database_schema()
    sys.exit(0 if success else 1)

