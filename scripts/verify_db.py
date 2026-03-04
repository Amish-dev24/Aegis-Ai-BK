"""
Quick script to verify database setup.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.database import SessionLocal
from app.models import User

db = SessionLocal()
try:
    admin = db.query(User).filter(User.username == "admin").first()
    if admin:
        print("SUCCESS: Database is set up correctly!")
        print(f"Admin user exists: {admin.username}")
        print(f"Email: {admin.email}")
        print(f"Role: {admin.role.value}")
        print("\nYou can now start the application:")
        print("  python run.py")
    else:
        print("WARNING: Admin user not found. Run:")
        print("  python scripts/create_admin.py")
except Exception as e:
    print(f"ERROR: {e}")
finally:
    db.close()


