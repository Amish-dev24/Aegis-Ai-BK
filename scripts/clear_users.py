"""
Script to clear all users from the database.
WARNING: This will delete all users!
Run: python scripts/clear_users.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.database import SessionLocal
from app.models.user import User


def clear_users():
    """Clear all users from the database."""
    db = SessionLocal()
    try:
        users = db.query(User).all()
        user_count = len(users)

        if user_count == 0:
            print("No users to delete.")
            return True

        print("=" * 60)
        print("WARNING: This will DELETE ALL USERS in the database!")
        print("=" * 60)
        print(f"Found {user_count} user(s):")
        for user in users:
            print(f"  - {user.username} ({user.email})")

        response = input("\nAre you sure you want to delete all users? (yes/no): ")
        if response.lower() != 'yes':
            print("Cancelled.")
            return False

        print("\nDeleting all users...")
        db.query(User).delete()
        db.commit()

        print(f"[OK] Deleted {user_count} user(s)")
        print("\nYou can now register the first admin user via the API:")
        print("  POST http://localhost:8000/api/v1/auth/register")

        return True

    except Exception as e:
        print(f"Error clearing users: {e}")
        db.rollback()
        return False
    finally:
        db.close()


if __name__ == "__main__":
    success = clear_users()
    sys.exit(0 if success else 1)


