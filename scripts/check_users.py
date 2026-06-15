"""
Script to check existing users in the database.
Run: python scripts/check_users.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.database import SessionLocal
from app.models.user import User


def check_users():
    """Check existing users in the database."""
    db = SessionLocal()
    try:
        users = db.query(User).all()
        user_count = len(users)

        print(f"Found {user_count} user(s) in the database:")
        print()

        if user_count == 0:
            print("No users found. You can register the first admin user via the API.")
        else:
            for user in users:
                print(f"  - ID: {user.id}")
                print(f"    Username: {user.username}")
                print(f"    Email: {user.email}")
                print(f"    Role: {user.role.value}")
                print(f"    Company ID: {user.company_id}")
                print(f"    Active: {user.is_active}")
                print()

            print("\nSince users exist, you need admin authentication to register new users.")
            print("You can either:")
            print("  1. Login with an existing admin account")
            print("  2. Delete existing users (if they're test data)")
            print("     Run: python scripts/clear_users.py")

        return True

    except Exception as e:
        print(f"Error checking users: {e}")
        return False
    finally:
        db.close()


if __name__ == "__main__":
    check_users()


