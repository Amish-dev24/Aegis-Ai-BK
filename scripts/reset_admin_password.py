"""
Script to reset admin user password.
Run: python scripts/reset_admin_password.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.database import SessionLocal
from app.models.user import User
from app.core.security import get_password_hash, verify_password


def reset_admin_password():
    """Reset admin user password."""
    db = SessionLocal()
    try:
        # Find admin user
        admin = db.query(User).filter(User.username == "admin").first()
        
        if not admin:
            print("Admin user not found!")
            return False
        
        print("Current admin user:")
        print(f"  Username: {admin.username}")
        print(f"  Email: {admin.email}")
        print(f"  Role: {admin.role.value}")
        print()
        
        # Test current password
        test_password = "admin123"
        if verify_password(test_password, admin.hashed_password):
            print(f"[OK] Current password is: {test_password}")
        else:
            print(f"[INFO] Current password is NOT: {test_password}")
        
        print()
        print("Options:")
        print("  1. Keep current password (admin123)")
        print("  2. Set new password")
        
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == "2":
            new_password = input("Enter new password: ").strip()
            if not new_password:
                print("Password cannot be empty!")
                return False
            
            admin.hashed_password = get_password_hash(new_password)
            db.commit()
            print(f"[OK] Password updated successfully!")
            print(f"New password: {new_password}")
        else:
            print("[INFO] Keeping current password")
            print(f"Password: {test_password}")
        
        print()
        print("Login credentials:")
        print(f"  Username: {admin.username}")
        print(f"  Password: {test_password if choice == '1' else new_password}")
        print()
        print("Use these credentials to login via:")
        print("  POST http://localhost:8000/api/v1/auth/login")
        print("  Form data: username=admin&password=<password>")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        db.rollback()
        return False
    finally:
        db.close()


if __name__ == "__main__":
    success = reset_admin_password()
    sys.exit(0 if success else 1)


