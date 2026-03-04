import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.database import SessionLocal
from app.models.user import User
from app.core.security import get_password_hash

def reset_admin_password():
    db = SessionLocal()
    user = db.query(User).filter(User.username == "admin").first()
    if user:
        print(f"Resetting password for {user.username}")
        user.hashed_password = get_password_hash("password123")
        db.commit()
        print("Password reset to 'password123'")
    else:
        print("User 'admin' not found")
    db.close()

if __name__ == "__main__":
    reset_admin_password()
