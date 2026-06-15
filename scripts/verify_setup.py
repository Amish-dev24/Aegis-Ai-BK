"""
Script to verify the setup is correct.
Run: python scripts/verify_setup.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def verify_setup():
    """Verify that the setup is correct."""
    print("Verifying Aegis AI Backend Setup...")
    print("=" * 50)

    issues = []

    # Check Python version
    print("✓ Checking Python version...")
    if sys.version_info < (3, 11):
        issues.append(f"Python 3.11+ required, found {sys.version}")
    else:
        print(f"  Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

    # Check required packages
    print("\n✓ Checking required packages...")
    required_packages = [
        "fastapi",
        "sqlalchemy",
        "pydantic",
        "uvicorn",
        "jose",
        "passlib",
        "cv2",
        "numpy"
    ]

    for package in required_packages:
        try:
            if package == "cv2":
                import cv2
            elif package == "jose":
                from jose import jwt
            else:
                __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            issues.append(f"Missing package: {package}")
            print(f"  ✗ {package} (MISSING)")

    # Check database connection
    print("\n✓ Checking database connection...")
    try:
        from app.database import engine
        with engine.connect() as conn:
            print("  ✓ Database connection successful")
    except Exception as e:
        issues.append(f"Database connection failed: {e}")
        print(f"  ✗ Database connection failed: {e}")

    # Check configuration
    print("\n✓ Checking configuration...")
    try:
        from app.config import settings
        print("  ✓ Configuration loaded")
        print(f"    - Database URL: {settings.DATABASE_URL[:30]}...")
        print(f"    - Debug mode: {settings.DEBUG}")
    except Exception as e:
        issues.append(f"Configuration error: {e}")
        print(f"  ✗ Configuration error: {e}")

    # Check directories
    print("\n✓ Checking directories...")
    import os

    from app.config import settings

    dirs = [
        settings.UPLOAD_DIR,
        settings.EVIDENCE_DIR,
        "./models"
    ]

    for dir_path in dirs:
        if os.path.exists(dir_path):
            print(f"  ✓ {dir_path}")
        else:
            print(f"  ⚠ {dir_path} (will be created)")
            os.makedirs(dir_path, exist_ok=True)

    # Summary
    print("\n" + "=" * 50)
    if issues:
        print("⚠ Setup verification found issues:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nPlease fix these issues before running the application.")
        return False
    else:
        print("✓ Setup verification complete! All checks passed.")
        print("\nNext steps:")
        print("  1. Run: python scripts/create_admin.py")
        print("  2. Run: python run.py")
        print("  3. Visit: http://localhost:8000/docs")
        return True

if __name__ == "__main__":
    success = verify_setup()
    sys.exit(0 if success else 1)

