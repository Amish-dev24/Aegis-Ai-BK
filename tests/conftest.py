"""
Pytest configuration and shared fixtures for Aegis AI backend tests.

Strategy
--------
* Set DATABASE_URL to SQLite *before* any app module is imported so that
  pydantic-settings picks it up at Settings() instantiation time.
* Replace app.database.engine/SessionLocal with a SQLite-safe engine
  (check_same_thread=False) before app.main is imported.
* Override the FastAPI get_db dependency so every endpoint uses the test
  session factory.
* Model files are set to non-existent paths so _load_models logs warnings
  and leaves all model references as None — no AI inference in tests.
"""

# ── 1. Environment variables must be set before any app import ──────────────
import os

os.environ.setdefault("DATABASE_URL", "sqlite:///./test_aegis.db")
os.environ.setdefault("SECRET_KEY", "aegis-test-secret-key-do-not-use-in-production")
os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("DATA_RETENTION_DAYS", "0")  # disable retention cleanup
os.environ.setdefault("MODEL_PATH", "./models/fake.pt")
os.environ.setdefault("FACE_MODEL_PATH", "./models/fake_face.pt")
os.environ.setdefault("CROWD_MODEL_PATH", "./models/fake_crowd.onnx")
os.environ.setdefault("VIOLENCE_MODEL_PT_PATH", "./models/fake_violence.pt")

# ── 2. Import app.database and replace the engine before app.main loads ─────
from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

import app.database as _appdb  # noqa: E402

_TEST_DB_URL = "sqlite:///./test_aegis.db"
_test_engine = create_engine(
    _TEST_DB_URL,
    connect_args={"check_same_thread": False},
)
_TestSession = sessionmaker(autocommit=False, autoflush=False, bind=_test_engine)

# Replace module-level singletons before app.main imports them
_appdb.engine = _test_engine
_appdb.SessionLocal = _TestSession

# ── 3. Now it is safe to import the app ─────────────────────────────────────
from datetime import datetime, timedelta  # noqa: E402

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.core.security import create_access_token, get_password_hash  # noqa: E402
from app.database import Base, get_db  # noqa: E402
from app.main import app  # noqa: E402
from app.models.alert import Alert, AlertStatus  # noqa: E402
from app.models.camera import Camera  # noqa: E402
from app.models.company import Company  # noqa: E402
from app.models.detection import Detection, DetectionType, ThreatLevel  # noqa: E402
from app.models.user import Role, User  # noqa: E402

# ── 5. Override get_db dependency ────────────────────────────────────────────


def _override_get_db():
    db = _TestSession()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = _override_get_db

# ── 6. Database lifecycle ─────────────────────────────────────────────────────


@pytest.fixture(scope="session", autouse=True)
def setup_database():
    """Create all tables once for the test session, drop them after."""
    Base.metadata.create_all(bind=_test_engine)
    yield
    Base.metadata.drop_all(bind=_test_engine)
    try:
        if os.path.exists("./test_aegis.db"):
            os.remove("./test_aegis.db")
    except OSError:
        pass  # Windows may lock the file briefly; ignore


@pytest.fixture(scope="session")
def db(setup_database):
    """Shared DB session for seed fixtures (session scope = created once)."""
    session = _TestSession()
    yield session
    session.close()


# ── 7. Seed fixtures ─────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def test_company(db):
    company = Company(
        name="AcmeSecurity Ltd",
        domain="acmesecurity.example",
        contact_email="admin@acmesecurity.example",
        is_active=True,
        is_verified=True,
    )
    db.add(company)
    db.commit()
    db.refresh(company)
    return company


@pytest.fixture(scope="session")
def aegis_admin(db, setup_database):
    user = User(
        username="aegis_admin_ci",
        email="aegis@aegisai.example",
        hashed_password=get_password_hash("AegisAdmin1!"),
        full_name="Aegis Platform Admin",
        role=Role.AEGIS_ADMIN,
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@pytest.fixture(scope="session")
def company_admin(db, test_company):
    user = User(
        username="company_admin_ci",
        email="cadmin@acmesecurity.example",
        hashed_password=get_password_hash("CompanyAdmin1!"),
        full_name="Company Admin",
        role=Role.ADMIN,
        company_id=test_company.id,
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@pytest.fixture(scope="session")
def officer(db, test_company):
    user = User(
        username="officer_ci",
        email="officer@acmesecurity.example",
        hashed_password=get_password_hash("Officer1!"),
        full_name="Security Officer",
        role=Role.SECURITY_OFFICER,
        company_id=test_company.id,
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@pytest.fixture(scope="session")
def viewer(db, test_company):
    user = User(
        username="viewer_ci",
        email="viewer@acmesecurity.example",
        hashed_password=get_password_hash("Viewer1!"),
        full_name="Read-Only Viewer",
        role=Role.VIEWER,
        company_id=test_company.id,
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


# ── 8. Token helpers ─────────────────────────────────────────────────────────


def _token(user: User) -> str:
    return create_access_token(
        data={"sub": user.username, "role": user.role.value},
        expires_delta=timedelta(minutes=60),
    )


@pytest.fixture(scope="session")
def aegis_admin_headers(aegis_admin):
    return {"Authorization": f"Bearer {_token(aegis_admin)}"}


@pytest.fixture(scope="session")
def admin_headers(company_admin):
    return {"Authorization": f"Bearer {_token(company_admin)}"}


@pytest.fixture(scope="session")
def officer_headers(officer):
    return {"Authorization": f"Bearer {_token(officer)}"}


@pytest.fixture(scope="session")
def viewer_headers(viewer):
    return {"Authorization": f"Bearer {_token(viewer)}"}


# ── 9. Camera fixture ─────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def test_camera(db, test_company):
    cam = Camera(
        name="Main Gate Camera",
        location="Front entrance",
        zone="Zone-A",
        company_id=test_company.id,
        is_active=True,
        latitude=31.5204,
        longitude=74.3587,
    )
    db.add(cam)
    db.commit()
    db.refresh(cam)
    return cam


# ── 10. Detection + alert fixtures ───────────────────────────────────────────


@pytest.fixture(scope="session")
def test_detection(db, test_company, test_camera):
    det = Detection(
        camera_id=test_camera.id,
        company_id=test_company.id,
        detection_type=DetectionType.WEAPON,
        threat_level=ThreatLevel.HIGH,
        confidence=0.87,
        frame_timestamp=datetime.utcnow(),
        bbox_x=0.1,
        bbox_y=0.2,
        bbox_width=0.15,
        bbox_height=0.2,
    )
    db.add(det)
    db.commit()
    db.refresh(det)
    return det


@pytest.fixture(scope="session")
def test_alert(db, test_detection, test_company):
    alert = Alert(
        detection_id=test_detection.id,
        company_id=test_company.id,
        title="Weapon detected at Main Gate",
        message="YOLOv8 detected a weapon with 87% confidence.",
        status=AlertStatus.PENDING,
        email_sent=False,
    )
    db.add(alert)
    db.commit()
    db.refresh(alert)
    return alert


# ── 11. Test client ───────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def client(setup_database):
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c
