"""
Database connection and session management.
"""
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.config import settings

engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,   # Re-validate stale connections (important behind AWS RDS proxy / NAT)
    pool_size=10,
    max_overflow=20,
    pool_timeout=30,      # Raise after 30s waiting for a free connection instead of hanging
    pool_recycle=1800,    # Recycle connections after 30 min — AWS RDS drops idle TCP after ~8 hrs
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """Dependency for getting database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

