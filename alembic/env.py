from logging.config import fileConfig
from sqlalchemy import engine_from_config
from sqlalchemy import pool
from alembic import context
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.database import Base
from app.config import settings

# this is the Alembic Config object
config = context.config

# Override sqlalchemy.url with settings
config.set_main_option("sqlalchemy.url", settings.DATABASE_URL)

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Import all models so Alembic can detect them
from app.models import user, company, camera, detection, alert, evidence, audit_log, detection_settings

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    try:
        connectable = engine_from_config(
            config.get_section(config.config_ini_section, {}),
            prefix="sqlalchemy.",
            poolclass=pool.NullPool,
        )

        with connectable.connect() as connection:
            context.configure(
                connection=connection, target_metadata=target_metadata
            )

            with context.begin_transaction():
                context.run_migrations()
    except Exception as e:
        error_msg = str(e)
        if "password authentication failed" in error_msg or "could not connect" in error_msg.lower():
            print("\n" + "="*60)
            print("DATABASE CONNECTION ERROR")
            print("="*60)
            print("\nCannot connect to the database. Please ensure:")
            print("  1. PostgreSQL is running")
            print("  2. Database exists and credentials are correct")
            print("  3. Check your .env file DATABASE_URL setting")
            print("\nTo set up the database:")
            print("  Option 1: Use Docker Compose")
            print("    docker-compose up -d db")
            print("\n  Option 2: Create database manually")
            print("    createdb aegis_db")
            print("    # Or use: python scripts/setup_database.py")
            print("\n  Option 3: Use SQLAlchemy to create tables (no migrations)")
            print("    python scripts/create_tables.py")
            print("="*60 + "\n")
        raise


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

