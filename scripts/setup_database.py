"""
Script to set up the database (create database and user if needed).
Run: python scripts/setup_database.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from app.config import settings
import re


def parse_database_url(url: str) -> dict:
    """Parse PostgreSQL connection URL."""
    # Format: postgresql://user:password@host:port/database
    pattern = r'postgresql://([^:]+):([^@]+)@([^:]+):(\d+)/(.+)'
    match = re.match(pattern, url)
    if not match:
        raise ValueError(f"Invalid DATABASE_URL format: {url}")
    
    return {
        'user': match.group(1),
        'password': match.group(2),
        'host': match.group(3),
        'port': match.group(4),
        'database': match.group(5)
    }


def setup_database():
    """Set up the database and user."""
    try:
        db_config = parse_database_url(settings.DATABASE_URL)
        
        print("Setting up database...")
        print(f"  Host: {db_config['host']}")
        print(f"  Port: {db_config['port']}")
        print(f"  Database: {db_config['database']}")
        print(f"  User: {db_config['user']}")
        print()
        
        # Connect to PostgreSQL server (default database)
        try:
            conn = psycopg2.connect(
                host=db_config['host'],
                port=db_config['port'],
                user='postgres',  # Try with default postgres user
                password=input("Enter PostgreSQL 'postgres' user password (or press Enter to skip): ").strip() or None
            )
        except psycopg2.OperationalError:
            print("\n⚠️  Could not connect as 'postgres' user.")
            print("   Please create the database manually:")
            print(f"   1. Connect to PostgreSQL: psql -U postgres")
            print(f"   2. Run: CREATE DATABASE {db_config['database']};")
            print(f"   3. Run: CREATE USER {db_config['user']} WITH PASSWORD '{db_config['password']}';")
            print(f"   4. Run: GRANT ALL PRIVILEGES ON DATABASE {db_config['database']} TO {db_config['user']};")
            return False
        
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Check if database exists
        cursor.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s",
            (db_config['database'],)
        )
        if cursor.fetchone():
            print(f"✓ Database '{db_config['database']}' already exists")
        else:
            print(f"Creating database '{db_config['database']}'...")
            cursor.execute(f'CREATE DATABASE {db_config["database"]}')
            print(f"✓ Database '{db_config['database']}' created")
        
        # Check if user exists
        cursor.execute(
            "SELECT 1 FROM pg_user WHERE usename = %s",
            (db_config['user'],)
        )
        if cursor.fetchone():
            print(f"✓ User '{db_config['user']}' already exists")
        else:
            print(f"Creating user '{db_config['user']}'...")
            cursor.execute(
                f"CREATE USER {db_config['user']} WITH PASSWORD %s",
                (db_config['password'],)
            )
            print(f"✓ User '{db_config['user']}' created")
        
        # Grant privileges
        cursor.execute(
            f"GRANT ALL PRIVILEGES ON DATABASE {db_config['database']} TO {db_config['user']}"
        )
        print(f"✓ Privileges granted")
        
        cursor.close()
        conn.close()
        
        print("\n✓ Database setup complete!")
        print("\nNext steps:")
        print("  1. Run: python scripts/create_tables.py  (to create tables)")
        print("  2. Run: python scripts/create_admin.py  (to create admin user)")
        print("  3. Or use Alembic: alembic revision --autogenerate -m 'Initial migration'")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error setting up database: {e}")
        print("\nAlternative: Use Docker Compose")
        print("  docker-compose up -d db")
        return False


if __name__ == "__main__":
    setup_database()


