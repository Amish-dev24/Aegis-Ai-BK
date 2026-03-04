# Database Setup Guide

## Quick Solutions

### Option 1: Use Docker Compose (Easiest) ⭐ Recommended

```bash
# Start PostgreSQL database
docker-compose up -d db

# Wait a few seconds for database to be ready, then create tables
python scripts/create_tables.py

# Create admin user
python scripts/create_admin.py
```

### Option 2: Manual PostgreSQL Setup

#### Step 1: Install PostgreSQL
- Download from: https://www.postgresql.org/download/
- Or use: `choco install postgresql` (Windows with Chocolatey)

#### Step 2: Create Database and User

**Using psql command line:**
```bash
# Connect to PostgreSQL
psql -U postgres

# Run these commands:
CREATE DATABASE aegis_db;
CREATE USER aegis_user WITH PASSWORD 'aegis_password';
GRANT ALL PRIVILEGES ON DATABASE aegis_db TO aegis_user;
\q
```

**Or use the setup script:**
```bash
python scripts/setup_database.py
```

#### Step 3: Create Tables

**Option A: Using SQLAlchemy (Simple)**
```bash
python scripts/create_tables.py
```

**Option B: Using Alembic (Recommended for production)**
```bash
# Generate migration
alembic revision --autogenerate -m "Initial migration"

# Apply migration
alembic upgrade head
```

#### Step 4: Create Admin User
```bash
python scripts/create_admin.py
```

### Option 3: Use SQLite for Development (No Setup Required)

If you just want to test quickly, you can use SQLite instead of PostgreSQL:

1. Edit `.env` file:
```env
DATABASE_URL=sqlite:///./aegis.db
```

2. Create tables:
```bash
python scripts/create_tables.py
```

3. Create admin user:
```bash
python scripts/create_admin.py
```

**Note:** SQLite has limitations and is not recommended for production.

## Troubleshooting

### Error: "password authentication failed"

**Causes:**
- Wrong password in `.env` file
- User doesn't exist in PostgreSQL
- PostgreSQL authentication method doesn't allow password login

**Solutions:**

1. **Check your `.env` file:**
   ```env
   DATABASE_URL=postgresql://aegis_user:aegis_password@localhost:5432/aegis_db
   ```
   Make sure the password matches what you set in PostgreSQL.

2. **Reset PostgreSQL user password:**
   ```bash
   psql -U postgres
   ALTER USER aegis_user WITH PASSWORD 'aegis_password';
   ```

3. **Check PostgreSQL authentication:**
   - Edit `pg_hba.conf` (usually in PostgreSQL data directory)
   - Ensure line for local connections uses `md5` or `scram-sha-256`:
     ```
     host    all             all             127.0.0.1/32            md5
     ```
   - Restart PostgreSQL service

### Error: "could not connect to server"

**Causes:**
- PostgreSQL is not running
- Wrong host/port in connection string
- Firewall blocking connection

**Solutions:**

1. **Check if PostgreSQL is running:**
   ```bash
   # Windows
   Get-Service postgresql*
   
   # Linux/Mac
   sudo systemctl status postgresql
   ```

2. **Start PostgreSQL:**
   ```bash
   # Windows
   Start-Service postgresql-x64-15  # Adjust version number
   
   # Linux/Mac
   sudo systemctl start postgresql
   ```

3. **Verify connection:**
   ```bash
   psql -U aegis_user -d aegis_db -h localhost
   ```

### Error: "database does not exist"

**Solution:**
```bash
# Create the database
psql -U postgres
CREATE DATABASE aegis_db;
\q
```

Or use the setup script:
```bash
python scripts/setup_database.py
```

## Verification

After setup, verify everything works:

```bash
# Test database connection
python -c "from app.database import engine; engine.connect(); print('✓ Database connection successful')"

# Test table creation
python scripts/create_tables.py

# Test admin user creation
python scripts/create_admin.py

# Start the application
python run.py
```

## Environment Variables

Make sure your `.env` file has the correct database URL:

```env
# PostgreSQL (Production/Development)
DATABASE_URL=postgresql://username:password@localhost:5432/database_name

# SQLite (Development only)
DATABASE_URL=sqlite:///./aegis.db
```

## Next Steps

After database setup:
1. ✅ Create tables: `python scripts/create_tables.py`
2. ✅ Create admin user: `python scripts/create_admin.py`
3. ✅ Start application: `python run.py`
4. ✅ Visit API docs: http://localhost:8000/docs


