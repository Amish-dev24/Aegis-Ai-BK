# Quick Start - Database Setup

## Your Database Credentials
- **Database Name**: `aegis_db`
- **User**: `postgres` (default PostgreSQL user)
- **Password**: `1234`

## Step 1: Create .env File

Create a file named `.env` in the project root (`F:\Aegis_Ai_Bk\.env`) with this content:

```env
# Database Configuration
DATABASE_URL=postgresql://postgres:1234@localhost:5432/aegis_db

# Security
SECRET_KEY=your-secret-key-change-in-production-make-it-long-and-random
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Email Configuration (optional for now)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=
SMTP_PASSWORD=
SMTP_FROM_EMAIL=noreply@aegisai.com
SMTP_FROM_NAME=Aegis AI Security System

# Detection Model Configuration
MODEL_PATH=./models/yolov8_weapon.pt
POSE_MODEL_PATH=./models/pose_estimation.pb
CONFIDENCE_THRESHOLD=0.5
ABANDONED_OBJECT_THRESHOLD_SECONDS=60

# File Storage
UPLOAD_DIR=./uploads
EVIDENCE_DIR=./evidence
MAX_FILE_SIZE_MB=500

# Redis (optional)
REDIS_URL=redis://localhost:6379/0

# Application
API_V1_PREFIX=/api/v1
DEBUG=True
ENVIRONMENT=development
```

## Step 2: Test Database Connection

```bash
python -c "from app.database import engine; conn = engine.connect(); print('✓ Database connection successful!'); conn.close()"
```

## Step 3: Create Tables

```bash
python scripts/create_tables.py
```

## Step 4: Create Admin User

```bash
python scripts/create_admin.py
```

## Step 5: Run the Application

```bash
python run.py
```

Visit: http://localhost:8000/docs

## Troubleshooting

If you get connection errors:

1. **Verify PostgreSQL is running:**
   ```powershell
   Get-Service postgresql*
   ```

2. **Test connection manually:**
   ```bash
   psql -U postgres -d aegis_db
   # Enter password: 1234
   ```

3. **Check if database exists:**
   ```bash
   psql -U postgres -c "\l" | findstr aegis_db
   ```

If the database doesn't exist, create it:
```bash
psql -U postgres
CREATE DATABASE aegis_db;
\q
```


