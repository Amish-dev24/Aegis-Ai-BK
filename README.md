# Aegis AI - Intelligent Surveillance Platform (Backend)

Intelligent surveillance platform that analyzes **uploaded videos and images** to detect security threats automatically using AI models.

> **Note:** Currently supports uploaded video files and images only. Live camera streaming (RTSP) is not yet implemented.

## Features

- **Weapon Detection** - Detects Weapons, Bags, and Boxes using YOLOv8 (`best.pt`)
- **Face Obfuscation Detection** - Detects covered/uncovered faces using YOLOv8 (`best (2).pt`)
- **Crowd Density Monitoring** - People counting using CSRNet density estimator (`csrnet_bestmodel`)
- **Abandoned Object Detection** - Tracks stationary Bags/Boxes over time using the weapon model + background subtraction
- **Violence Detection** - Placeholder (pose model not yet trained)
- **Real-time Alerts** - Auto-creates alerts for HIGH/CRITICAL threats with email notifications
- **Evidence Logging** - Stores detection snapshots, metadata, and timestamps
- **Analytics Dashboard** - Heatmaps, timelines, zone stats, threat distribution
- **Role-Based Access Control** - 4 roles: `aegis_admin`, `admin`, `security_officer`, `viewer`
- **Multi-Tenant Isolation** - Company-based data isolation across all endpoints
- **Audit Logging** - Tracks logins, user creation, alert actions, evidence exports
- **JWT Authentication** - Access token (30 min) + Refresh token (2 hours)

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | FastAPI (Python 3.11+) |
| Database | PostgreSQL 15+ |
| Cache | Redis 7+ |
| AI Models | YOLOv8 (ultralytics), CSRNet (PyTorch) |
| Auth | JWT (python-jose) + Argon2/Bcrypt password hashing |
| Email | SMTP via aiosmtplib (TLS) |
| Video | OpenCV |

## Project Structure

```
Aegis_Ai_Bk/
├── app/
│   ├── main.py                    # FastAPI app, CORS, startup events
│   ├── config.py                  # Settings from .env
│   ├── database.py                # SQLAlchemy engine & session
│   ├── api/v1/
│   │   ├── auth.py                # Login, register, refresh token
│   │   ├── users.py               # User CRUD (admin only)
│   │   ├── cameras.py             # Camera CRUD
│   │   ├── detections.py          # Create detection, process video
│   │   ├── alerts.py              # Alert CRUD, acknowledge
│   │   ├── evidence.py            # Evidence upload, download, export
│   │   ├── analytics.py           # Heatmap, timeline, zone stats
│   │   └── audit_logs.py          # Audit log listing
│   ├── models/                    # SQLAlchemy ORM models
│   │   ├── user.py                # User + Role enum
│   │   ├── camera.py
│   │   ├── detection.py
│   │   ├── alert.py
│   │   ├── evidence.py
│   │   └── audit_log.py
│   ├── schemas/                   # Pydantic request/response schemas
│   ├── services/
│   │   ├── detection_service.py   # AI model loading & inference
│   │   ├── email_service.py       # SMTP email alerts
│   │   └── video_service.py       # Video frame extraction (OpenCV)
│   └── core/
│       └── security.py            # JWT creation, password hashing, RBAC
├── models/                        # AI model weights (not in git)
│   ├── weapon_detection.pt        # YOLOv8 — Bags, Box, Weapons
│   ├── face_detection.pt          # YOLOv8 — covered, uncovered
│   └── csrnet_crowd.pth.tar       # CSRNet — crowd density
├── scripts/
│   ├── setup_database.py          # Create DB and user
│   ├── create_tables.py           # Create all tables
│   └── create_admin.py            # Create first admin user
├── uploads/                       # Uploaded video files
├── evidence/                      # Detection snapshot images
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## Setup (Local - Without Docker)

### Prerequisites

- Python 3.11+
- PostgreSQL 15+ (running locally)
- Redis 7+ (optional)

### Step 1: Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
pip install ultralytics torch torchvision argon2-cffi
```

> **Note:** `ultralytics` and `torch` are required for AI models. `argon2-cffi` is required for password hashing.

### Step 3: Configure Environment

Create a `.env` file in the project root:

```env
# Database
DATABASE_URL=postgresql://postgres:your_password@localhost:5432/aegis_db

# Security (change this in production!)
SECRET_KEY=your-random-secret-key-here

# Email (optional - for alert notifications)
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_app_password

# Redis (optional)
REDIS_URL=redis://localhost:6379/0
```

### Step 4: Setup Database

```bash
# Create the database and user
python scripts/setup_database.py

# Create all tables
python scripts/create_tables.py

# Create the first admin user
python scripts/create_admin.py
```

### Step 5: Add AI Model Files

Place your trained model files in the `models/` directory:

| File | Model | Classes |
|------|-------|---------|
| `models/weapon_detection.pt` | YOLOv8 | Bags, Box, Weapons |
| `models/face_detection.pt` | YOLOv8 | covered, uncovered |
| `models/csrnet_crowd.pth.tar` | CSRNet (VGG-16 + dilated conv) | crowd density count |

> The backend will start without models but detection features won't work until models are placed.

### Step 6: Run the Server

```bash
uvicorn app.main:app --reload --port 8000
```

The API is available at: `http://localhost:8000`
API docs (Swagger UI): `http://localhost:8000/docs`

## Setup (Docker Compose)

```bash
docker-compose up -d
```

This starts PostgreSQL, Redis, and the FastAPI app together. Place model files in the `models/` directory before processing videos.

## How It Works (Step-by-Step Usage Flow)

### Step 1: Login
```
POST /api/v1/auth/login
Body: { username, password }
```
Returns `access_token` (30 min) + `refresh_token` (2 hours). Use the access token in all subsequent requests as `Authorization: Bearer <access_token>`.

### Step 2: Create a Camera
Before uploading any video/image, you need to register a camera (this is like a "source" for detections).
```
POST /api/v1/cameras
Headers: Authorization: Bearer <access_token>
Body: { "name": "Front Gate Camera", "location": "Main Entrance", "zone": "Zone A" }
```
> `stream_url` is optional — not needed for uploaded files. Note the `camera_id` from the response.

### Step 3: Upload a Video or Image for AI Analysis

**Option A — Upload a Video:**
```
POST /api/v1/video/process?camera_id=1&generate_video=true
Headers: Authorization: Bearer <access_token>
Body: form-data → video_file: <your_video.mp4>
```
The backend will:
1. Save the video to `uploads/`
2. Extract frames using OpenCV (processes frames at the configured analysis rate)
3. Run each frame through all AI models:
   - **Weapon detection** (YOLOv8) → Weapons, Bags, Boxes
   - **Face detection** (YOLOv8) → covered/uncovered faces
   - **Crowd density** (CSRNet) → person count estimate
   - **Abandoned object** (YOLOv8 + background subtraction) → stationary Bags/Boxes for 60+ sec
4. Save each detection to the database with bounding box coordinates
5. Save evidence snapshots to `evidence/`
6. Auto-create alerts + send emails for HIGH/CRITICAL threats
7. Generate a browser-playable processed MP4 when `generate_video=true`
8. Delete the uploaded video after processing

> On Linux / Debian deployments, install `ffmpeg` so the processed MP4 can be re-encoded for browser playback.

**Option B — Upload a Single Image:**
```
POST /api/v1/detections/process-image?camera_id=1
Headers: Authorization: Bearer <access_token>
Body: form-data → image_file: <your_image.jpg>
```
Same as video but runs all models on just the one image frame.

### Step 4: View Results

**List all detections:**
```
GET /api/v1/detections
```

**View alerts (auto-created for dangerous detections):**
```
GET /api/v1/alerts
```

**Download evidence snapshots:**
```
GET /api/v1/evidence/{id}/download
```

**View analytics:**
```
GET /api/v1/analytics/heatmap
GET /api/v1/analytics/timeline
GET /api/v1/analytics/threat-distribution
GET /api/v1/analytics/top-cameras
```

### Step 5: Token Refresh
When the access token expires (30 min), call:
```
POST /api/v1/auth/refresh
Body: { "refresh_token": "<your_refresh_token>" }
```
Returns a new access token. After 2 hours (refresh token expires), user must login again.

---

## All API Endpoints

### Authentication
| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| POST | `/api/v1/auth/login` | Login (returns access + refresh token) | No |
| POST | `/api/v1/auth/refresh` | Refresh access token | No (uses refresh token) |
| POST | `/api/v1/auth/register` | Register new user | Admin only |
| GET | `/api/v1/auth/me` | Get current user info | Any |

### Users
| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| GET | `/api/v1/users` | List users | Admin |
| GET | `/api/v1/users/{id}` | Get user | Admin |
| PUT | `/api/v1/users/{id}` | Update user | Admin |
| DELETE | `/api/v1/users/{id}` | Delete user | Admin |

### Cameras
| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| POST | `/api/v1/cameras` | Create camera | Security Officer+ |
| GET | `/api/v1/cameras` | List cameras | Any |
| GET | `/api/v1/cameras/{id}` | Get camera | Any |
| PUT | `/api/v1/cameras/{id}` | Update camera | Security Officer+ |
| DELETE | `/api/v1/cameras/{id}` | Delete camera | Security Officer+ |

### Detections
| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| POST | `/api/v1/detections` | Create detection manually | Security Officer+ |
| GET | `/api/v1/detections` | List detections (filterable) | Any |
| GET | `/api/v1/detections/{id}` | Get detection details | Any |
| POST | `/api/v1/detections/process-video` | Upload & process video with AI | Security Officer+ |
| POST | `/api/v1/detections/process-image` | Upload & process image with AI | Security Officer+ |
| GET | `/api/v1/detections/stats/summary` | Detection statistics | Any |

### Alerts
| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| POST | `/api/v1/alerts` | Create alert | Security Officer+ |
| GET | `/api/v1/alerts` | List alerts | Any |
| GET | `/api/v1/alerts/{id}` | Get alert | Any |
| PUT | `/api/v1/alerts/{id}` | Update alert | Security Officer+ |
| POST | `/api/v1/alerts/{id}/acknowledge` | Acknowledge alert | Security Officer+ |

### Evidence
| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| POST | `/api/v1/evidence` | Upload evidence | Security Officer+ |
| GET | `/api/v1/evidence` | List evidence | Any |
| GET | `/api/v1/evidence/{id}` | Get evidence metadata | Any |
| GET | `/api/v1/evidence/{id}/download` | Download evidence image | Any |
| POST | `/api/v1/evidence/export` | Export evidence as CSV | Admin |

### Analytics
| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| GET | `/api/v1/analytics/heatmap` | Detection heatmap data | Any |
| GET | `/api/v1/analytics/timeline` | Detection timeline | Any |
| GET | `/api/v1/analytics/by-zone` | Detections grouped by zone | Any |
| GET | `/api/v1/analytics/threat-distribution` | Threat level breakdown | Any |
| GET | `/api/v1/analytics/top-cameras` | Top cameras by detection count | Any |

## Roles & Permissions

| Role | Scope | Permissions |
|------|-------|-------------|
| `aegis_admin` | Platform-wide | Full access to all companies and resources |
| `admin` | Company-level | Manage users, cameras, settings within their company |
| `security_officer` | Company-level | Create/manage detections, alerts, cameras |
| `viewer` | Company-level | Read-only access to detections, alerts, analytics |

## Security Notes

- **SECRET_KEY**: Used internally by the server to sign JWT tokens. Not an API key. Change it in `.env` before deploying to production.
- **Passwords**: Hashed with Argon2 (with Bcrypt fallback). Never stored in plain text.
- **CORS**: Restricted to `ALLOWED_ORIGINS` in production mode.
- **Docs**: Swagger UI is disabled when `ENVIRONMENT=production`.
- **Error details**: Internal error messages are hidden in production.
- **Audit logging**: All sensitive actions (login, user creation, alert changes, evidence export) are logged.

## Useful Scripts

```bash
python scripts/setup_database.py    # Create database and user
python scripts/create_tables.py     # Create all tables
python scripts/create_admin.py      # Create first admin user
python scripts/reset_admin_pass.py  # Reset admin password
python scripts/check_users.py       # List all users in DB
python scripts/verify_setup.py      # Verify DB and tables exist
```
