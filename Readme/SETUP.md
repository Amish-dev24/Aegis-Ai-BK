# Setup Guide for Aegis AI Backend

## Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install packages
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your settings:
# - DATABASE_URL: PostgreSQL connection string
# - SECRET_KEY: Generate a secure random key
# - SMTP settings: For email alerts
# - Model paths: Point to your trained models
```

### 3. Set Up Database

**Option A: Using Docker Compose (Recommended)**
```bash
docker-compose up -d db
```

**Option B: Manual PostgreSQL Setup**
```bash
# Create database
createdb aegis_db

# Or using psql:
psql -U postgres
CREATE DATABASE aegis_db;
CREATE USER aegis_user WITH PASSWORD 'aegis_password';
GRANT ALL PRIVILEGES ON DATABASE aegis_db TO aegis_user;
```

### 4. Initialize Database Tables

The application will automatically create tables on first run, or you can use Alembic:

```bash
# Create initial migration
alembic revision --autogenerate -m "Initial migration"

# Apply migrations
alembic upgrade head
```

### 5. Create Admin User

```bash
python scripts/create_admin.py
```

Default credentials:
- Username: `admin`
- Password: `admin123`

**⚠️ Change the password immediately after first login!**

### 6. Run the Application

```bash
# Option 1: Using run.py
python run.py

# Option 2: Using uvicorn directly
uvicorn app.main:app --reload

# Option 3: Using Docker Compose
docker-compose up
```

The API will be available at:
- API: http://localhost:8000
- Docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Integrating Your Detection Models

### 1. Update DetectionService

Edit `app/services/detection_service.py` and replace placeholder methods with your actual model implementations.

**Example for YOLOv8:**
```python
from ultralytics import YOLO

def _load_models(self):
    """Load detection models."""
    self.model = YOLO(self.model_path)
    # Load other models as needed

def detect_weapons(self, frame, frame_timestamp):
    """Detect weapons using YOLOv8."""
    results = self.model(frame, conf=self.confidence_threshold)
    detections = []
    
    for result in results:
        for box in result.boxes:
            # Filter for weapon classes (adjust class IDs as needed)
            if int(box.cls) in [0, 1, 2]:  # Example: gun, knife, rifle
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                width = x2 - x1
                height = y2 - y1
                
                # Normalize coordinates (0-1)
                h, w = frame.shape[:2]
                detections.append({
                    "bbox": [x1/w, y1/h, width/w, height/h],
                    "confidence": float(box.conf),
                    "class": self.model.names[int(box.cls)]
                })
    
    return detections
```

### 2. Add Model Files

Place your trained model files in the `models/` directory:
- `yolov8_weapon.pt` - Weapon detection model
- `pose_estimation.pb` - Pose estimation model (if using TensorFlow)
- Other model files as needed

### 3. Update Configuration

In `.env`, set the correct paths:
```
MODEL_PATH=./models/yolov8_weapon.pt
POSE_MODEL_PATH=./models/pose_estimation.pb
CONFIDENCE_THRESHOLD=0.5
```

## Testing the API

### 1. Get Access Token

```bash
curl -X POST "http://localhost:8000/api/v1/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=admin123"
```

### 2. Create a Camera

```bash
curl -X POST "http://localhost:8000/api/v1/cameras" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Main Entrance",
    "location": "Building A - Entrance",
    "stream_url": "rtsp://camera-ip:554/stream",
    "zone": "Zone-1"
  }'
```

### 3. Process a Video

```bash
curl -X POST "http://localhost:8000/api/v1/detections/process-video?camera_id=1" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "video_file=@path/to/video.mp4"
```

## Production Deployment

### 1. Security Checklist

- [ ] Change `SECRET_KEY` to a strong random value
- [ ] Update `DATABASE_URL` with production credentials
- [ ] Configure TLS/SSL certificates
- [ ] Set `DEBUG=False` in production
- [ ] Configure proper CORS origins
- [ ] Set up firewall rules
- [ ] Enable rate limiting
- [ ] Set up monitoring and logging

### 2. Environment Variables for Production

```env
ENVIRONMENT=production
DEBUG=False
SECRET_KEY=<generate-strong-random-key>
DATABASE_URL=postgresql://user:pass@db-host:5432/aegis_db
SSL_CERT_PATH=/path/to/cert.pem
SSL_KEY_PATH=/path/to/key.pem
```

### 3. Using Docker in Production

```bash
# Build production image
docker build -t aegis-ai:latest .

# Run with production settings
docker run -d \
  --name aegis-api \
  -p 8000:8000 \
  -e DATABASE_URL=postgresql://... \
  -e SECRET_KEY=... \
  -v ./evidence:/app/evidence \
  aegis-ai:latest
```

### 4. AWS EC2 — Keep evidence images after restart (Option A)

Detection metadata is stored in **RDS**, but snapshot **files** live on the EC2 disk under `./evidence/`. If that folder is empty after a restart, the UI shows **404** for all images.

**One-time setup on EC2** (after `git clone`):

```bash
chmod +x scripts/setup_ec2_storage.sh scripts/verify_evidence_storage.sh
./scripts/setup_ec2_storage.sh
./deploy.sh
```

**Rules that matter:**

| Action | Disk kept? | Images after restart? |
|--------|----------|------------------------|
| **Stop** → **Start** (same instance) | Yes | Yes, if you deploy from the same folder |
| **Terminate** instance | No (unless EBS snapshot) | No — all local images lost |
| Redeploy with `./deploy.sh` | Yes | Yes — bind mounts preserve `./evidence/` |
| Fresh clone in a new directory | Empty `evidence/` | No — RDS paths point to files that are not on disk |

**Optional — dedicated EBS volume** (extra safety if root disk is small):

1. In AWS Console: EC2 → Volumes → Create volume → Attach to instance (e.g. `/dev/nvme1n1`).
2. On the instance:

```bash
sudo EBS_DEVICE=/dev/nvme1n1 ./scripts/setup_ec2_storage.sh --mount-ebs
./deploy.sh
```

**Verify after restart:**

```bash
ls evidence/ | head                    # should list .jpg files
./scripts/verify_evidence_storage.sh   # checks disk + HTTP /evidence/...
```

**If images still 404:** RDS still has old paths but files were lost (instance was terminated or deploy used a new empty folder). New detections will work; old ones cannot be recovered without a backup of `evidence/`.

## Troubleshooting

### Database Connection Issues

- Verify PostgreSQL is running: `pg_isready`
- Check connection string format
- Ensure database and user exist
- Check firewall/network settings

### Model Loading Errors

- Verify model file paths in `.env`
- Check model file formats match your code
- Ensure sufficient memory/GPU resources

### Email Not Sending

- Verify SMTP credentials
- Check firewall allows SMTP port
- For Gmail, use App Password (not regular password)
- Check spam folder

### Import Errors

- Ensure virtual environment is activated
- Verify all dependencies are installed: `pip install -r requirements.txt`
- Check Python version (3.11+)

## Next Steps

1. Integrate your detection models
2. Set up frontend dashboard (React)
3. Configure email alerts
4. Set up monitoring and logging
5. Configure backup strategy
6. Set up CI/CD pipeline

