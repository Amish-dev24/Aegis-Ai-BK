# Aegis AI — Code Fixes Applied (Proposal Alignment)

This document summarizes all issues found in the codebase when compared against the
**Proposal Defence** document, and the fixes that were applied.

---

## 1. Detection Service — All 5 AI Modules Were Placeholders

**File:** `app/services/detection_service.py`

**Problem:**
Every detection method (`detect_weapons`, `detect_violence`, `detect_abandoned_object`,
`detect_mask_face`, `calculate_crowd_density`) was a stub returning empty results.
The proposal promises YOLOv8, MediaPipe/OpenPose, MOG2 background subtraction, etc.

**Fix:**

| Module | Implementation |
|---|---|
| Weapon Detection | YOLOv8 model loading via `ultralytics.YOLO`, real inference with bounding-box normalization and weapon-class filtering |
| Violence Detection | MediaPipe Pose estimation + optical-flow motion magnitude heuristic (ready to swap in LSTM/1D-CNN when trained weights are available) |
| Abandoned Object Detection | OpenCV MOG2 background subtractor + contour tracking with configurable time threshold (default 60 s) |
| Mask / Face Obfuscation | Haar cascade face detection + skin-colour ratio heuristic to flag covered faces |
| Crowd Density | YOLOv8 person counting with HOG fallback; density normalized to 0–1 |
| Threat Classification | Improved thresholds — violence can now reach CRITICAL; abandoned objects have variable threat levels instead of flat MEDIUM |

---

## 2. Video Processing — Only Ran Weapon Detection

**File:** `app/api/v1/detections.py` → `POST /process-video`

**Problem:**
The `process_video` endpoint only called `detect_weapons()`. The proposal states all
5 detection modules should run on every frame.

**Fix:**
The endpoint now runs all 5 detection modules per frame:
1. Weapon detection (YOLOv8)
2. Violence / aggression detection (MediaPipe + optical flow)
3. Abandoned object detection (MOG2 background subtraction)
4. Mask / face obfuscation detection (Haar + skin ratio)
5. Crowd density monitoring (YOLOv8 person count)

A sliding window of the last 10 frames is maintained for temporal analysis.

---

## 3. No Evidence Snapshots Saved During Video Processing

**File:** `app/api/v1/detections.py`

**Problem:**
When processing a video, detections were created in the database but no evidence
snapshots were saved, even though `video_service.save_snapshot()` existed.

**Fix:**
Every detection now triggers `video_service.save_snapshot()` and creates an `Evidence`
record linked to the detection, storing the frame as a JPEG with metadata.

---

## 4. Auto-Created Alerts Did Not Send Emails

**Files:** `app/api/v1/detections.py`

**Problem:**
When `create_detection` or `process_video` auto-created alerts for HIGH/CRITICAL
threats, no email was sent. The proposal requires instant email notifications.

**Fix:**
Both `create_detection` and `process_video` now call `email_service.send_alert_email()`
for HIGH/CRITICAL threats, including snapshot attachment and full metadata
(detection type, threat level, confidence, camera name, timestamp).

---

## 5. Analytics Endpoints Missing Tenant Isolation

**File:** `app/api/v1/analytics.py`

**Problem:**
All 5 analytics endpoints (heatmap, timeline, by-zone, threat-distribution,
top-cameras) returned data from **all companies**, violating the RBAC requirement.

**Fix:**
Every endpoint now calls `get_user_company_filter()` and applies `Detection.company_id`
filtering. AEGIS_ADMIN users still see all data; company users see only their own.

---

## 6. Evidence Endpoints Missing Tenant Isolation

**File:** `app/api/v1/evidence.py`

**Problem:**
Any authenticated user could view, download, or export evidence from any company.

**Fix:**

| Endpoint | Change |
|---|---|
| `POST /evidence` | Checks company access on parent detection before allowing upload |
| `GET /evidence` | Joins Detection table and filters by company |
| `GET /evidence/{id}` | Checks company access before returning |
| `GET /evidence/{id}/download` | Checks company access before serving file |
| `POST /evidence/export` | Filters detection IDs by company before exporting CSV |

---

## 7. Video Timestamps Were Incorrect

**File:** `app/services/video_service.py`

**Problem:**
`read_video_file()` used `datetime.now()` for every frame, meaning all frames got
near-identical timestamps instead of reflecting their actual position in the video.

**Fix:**
Timestamps are now calculated as `video_start + (frame_count / fps)` using the
video's actual FPS, preserving correct temporal ordering.

---

## 8. SMTP TLS Misconfiguration

**File:** `app/services/email_service.py`

**Problem:**
The email service used `use_tls=True` for port 587. Port 587 requires **STARTTLS**
(upgrade plaintext → TLS), not implicit TLS (which is port 465).

**Fix:**
- Port 587 → `start_tls=True`, `use_tls=False`
- Port 465 → `use_tls=True`, `start_tls=False`
- Automatically selected based on the configured `SMTP_PORT`.

---

## 9. Security Hardening

**Files:** `app/main.py`, `app/config.py`

### 9a. Internal Error Leaking
**Problem:** The global exception handler returned `str(exc)` to the client, potentially
exposing stack traces, file paths, and database details.

**Fix:** In production (`DEBUG=False`), responses now return a generic
`"Internal server error"` message. Full details are only shown in debug mode.

### 9b. CORS Wide Open
**Problem:** `allow_origins=["*"]` was hardcoded.

**Fix:** In production, origins are read from a new `ALLOWED_ORIGINS` env var
(comma-separated). `*` is only used when `DEBUG=True`.

### 9c. Weak Secret Key
**Problem:** `SECRET_KEY` defaulted to `"your-secret-key-change-in-production"` with
no warning.

**Fix:** A startup warning is now logged when the default key is detected.

### 9d. API Docs Exposed in Production
**Fix:** `/docs` and `/redoc` are disabled when `DEBUG=False`.

---

## 10. Missing Audit Logging on Key Actions

**Files:** `app/api/v1/detections.py`, `app/api/v1/alerts.py`, `app/api/v1/evidence.py`

**Problem:**
Auth and user endpoints had audit logging, but detections, alerts, and evidence
operations did not — violating the proposal's audit logging requirement.

**Fix:**

| Action | Audit Entry |
|---|---|
| `create_detection` | Logs detection type and threat level |
| `process_video` | Logs filename, detection count, alert count |
| `create_alert` | Logs detection ID and alert title |
| `update_alert` | Logs new status |
| `acknowledge_alert` | Logs acknowledgment |
| `export_evidence` | Logs exported detection IDs |

---

## 11. Temporary Upload Files Never Cleaned Up

**File:** `app/api/v1/detections.py`

**Problem:**
`process_video` saved the uploaded file to `./uploads/` but never deleted it after
processing, causing disk bloat over time.

**Fix:**
A `try/finally` block ensures the temporary file is deleted after processing,
even if an error occurs.

---

## New Configuration Options

| Setting | Default | Purpose |
|---|---|---|
| `ALLOWED_ORIGINS` | `""` | Comma-separated CORS origins for production |

---

## Dependencies Required

The following packages must be installed for the new detection features:

```bash
pip install ultralytics mediapipe
```

- `ultralytics` — YOLOv8 model loading and inference
- `mediapipe` — Pose estimation for violence detection

Both degrade gracefully: if not installed, their respective detection modules
are disabled with a logged warning.
