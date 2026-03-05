"""
Detection service for integrating with AI models.

Models used:
- weapon_detection.pt  (YOLOv8) — classes: {0: Bags, 1: Box, 2: Weapons}
- face_detection.pt    (YOLOv8) — classes: {0: covered, 1: uncovered}
- csrnet_crowd.pth.tar (CSRNet) — crowd density estimation
- MediaPipe Pose       (optional) — violence / aggression detection
- MOG2 background subtraction    — abandoned object detection
"""
import cv2
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
import logging
from app.config import settings
from app.models.detection import DetectionType, ThreatLevel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CSRNet architecture (VGG16 frontend + dilated convolution backend)
# ---------------------------------------------------------------------------
class CSRNet(nn.Module):
    """CSRNet for crowd density estimation (VGG-16 frontend through pool4)."""

    def __init__(self):
        super().__init__()
        # VGG-16 conv layers up to pool4 — matches checkpoint frontend.0 … frontend.21
        self.frontend = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True),     # 0, 1
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),    # 2, 3
            nn.MaxPool2d(2, 2),                                         # 4
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),   # 5, 6
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),  # 7, 8
            nn.MaxPool2d(2, 2),                                         # 9
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),  # 10, 11
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),  # 12, 13
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),  # 14, 15
            nn.MaxPool2d(2, 2),                                         # 16
            # Block 4
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(inplace=True),  # 17, 18
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),  # 19, 20
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),  # 21, 22
        )
        # Dilated convolution backend (input: 512 channels from block 4)
        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
        )
        self.output_layer = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x


class DetectionService:
    """Service for running AI detection models."""

    def __init__(self):
        self.confidence_threshold = settings.CONFIDENCE_THRESHOLD
        self.abandoned_threshold = settings.ABANDONED_OBJECT_THRESHOLD_SECONDS

        # Model references (populated by _load_models)
        self.weapon_model = None       # YOLOv8 — Bags / Box / Weapons
        self.face_model = None         # YOLOv8 — covered / uncovered
        self.crowd_model = None        # CSRNet
        self.pose_model = None         # MediaPipe Pose
        self.mp_pose = None
        self.bg_subtractor = None      # MOG2

        self._load_models()

    # ==================================================================
    # Model loading
    # ==================================================================
    def _load_models(self):
        """Load all detection models. Each one degrades gracefully."""

        # --- 1. YOLOv8 weapon / object model (Bags, Box, Weapons) ---
        weapon_path = Path(settings.MODEL_PATH)
        if weapon_path.exists():
            try:
                from ultralytics import YOLO
                self.weapon_model = YOLO(str(weapon_path))
                logger.info("Weapon model loaded from %s — classes: %s", weapon_path, self.weapon_model.names)
            except ImportError:
                logger.warning("ultralytics not installed — weapon detection disabled")
            except Exception as e:
                logger.error("Failed to load weapon model: %s", e)
        else:
            logger.warning("Weapon model not found at %s — weapon detection disabled", weapon_path)

        # --- 2. YOLOv8 face model (covered / uncovered) ---
        face_path = Path(settings.FACE_MODEL_PATH)
        if face_path.exists():
            try:
                from ultralytics import YOLO
                self.face_model = YOLO(str(face_path))
                logger.info("Face model loaded from %s — classes: %s", face_path, self.face_model.names)
            except ImportError:
                logger.warning("ultralytics not installed — face detection disabled")
            except Exception as e:
                logger.error("Failed to load face model: %s", e)
        else:
            logger.warning("Face model not found at %s — face detection disabled", face_path)

        # --- 3. CSRNet crowd density model ---
        crowd_path = Path(settings.CROWD_MODEL_PATH)
        if crowd_path.exists():
            try:
                self.crowd_model = CSRNet()
                checkpoint = torch.load(str(crowd_path), map_location="cpu", weights_only=False)
                self.crowd_model.load_state_dict(checkpoint["model_state"])
                self.crowd_model.eval()
                logger.info("CSRNet crowd model loaded from %s (epoch %s, MAE %.2f)",
                            crowd_path, checkpoint.get("epoch"), checkpoint.get("best_mae", 0))
            except Exception as e:
                self.crowd_model = None
                logger.error("Failed to load CSRNet model: %s", e)
        else:
            logger.warning("CSRNet model not found at %s — crowd density disabled", crowd_path)

        # --- 4. MediaPipe pose estimation (for violence detection) ---
        try:
            import mediapipe as mp
            self.mp_pose = mp.solutions.pose
            self.pose_model = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            logger.info("MediaPipe Pose model loaded")
        except ImportError:
            self.mp_pose = None
            logger.warning("mediapipe not installed — violence/pose detection disabled")
        except Exception as e:
            self.mp_pose = None
            logger.error("Failed to load MediaPipe Pose: %s", e)

        # --- 5. MOG2 background subtractor (abandoned object detection) ---
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=True
        )

    # ==================================================================
    # 1. Weapon detection  (YOLOv8 — class "Weapons")
    # ==================================================================
    def detect_weapons(
        self,
        frame: np.ndarray,
        frame_timestamp: datetime,
    ) -> List[Dict[str, Any]]:
        """Detect weapons in a frame. Uses higher confidence to avoid false positives."""
        if self.weapon_model is None:
            return []

        h, w = frame.shape[:2]
        # Use higher threshold for weapons (0.7) to reduce false positives
        weapon_conf = max(self.confidence_threshold, 0.7)
        results = self.weapon_model(frame, conf=weapon_conf, verbose=False)
        detections: List[Dict[str, Any]] = []

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                cls_name = self.weapon_model.names.get(cls_id, "unknown")
                # Only flag "Weapons" as weapon detections
                if cls_name.lower() != "weapons":
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({
                    "bbox": [x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h],
                    "confidence": float(box.conf[0]),
                    "class": cls_name,
                })

        return detections

    # ==================================================================
    # 2. Violence / aggression detection  (MediaPipe + optical flow)
    # ==================================================================
    def detect_violence(
        self,
        frame: np.ndarray,
        previous_frames: List[np.ndarray],
        frame_timestamp: datetime,
    ) -> Dict[str, Any]:
        """
        Detect violent behaviour via pose estimation + temporal motion.

        Uses MediaPipe keypoints and inter-frame optical-flow magnitude.
        A dedicated LSTM/1D-CNN can replace the heuristic when trained.
        """
        if self.pose_model is None:
            return {"is_violent": False, "confidence": 0.0, "pose_data": {}}

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.pose_model.process(rgb)

        pose_data: Dict[str, Any] = {}
        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark
            pose_data = {
                "landmarks": [
                    {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility}
                    for lm in landmarks
                ]
            }

        # Motion magnitude via optical flow
        motion_score = 0.0
        if previous_frames:
            prev_gray = cv2.cvtColor(previous_frames[-1], cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
            )
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            motion_score = float(np.mean(mag))

        confidence = min(1.0, motion_score / 10.0)
        is_violent = confidence >= 0.8

        return {
            "is_violent": is_violent,
            "confidence": round(confidence, 4),
            "pose_data": pose_data,
        }

    # ==================================================================
    # 3. Abandoned object detection  (YOLOv8 Bags/Box + MOG2 tracking)
    # ==================================================================
    def detect_abandoned_object(
        self,
        frame: np.ndarray,
        background: Optional[np.ndarray],
        frame_timestamp: datetime,
        object_history: Dict[str, List[datetime]],
    ) -> List[Dict[str, Any]]:
        """
        Detect abandoned objects.

        Uses the weapon model's Bags/Box classes to find objects, then
        tracks how long they remain stationary via MOG2 background
        subtraction.  Objects stationary > threshold → flagged.
        """
        h, w = frame.shape[:2]
        detections: List[Dict[str, Any]] = []

        # Strategy A: Use YOLO Bags/Box detections for precise object tracking
        if self.weapon_model is not None:
            results = self.weapon_model(frame, conf=0.4, verbose=False)
            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = self.weapon_model.names.get(cls_id, "")
                    if cls_name.lower() not in ("bags", "box"):
                        continue

                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    obj_key = f"{cls_name}_{cx // 40}_{cy // 40}"

                    if obj_key not in object_history:
                        object_history[obj_key] = []
                    object_history[obj_key].append(frame_timestamp)

                    first_seen = object_history[obj_key][0]
                    duration = (frame_timestamp - first_seen).total_seconds()

                    if duration >= self.abandoned_threshold:
                        detections.append({
                            "bbox": [x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h],
                            "confidence": min(1.0, duration / (self.abandoned_threshold * 2)),
                            "class": f"abandoned_{cls_name.lower()}",
                            "duration_seconds": duration,
                        })
            return detections

        # Strategy B: Fallback to pure MOG2 contour tracking
        fg_mask = self.bg_subtractor.apply(frame)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if cv2.contourArea(cnt) < 500:
                continue
            x, y, cw, ch = cv2.boundingRect(cnt)
            obj_key = f"obj_{(x + cw // 2) // 40}_{(y + ch // 2) // 40}"

            if obj_key not in object_history:
                object_history[obj_key] = []
            object_history[obj_key].append(frame_timestamp)

            first_seen = object_history[obj_key][0]
            duration = (frame_timestamp - first_seen).total_seconds()

            if duration >= self.abandoned_threshold:
                detections.append({
                    "bbox": [x / w, y / h, cw / w, ch / h],
                    "confidence": min(1.0, duration / (self.abandoned_threshold * 2)),
                    "class": "abandoned_object",
                    "duration_seconds": duration,
                })

        return detections

    # ==================================================================
    # 4. Mask / face obfuscation detection  (YOLOv8 — covered/uncovered)
    # ==================================================================
    def detect_mask_face(
        self,
        frame: np.ndarray,
        frame_timestamp: datetime,
    ) -> List[Dict[str, Any]]:
        """
        Detect faces and classify as covered or uncovered using the
        dedicated YOLOv8 face model.
        """
        if self.face_model is None:
            return []

        h, w = frame.shape[:2]
        results = self.face_model(frame, conf=self.confidence_threshold, verbose=False)
        detections: List[Dict[str, Any]] = []

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                cls_name = self.face_model.names.get(cls_id, "unknown")
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                is_masked = cls_name.lower() == "covered"

                detections.append({
                    "bbox": [x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h],
                    "confidence": float(box.conf[0]),
                    "class": cls_name,
                    "is_masked": is_masked,
                })

        return detections

    # ==================================================================
    # 5. Crowd density estimation  (CSRNet)
    # ==================================================================
    def calculate_crowd_density(
        self,
        frame: np.ndarray,
        frame_timestamp: datetime,
    ) -> Dict[str, Any]:
        """
        Estimate crowd count and density using the CSRNet model.

        The model outputs a density map; summing it gives the estimated
        person count.  Falls back to HOG people detector if CSRNet
        is unavailable.
        """
        if self.crowd_model is not None:
            try:
                # Preprocess: resize, normalise, convert to tensor
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (640, 480))
                img = img.astype(np.float32) / 255.0
                # Normalise with ImageNet stats
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                img = (img - mean) / std
                # HWC → CHW → NCHW
                tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)

                with torch.no_grad():
                    density_map = self.crowd_model(tensor)

                person_count = max(0, int(density_map.sum().item()))
                density = min(1.0, person_count / 50.0)

                return {
                    "count": person_count,
                    "density": round(density, 4),
                    "confidence": round(density, 4),
                }
            except Exception as e:
                logger.error("CSRNet inference error: %s", e)

        # Fallback: HOG people detector
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        boxes, _ = hog.detectMultiScale(frame, winStride=(8, 8), scale=1.05)
        person_count = len(boxes)
        density = min(1.0, person_count / 50.0)

        return {
            "count": person_count,
            "density": round(density, 4),
            "confidence": round(density, 4),
        }

    # ==================================================================
    # Threat-level classification (decision engine)
    # ==================================================================

    # Default thresholds per detection type: (critical, high, medium)
    DEFAULT_THRESHOLDS = {
        DetectionType.WEAPON:           (0.9, 0.7, 0.5),
        DetectionType.VIOLENCE:         (0.8, 0.65, 0.5),
        DetectionType.ABANDONED_OBJECT: (None, 0.8, 0.5),   # No CRITICAL for abandoned
        DetectionType.MASK_FACE:        (None, None, 0.9),   # Max MEDIUM for face/mask
        DetectionType.CROWD_DENSITY:    (None, 0.9, 0.7),    # Uses density value
    }

    def classify_threat_level(
        self,
        detection_type: DetectionType,
        confidence: float,
        metadata: Optional[Dict[str, Any]] = None,
        company_thresholds: Optional[Dict[str, Optional[float]]] = None,
    ) -> ThreatLevel:
        """
        Classify threat level based on detection type and confidence.

        If company_thresholds is provided (from CompanyDetectionSettings),
        it overrides the default thresholds for that module.
        company_thresholds keys: critical_threshold, high_threshold, medium_threshold
        """
        # For crowd density, use the density value instead of raw confidence
        value = confidence
        if detection_type == DetectionType.CROWD_DENSITY:
            value = metadata.get("density", 0.0) if metadata else 0.0

        # Get thresholds — company overrides > defaults
        defaults = self.DEFAULT_THRESHOLDS.get(detection_type, (None, None, None))
        if company_thresholds:
            ct = company_thresholds.get("critical_threshold") or defaults[0]
            ht = company_thresholds.get("high_threshold") or defaults[1]
            mt = company_thresholds.get("medium_threshold") or defaults[2]
        else:
            ct, ht, mt = defaults

        if ct is not None and value >= ct:
            return ThreatLevel.CRITICAL
        if ht is not None and value >= ht:
            return ThreatLevel.HIGH
        if mt is not None and value >= mt:
            return ThreatLevel.MEDIUM
        return ThreatLevel.LOW

    def get_enabled_modules(
        self,
        db,
        company_id: Optional[int] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Return a dict of module_name → settings for modules that are active.

        A module is active only if:
        1. It is globally enabled (GlobalModuleSettings)
        2. AND the company has not disabled it (CompanyDetectionSettings)

        Returns dict like:
        {
            "weapon": {"critical_threshold": 0.9, "high_threshold": 0.7, ...},
            "mask_face": {...},
        }
        """
        from app.models.detection_settings import GlobalModuleSettings, CompanyDetectionSettings

        # 1. Check global settings
        global_settings = {s.module_name: s.is_enabled for s in db.query(GlobalModuleSettings).all()}

        # 2. Check company settings
        company_settings = {}
        if company_id:
            for cs in db.query(CompanyDetectionSettings).filter(CompanyDetectionSettings.company_id == company_id).all():
                company_settings[cs.module_name] = cs

        enabled: Dict[str, Dict[str, Any]] = {}
        for dt in DetectionType:
            module = dt.value

            # Skip if globally disabled
            if not global_settings.get(module, True):
                continue

            # Skip if company disabled it
            cs = company_settings.get(module)
            if cs and not cs.is_enabled:
                continue

            # Build threshold overrides
            thresholds: Dict[str, Any] = {}
            if cs:
                thresholds["critical_threshold"] = cs.critical_threshold
                thresholds["high_threshold"] = cs.high_threshold
                thresholds["medium_threshold"] = cs.medium_threshold
                thresholds["min_confidence"] = cs.min_confidence
            enabled[module] = thresholds

        return enabled


detection_service = DetectionService()
