"""
Detection service for integrating with AI models.

Models used:
- weapon_detection.pt  (YOLOv8) — classes: {0: Bags, 1: Box, 2: Weapons}
- face_detection.pt    (YOLOv8) — classes: {0: covered, 1: uncovered}
- csrnet_crowd.pth.tar (CSRNet) — crowd density estimation
- MediaPipe Pose       (optional) — violence / aggression detection
- MOG2 background subtraction    — abandoned object detection

Performance optimizations:
- ONNX Runtime for all models (2-4x faster than PyTorch on CPU)
- YOLO runs ONCE per frame; results shared between weapon + abandoned object detection
- All model inferences run in parallel via ThreadPoolExecutor
- MediaPipe model_complexity=0 (fastest)
- Lighter optical flow parameters
- Frame-similarity check to skip unchanged frames
"""
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from app.config import settings
from app.models.detection import DetectionType, ThreatLevel

logger = logging.getLogger(__name__)

# Try to import ONNX Runtime — falls back to PyTorch if unavailable
try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
    # Optimize thread count for CPU — use physical cores (not hyperthreads)
    cpu_cores = os.cpu_count() or 4
    ort.set_default_logger_severity(3)  # suppress verbose logs
    logger.info("ONNX Runtime available — using accelerated inference (%d CPU threads)", cpu_cores)
except ImportError:
    ORT_AVAILABLE = False
    logger.warning("onnxruntime not installed — using PyTorch (slower). Install with: pip install onnxruntime")

# Optimize PyTorch CPU threads (for CSRNet fallback)
cpu_cores = os.cpu_count() or 4
torch.set_num_threads(cpu_cores)
torch.set_num_interop_threads(max(1, cpu_cores // 2))


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
        self.crowd_model = None        # CSRNet (or "onnx" sentinel when using ONNX)
        self.crowd_onnx_session = None # ONNX Runtime session for CSRNet
        self.pose_model = None         # MediaPipe Pose
        self.mp_pose = None
        self.bg_subtractor = None      # MOG2

        # Thread pool for parallel model inference (3 = YOLO + CSRNet + MediaPipe)
        self._inference_pool = ThreadPoolExecutor(max_workers=3)

        # Frame similarity threshold — skip AI if frame barely changed
        self._prev_frame_gray = None
        self.similarity_threshold = 0.98  # skip if > 98% similar

        self._load_models()

    # ==================================================================
    # Model loading
    # ==================================================================
    def _load_yolo_onnx(self, pt_path: Path, model_name: str):
        """
        Load a YOLO model. If ONNX Runtime is available, export to ONNX
        on first run and load the ONNX version (2-4x faster on CPU).
        Falls back to PyTorch if onnxruntime is not installed.
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            logger.warning("ultralytics not installed — %s detection disabled", model_name)
            return None

        onnx_path = pt_path.with_suffix(".onnx")

        if ORT_AVAILABLE and onnx_path.exists():
            # ONNX already exported — load directly (fast startup)
            try:
                model = YOLO(str(onnx_path), task="detect")
                logger.info("%s loaded from ONNX: %s — classes: %s", model_name, onnx_path, model.names)
                return model
            except Exception as e:
                logger.warning("Failed to load ONNX %s, falling back to PyTorch: %s", model_name, e)

        if ORT_AVAILABLE and not onnx_path.exists():
            # Export .pt → .onnx (one-time, takes ~30s)
            try:
                logger.info("Exporting %s to ONNX (one-time operation)...", model_name)
                temp_model = YOLO(str(pt_path))
                temp_model.export(format="onnx", opset=17, simplify=True)
                logger.info("ONNX export complete: %s", onnx_path)
                # Load the exported ONNX model
                model = YOLO(str(onnx_path), task="detect")
                logger.info("%s loaded from ONNX: %s — classes: %s", model_name, onnx_path, model.names)
                return model
            except Exception as e:
                logger.warning("ONNX export failed for %s, using PyTorch: %s", model_name, e)

        # Fallback: plain PyTorch
        try:
            model = YOLO(str(pt_path))
            logger.info("%s loaded from PyTorch: %s — classes: %s", model_name, pt_path, model.names)
            return model
        except Exception as e:
            logger.error("Failed to load %s: %s", model_name, e)
            return None

    def _load_csrnet_onnx(self, crowd_path: Path):
        """
        Load CSRNet. If ONNX Runtime is available, export to ONNX on
        first run and use ort.InferenceSession (2-4x faster on CPU).
        """
        onnx_path = crowd_path.with_suffix(".onnx")

        # Try loading existing ONNX session
        if ORT_AVAILABLE and onnx_path.exists():
            try:
                session = ort.InferenceSession(
                    str(onnx_path),
                    providers=["CPUExecutionProvider"],
                )
                self.crowd_onnx_session = session
                self.crowd_model = "onnx"  # sentinel to indicate ONNX mode
                logger.info("CSRNet loaded from ONNX: %s", onnx_path)
                return
            except Exception as e:
                logger.warning("Failed to load CSRNet ONNX, falling back to PyTorch: %s", e)

        # Load PyTorch model (needed for fallback or export)
        try:
            pt_model = CSRNet()
            checkpoint = torch.load(str(crowd_path), map_location="cpu", weights_only=False)
            pt_model.load_state_dict(checkpoint["model_state"])
            pt_model.eval()
            logger.info("CSRNet loaded from PyTorch: %s (epoch %s, MAE %.2f)",
                        crowd_path, checkpoint.get("epoch"), checkpoint.get("best_mae", 0))
        except Exception as e:
            self.crowd_model = None
            logger.error("Failed to load CSRNet: %s", e)
            return

        # Export to ONNX if runtime is available
        if ORT_AVAILABLE and not onnx_path.exists():
            try:
                logger.info("Exporting CSRNet to ONNX (one-time operation)...")
                dummy = torch.randn(1, 3, 384, 512)
                torch.onnx.export(
                    pt_model, dummy, str(onnx_path),
                    opset_version=17,
                    input_names=["input"],
                    output_names=["density_map"],
                    dynamic_axes={"input": {2: "height", 3: "width"}},
                )
                session = ort.InferenceSession(
                    str(onnx_path),
                    providers=["CPUExecutionProvider"],
                )
                self.crowd_onnx_session = session
                self.crowd_model = "onnx"
                logger.info("CSRNet ONNX export complete: %s", onnx_path)
                return
            except Exception as e:
                logger.warning("CSRNet ONNX export failed, using PyTorch: %s", e)

        # Fallback: keep PyTorch model
        self.crowd_model = pt_model

    def _load_models(self):
        """Load all detection models. Each one degrades gracefully."""

        # --- 1. YOLOv8 weapon / object model (Bags, Box, Weapons) ---
        weapon_path = Path(settings.MODEL_PATH)
        if weapon_path.exists():
            self.weapon_model = self._load_yolo_onnx(weapon_path, "Weapon")
        else:
            logger.warning("Weapon model not found at %s — weapon detection disabled", weapon_path)

        # --- 2. YOLOv8 face model (covered / uncovered) ---
        face_path = Path(settings.FACE_MODEL_PATH)
        if face_path.exists():
            self.face_model = self._load_yolo_onnx(face_path, "Face")
        else:
            logger.warning("Face model not found at %s — face detection disabled", face_path)

        # --- 3. CSRNet crowd density model ---
        crowd_path = Path(settings.CROWD_MODEL_PATH)
        if crowd_path.exists():
            self._load_csrnet_onnx(crowd_path)
        else:
            logger.warning("CSRNet model not found at %s — crowd density disabled", crowd_path)

        # --- 4. MediaPipe pose estimation (for violence detection) ---
        try:
            import mediapipe as mp
            self.mp_pose = mp.solutions.pose
            self.pose_model = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=0,        # 0=fastest (was 1)
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            logger.info("MediaPipe Pose model loaded (complexity=0)")
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
    # Frame similarity check — skip AI on nearly identical frames
    # ==================================================================
    def is_frame_similar(self, frame: np.ndarray) -> bool:
        """Return True if frame is very similar to the previous one (skip AI)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Downscale for fast comparison
        small = cv2.resize(gray, (160, 120))

        if self._prev_frame_gray is None:
            self._prev_frame_gray = small
            return False

        # Structural similarity via normalized correlation
        score = cv2.matchTemplate(
            self._prev_frame_gray, small, cv2.TM_CCORR_NORMED
        )[0][0]

        self._prev_frame_gray = small
        return score > self.similarity_threshold

    # ==================================================================
    # Shared YOLO inference — run weapon model ONCE, split results
    # ==================================================================
    def run_yolo_shared(
        self, frame: np.ndarray, conf: float = 0.4
    ) -> Dict[str, list]:
        """
        Run the weapon YOLO model once and split results into categories.
        Returns {"weapons": [...], "bags_boxes": [...], "all_boxes": [...]}
        """
        if self.weapon_model is None:
            return {"weapons": [], "bags_boxes": [], "all_boxes": []}

        h, w = frame.shape[:2]
        results = self.weapon_model(frame, conf=conf, verbose=False)

        weapons = []
        bags_boxes = []
        all_boxes = []

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                cls_name = self.weapon_model.names.get(cls_id, "unknown")
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                det = {
                    "bbox": [x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h],
                    "confidence": float(box.conf[0]),
                    "class": cls_name,
                    "pixel_bbox": [x1, y1, x2, y2],
                }
                all_boxes.append(det)

                if cls_name.lower() == "weapons":
                    weapons.append(det)
                elif cls_name.lower() in ("bags", "box"):
                    bags_boxes.append(det)

        return {"weapons": weapons, "bags_boxes": bags_boxes, "all_boxes": all_boxes}

    # ==================================================================
    # Parallel detection — run all enabled modules concurrently
    # ==================================================================
    def detect_all_parallel(
        self,
        frame: np.ndarray,
        previous_frames: List[np.ndarray],
        frame_timestamp: datetime,
        object_history: Dict[str, List[datetime]],
        enabled_modules: Dict[str, Dict[str, Any]],
    ) -> List[tuple]:
        """
        Run all enabled detection modules in parallel.
        Returns list of (DetectionType, detection_dict) tuples.
        """
        all_results: List[tuple] = []
        futures = {}

        # Check frame similarity — skip expensive AI if frame barely changed
        if self.is_frame_similar(frame):
            return []

        # Step 1: Run shared YOLO once (needed by weapon + abandoned_object)
        need_yolo = "weapon" in enabled_modules or "abandoned_object" in enabled_modules
        yolo_results = self.run_yolo_shared(frame, conf=0.4) if need_yolo else None

        # Step 2: Submit independent models in parallel
        if "weapon" in enabled_modules and yolo_results:
            futures["weapon"] = self._inference_pool.submit(
                self._extract_weapons, yolo_results
            )

        if "abandoned_object" in enabled_modules and yolo_results:
            futures["abandoned_object"] = self._inference_pool.submit(
                self._extract_abandoned, yolo_results, frame, frame_timestamp, object_history
            )

        if "mask_face" in enabled_modules:
            futures["mask_face"] = self._inference_pool.submit(
                self.detect_mask_face, frame, frame_timestamp
            )

        if "crowd_density" in enabled_modules:
            futures["crowd_density"] = self._inference_pool.submit(
                self.calculate_crowd_density, frame, frame_timestamp
            )

        if "violence" in enabled_modules:
            futures["violence"] = self._inference_pool.submit(
                self.detect_violence, frame, previous_frames, frame_timestamp
            )

        # Step 3: Collect results
        for module, future in futures.items():
            try:
                result = future.result(timeout=30)

                if module == "weapon":
                    for det in result:
                        all_results.append((DetectionType.WEAPON, det))
                elif module == "abandoned_object":
                    for det in result:
                        all_results.append((DetectionType.ABANDONED_OBJECT, det))
                elif module == "mask_face":
                    for det in result:
                        all_results.append((DetectionType.MASK_FACE, det))
                elif module == "crowd_density":
                    if result["count"] > 0:
                        all_results.append((DetectionType.CROWD_DENSITY, result))
                elif module == "violence":
                    if result["is_violent"]:
                        all_results.append((DetectionType.VIOLENCE, result))
            except Exception as e:
                logger.warning("Module %s failed: %s", module, e)

        return all_results

    # ==================================================================
    # 1. Weapon detection — extract from shared YOLO results
    # ==================================================================
    def _extract_weapons(self, yolo_results: Dict[str, list]) -> List[Dict[str, Any]]:
        """Extract weapon detections from shared YOLO results."""
        weapon_conf = max(self.confidence_threshold, 0.7)
        return [
            det for det in yolo_results["weapons"]
            if det["confidence"] >= weapon_conf
        ]

    def detect_weapons(
        self,
        frame: np.ndarray,
        frame_timestamp: datetime,
    ) -> List[Dict[str, Any]]:
        """Detect weapons in a frame (standalone, for backward compat)."""
        if self.weapon_model is None:
            return []
        results = self.run_yolo_shared(frame)
        return self._extract_weapons(results)

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
        Optimized: lighter optical flow params, model_complexity=0.
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

        # Motion magnitude via optical flow (lighter params)
        motion_score = 0.0
        if previous_frames:
            prev_gray = cv2.cvtColor(previous_frames[-1], cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None,
                pyr_scale=0.5, levels=2, winsize=11,    # was levels=3, winsize=15
                iterations=2, poly_n=5, poly_sigma=1.1,  # was iterations=3
                flags=0,
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
    # 3. Abandoned object detection — extract from shared YOLO results
    # ==================================================================
    def _extract_abandoned(
        self,
        yolo_results: Dict[str, list],
        frame: np.ndarray,
        frame_timestamp: datetime,
        object_history: Dict[str, List[datetime]],
    ) -> List[Dict[str, Any]]:
        """Extract abandoned objects from shared YOLO results (Bags/Box)."""
        detections: List[Dict[str, Any]] = []
        h, w = frame.shape[:2]

        for det in yolo_results["bags_boxes"]:
            pixel_bbox = det["pixel_bbox"]
            x1, y1, x2, y2 = pixel_bbox
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            cls_name = det["class"]
            obj_key = f"{cls_name}_{cx // 40}_{cy // 40}"

            if obj_key not in object_history:
                object_history[obj_key] = []
            object_history[obj_key].append(frame_timestamp)

            first_seen = object_history[obj_key][0]
            duration = (frame_timestamp - first_seen).total_seconds()

            if duration >= self.abandoned_threshold:
                detections.append({
                    "bbox": det["bbox"],
                    "confidence": min(1.0, duration / (self.abandoned_threshold * 2)),
                    "class": f"abandoned_{cls_name.lower()}",
                    "duration_seconds": duration,
                })

        return detections

    def detect_abandoned_object(
        self,
        frame: np.ndarray,
        background: Optional[np.ndarray],
        frame_timestamp: datetime,
        object_history: Dict[str, List[datetime]],
    ) -> List[Dict[str, Any]]:
        """Detect abandoned objects (standalone, for backward compat)."""
        h, w = frame.shape[:2]
        detections: List[Dict[str, Any]] = []

        if self.weapon_model is not None:
            yolo_results = self.run_yolo_shared(frame)
            return self._extract_abandoned(yolo_results, frame, frame_timestamp, object_history)

        # Fallback: MOG2 contour tracking
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
        """Detect faces and classify as covered or uncovered."""
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
        """Estimate crowd count and density using CSRNet or HOG fallback."""
        if self.crowd_model is not None:
            try:
                # Preprocessing must match training pipeline exactly
                # (see E:\CSRNet-pytorch\video_inference.py — preprocess_frame)
                from PIL import Image
                from torchvision import transforms

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                pil_img = pil_img.resize((512, 384), Image.BILINEAR)

                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ])
                input_tensor = transform(pil_img).unsqueeze(0)  # (1, 3, 384, 512)
                input_array = input_tensor.numpy()

                # Use ONNX Runtime if available (2-4x faster on CPU)
                if self.crowd_onnx_session is not None:
                    outputs = self.crowd_onnx_session.run(
                        None, {"input": input_array}
                    )
                    density_sum = float(outputs[0].sum())
                else:
                    # Fallback: PyTorch
                    with torch.no_grad():
                        density_map = self.crowd_model(input_tensor)
                    density_sum = float(density_map.sum().item())

                person_count = max(0, int(density_sum))
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
        small = cv2.resize(frame, (320, 240))  # downscale for HOG too
        boxes, _ = hog.detectMultiScale(small, winStride=(8, 8), scale=1.05)
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
        """Classify threat level based on detection type and confidence."""
        value = confidence
        if detection_type == DetectionType.CROWD_DENSITY:
            value = metadata.get("density", 0.0) if metadata else 0.0

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
        """Return a dict of module_name → settings for modules that are active."""
        from app.models.detection_settings import GlobalModuleSettings, CompanyDetectionSettings

        global_settings = {s.module_name: s.is_enabled for s in db.query(GlobalModuleSettings).all()}

        company_settings = {}
        if company_id:
            for cs in db.query(CompanyDetectionSettings).filter(CompanyDetectionSettings.company_id == company_id).all():
                company_settings[cs.module_name] = cs

        enabled: Dict[str, Dict[str, Any]] = {}
        for dt in DetectionType:
            module = dt.value

            if not global_settings.get(module, True):
                continue

            cs = company_settings.get(module)
            if cs and not cs.is_enabled:
                continue

            thresholds: Dict[str, Any] = {}
            if cs:
                thresholds["critical_threshold"] = cs.critical_threshold
                thresholds["high_threshold"] = cs.high_threshold
                thresholds["medium_threshold"] = cs.medium_threshold
                thresholds["min_confidence"] = cs.min_confidence
            enabled[module] = thresholds

        return enabled


detection_service = DetectionService()
