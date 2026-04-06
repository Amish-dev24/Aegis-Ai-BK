"""
Detection service for integrating with AI models.

Models used:
- weapon-box-bags-v3.pt (YOLO) — Bags, Box, Weapons
- face_detection.pt    (YOLOv8) — classes: {0: covered, 1: uncovered}
- csrnet_crowd.pth.tar (CSRNet) — crowd density estimation
- MediaPipe Pose       (optional) — violence / aggression detection
- MOG2 background subtraction    — abandoned object detection

Performance optimizations:
- ONNX Runtime for YOLO on CPU (faster than raw PyTorch)
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

# weapon-box-bags-v3.pt — class names in the checkpoint (Ultralytics val table):
#   Bags | Box | Weapons
# Routing: Weapons → weapon detections; Bags + Box → abandoned / bags_boxes.
_BAG_BOX_CLASS_NAMES = frozenset({"bags", "bag", "box", "boxes"})


def _is_weapon_class(cls_name: str) -> bool:
    """True if this label is the weapon class (primary: ``Weapons`` from weapon-box-bags-v3)."""
    n = cls_name.lower().strip()
    if not n:
        return False
    if n in (
        "weapon",
        "weapons",
        "gun",
        "guns",
        "pistol",
        "rifle",
        "firearm",
        "handgun",
        "knife",
        "knives",
    ):
        return True
    return "weapon" in n or "gun" in n or "pistol" in n or "rifle" in n or "firearm" in n


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
        # When weapon model is loaded from .onnx (fixed input), cap predict imgsz (see ONNX_YOLO_IMGSZ)
        self._weapon_onnx_imgsz_cap: Optional[int] = None
        self.face_model = None         # YOLOv8 — covered / uncovered
        self.crowd_model = None        # CSRNet (or "onnx" sentinel when using ONNX)
        self.crowd_onnx_session = None # ONNX Runtime session for CSRNet
        self.pose_model = None         # MediaPipe Pose
        self.mp_pose = None
        self.violence_onnx_session = None  # Conv3D violence classifier (ONNX)
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
        Load a YOLO model. Prefer ONNX Runtime (export from `.pt` if needed), else PyTorch.

        Returns:
            (model, fixed_imgsz_cap): cap is set for ONNX fixed exports (``ONNX_YOLO_IMGSZ``),
            or None when using PyTorch weights.
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            logger.warning("ultralytics not installed — %s detection disabled", model_name)
            return None, None

        fixed_cap = int(getattr(settings, "ONNX_YOLO_IMGSZ", 640))
        onnx_path = pt_path.with_suffix(".onnx")

        if ORT_AVAILABLE and onnx_path.exists():
            try:
                model = YOLO(str(onnx_path), task="detect")
                logger.info(
                    "%s loaded from ONNX: %s — classes: %s (predict imgsz capped at %d)",
                    model_name,
                    onnx_path,
                    model.names,
                    fixed_cap,
                )
                return model, fixed_cap
            except Exception as e:
                logger.warning("Failed to load ONNX %s, falling back: %s", model_name, e)

        if ORT_AVAILABLE and not onnx_path.exists() and pt_path.suffix.lower() in (".pt", ".pth"):
            try:
                logger.info("Exporting %s to ONNX (one-time operation)...", model_name)
                temp_model = YOLO(str(pt_path))
                temp_model.export(format="onnx", opset=17, simplify=True, imgsz=fixed_cap)
                logger.info("ONNX export complete: %s", onnx_path)
                model = YOLO(str(onnx_path), task="detect")
                logger.info(
                    "%s loaded from ONNX: %s — classes: %s (predict imgsz capped at %d)",
                    model_name,
                    onnx_path,
                    model.names,
                    fixed_cap,
                )
                return model, fixed_cap
            except Exception as e:
                logger.warning("ONNX export failed for %s, using PyTorch: %s", model_name, e)

        try:
            model = YOLO(str(pt_path))
            logger.info("%s loaded from PyTorch: %s — classes: %s", model_name, pt_path, model.names)
            return model, None
        except Exception as e:
            logger.error("Failed to load %s: %s", model_name, e)
            return None, None

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
            self.weapon_model, self._weapon_onnx_imgsz_cap = self._load_yolo_onnx(
                weapon_path, "Weapon"
            )
        else:
            logger.warning("Weapon model not found at %s — weapon detection disabled", weapon_path)

        # --- 2. YOLOv8 face model (covered / uncovered) ---
        face_path = Path(settings.FACE_MODEL_PATH)
        if face_path.exists():
            self.face_model, _ = self._load_yolo_onnx(face_path, "Face")
        else:
            logger.warning("Face model not found at %s — face detection disabled", face_path)

        # --- 3. CSRNet crowd density model ---
        crowd_path = Path(settings.CROWD_MODEL_PATH)
        if crowd_path.exists():
            self._load_csrnet_onnx(crowd_path)
        else:
            logger.warning("CSRNet model not found at %s — crowd density disabled", crowd_path)

        # --- 4. Conv3D Violence classifier (trained on Real Life Violence dataset) ---
        violence_onnx_path = Path("./models/violence_model.onnx")
        violence_pt_path = Path("./models/violence_model.pt")
        if ORT_AVAILABLE and violence_onnx_path.exists():
            try:
                self.violence_onnx_session = ort.InferenceSession(
                    str(violence_onnx_path), providers=["CPUExecutionProvider"],
                )
                logger.info("Violence model loaded from ONNX: %s", violence_onnx_path)
            except Exception as e:
                logger.warning("Failed to load violence ONNX: %s", e)
        elif violence_pt_path.exists():
            try:
                # Export to ONNX on first run
                from app.services.violence_model import ViolenceClassifier as VC
                vc = VC()
                ckpt = torch.load(str(violence_pt_path), map_location="cpu", weights_only=False)
                vc.load_state_dict(ckpt["model_state"])
                vc.eval()
                logger.info("Violence model loaded from PyTorch (acc: %.1f%%)", ckpt.get("val_acc", 0))

                if ORT_AVAILABLE:
                    logger.info("Exporting violence model to ONNX...")
                    dummy = torch.randn(1, 3, 16, 64, 64)
                    torch.onnx.export(vc, dummy, str(violence_onnx_path), opset_version=18,
                                      input_names=["video_clip"], output_names=["prediction"],
                                      dynamic_axes={"video_clip": {0: "batch"}}, dynamo=False)
                    self.violence_onnx_session = ort.InferenceSession(
                        str(violence_onnx_path), providers=["CPUExecutionProvider"],
                    )
                    logger.info("Violence ONNX export complete: %s", violence_onnx_path)
                else:
                    # Keep PyTorch model as fallback
                    self.pose_model = vc
            except Exception as e:
                logger.error("Failed to load violence model: %s", e)
        else:
            logger.warning("Violence model not found at %s — violence detection disabled", violence_pt_path)

        # Also try MediaPipe as supplementary (optional)
        try:
            import mediapipe as mp
            self.mp_pose = mp.solutions.pose
            self.pose_model = self.mp_pose.Pose(
                static_image_mode=False, model_complexity=0,
                min_detection_confidence=0.5, min_tracking_confidence=0.5,
            )
            logger.info("MediaPipe Pose also loaded (supplementary)")
        except ImportError:
            self.mp_pose = None
        except Exception:
            self.mp_pose = None

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
        self, frame: np.ndarray, conf: float = 0.4, imgsz: Optional[int] = None
    ) -> Dict[str, list]:
        """
        Run the weapon YOLO model once and split results into categories.
        Returns {"weapons": [...], "bags_boxes": [...], "all_boxes": [...]}
        """
        if self.weapon_model is None:
            return {"weapons": [], "bags_boxes": [], "all_boxes": []}

        h, w = frame.shape[:2]
        max_side = max(h, w)
        if imgsz is not None:
            infer_sz = int(imgsz)
        else:
            # PyTorch: up to WEAPON_YOLO_IMGSZ. ONNX: fixed export (default 640) — larger causes ONNXRuntimeError.
            lim = (
                self._weapon_onnx_imgsz_cap
                if self._weapon_onnx_imgsz_cap is not None
                else getattr(settings, "WEAPON_YOLO_IMGSZ", 1280)
            )
            infer_sz = min(max_side, lim)
        infer_sz = max(32, int(infer_sz))
        if self._weapon_onnx_imgsz_cap is not None:
            infer_sz = min(infer_sz, self._weapon_onnx_imgsz_cap)
        results = self.weapon_model(
            frame, conf=conf, imgsz=infer_sz, verbose=False
        )

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

                if _is_weapon_class(cls_name):
                    weapons.append(det)
                elif cls_name.lower().strip() in _BAG_BOX_CLASS_NAMES:
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
        full_frame: Optional[np.ndarray] = None,
    ) -> List[tuple]:
        """
        Run all enabled detection modules in parallel.
        Args:
            frame: downscaled frame for YOLO/face/violence (640px max)
            full_frame: original resolution frame for CSRNet (needs full res for accurate count)
        Returns list of (DetectionType, detection_dict) tuples.
        """
        all_results: List[tuple] = []
        futures = {}

        # Skip near-duplicate frames for speed — but not when weapon/abandoned YOLO runs:
        # small objects (guns) barely move the correlation score frame-to-frame.
        need_yolo = "weapon" in enabled_modules or "abandoned_object" in enabled_modules
        similar = self.is_frame_similar(frame)
        if similar and not need_yolo:
            return []

        weapon_min = enabled_modules.get("weapon", {}).get("min_confidence")
        weapon_conf = (
            float(weapon_min) if weapon_min is not None else self.confidence_threshold
        )

        # Ultralytics `conf` drops boxes below this before post-processing. It must be
        # <= company min_confidence (e.g. 0.2 vs hardcoded 0.25 loses valid boxes).
        # When min is high, keep raw conf at 0.25 max and filter in _extract_weapons.
        if need_yolo:
            if "weapon" in enabled_modules:
                yolo_conf = max(0.01, min(weapon_conf, 0.25))
            else:
                yolo_conf = 0.25
            yolo_results = self.run_yolo_shared(frame, conf=yolo_conf)
        else:
            yolo_results = None

        # Step 2: Submit independent models in parallel
        if "weapon" in enabled_modules and yolo_results:
            futures["weapon"] = self._inference_pool.submit(
                self._extract_weapons, yolo_results, weapon_conf
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
            # CSRNet needs ORIGINAL resolution frame for accurate scale factor
            crowd_frame = full_frame if full_frame is not None else frame
            futures["crowd_density"] = self._inference_pool.submit(
                self.calculate_crowd_density, crowd_frame, frame_timestamp
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
    def _extract_weapons(
        self, yolo_results: Dict[str, list], weapon_conf: float
    ) -> List[Dict[str, Any]]:
        """Extract weapon detections from shared YOLO results."""
        return [
            det
            for det in yolo_results["weapons"]
            if det["confidence"] >= weapon_conf
        ]

    def detect_weapons(
        self,
        frame: np.ndarray,
        frame_timestamp: datetime,
        min_confidence: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Detect weapons in a frame (standalone, for backward compat)."""
        if self.weapon_model is None:
            return []
        weapon_conf = (
            float(min_confidence)
            if min_confidence is not None
            else self.confidence_threshold
        )
        yolo_conf = max(0.01, min(weapon_conf, 0.25))
        results = self.run_yolo_shared(frame, conf=yolo_conf)
        return self._extract_weapons(results, weapon_conf)

    # ==================================================================
    # 2. Violence / aggression detection  (Conv3D trained model)
    # ==================================================================
    def detect_violence(
        self,
        frame: np.ndarray,
        previous_frames: List[np.ndarray],
        frame_timestamp: datetime,
    ) -> Dict[str, Any]:
        """
        Detect violent behaviour using trained Conv3D classifier.
        Uses 16 frames (current + previous) resized to 64x64.
        Model trained on Real Life Violence Situations Dataset (~89.8% accuracy).
        Falls back to optical flow if Conv3D model not available.
        """
        # ── Primary: Conv3D trained model ──
        if self.violence_onnx_session is not None or (
            self.pose_model is not None and hasattr(self.pose_model, 'features')
        ):
            # Build a clip of 16 frames from previous_frames + current frame
            clip_frames = list(previous_frames[-15:]) + [frame]  # up to 16 frames

            # Pad with duplicates if we don't have 16 frames yet
            while len(clip_frames) < 16:
                clip_frames.insert(0, clip_frames[0])
            clip_frames = clip_frames[-16:]  # exactly 16

            # Resize all to 64x64 and normalize
            processed = []
            for f in clip_frames:
                resized = cv2.resize(f, (64, 64))
                processed.append(resized)

            clip = np.array(processed, dtype=np.float32) / 255.0  # (16, 64, 64, 3)
            clip = np.transpose(clip, (3, 0, 1, 2))  # (3, 16, 64, 64)
            clip = clip[np.newaxis, ...]  # (1, 3, 16, 64, 64)

            try:
                if self.violence_onnx_session is not None:
                    outputs = self.violence_onnx_session.run(
                        None, {"video_clip": clip}
                    )
                    logits = outputs[0][0]  # [non_violent, violent]
                else:
                    # PyTorch fallback
                    tensor = torch.from_numpy(clip)
                    with torch.no_grad():
                        logits = self.pose_model(tensor)[0].numpy()

                # Softmax to get probabilities
                exp_logits = np.exp(logits - np.max(logits))
                probs = exp_logits / exp_logits.sum()
                violence_prob = float(probs[1])  # index 1 = violent

                is_violent = violence_prob >= 0.6
                return {
                    "is_violent": is_violent,
                    "confidence": round(violence_prob, 4),
                    "model": "conv3d",
                    "pose_data": {},
                }
            except Exception as e:
                logger.warning("Conv3D violence inference failed: %s", e)

        # ── Fallback: optical flow motion detection ──
        motion_score = 0.0
        if previous_frames:
            prev_gray = cv2.cvtColor(previous_frames[-1], cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None,
                pyr_scale=0.5, levels=2, winsize=11,
                iterations=2, poly_n=5, poly_sigma=1.1,
                flags=0,
            )
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            motion_score = float(np.mean(mag))

        confidence = min(1.0, motion_score / 10.0)
        is_violent = confidence >= 0.8

        return {
            "is_violent": is_violent,
            "confidence": round(confidence, 4),
            "model": "optical_flow_fallback",
            "pose_data": {},
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
        """
        Estimate crowd count and density using CSRNet or HOG fallback.
        
        Returns: dict with count, density, confidence, and density_map_normalized
        """
        if self.crowd_model is not None:
            try:
                # Preprocessing must match training pipeline exactly
                # (see E:\CSRNet-pytorch\video_inference.py — preprocess_frame)
                from PIL import Image
                from torchvision import transforms

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                max_side = max(32, int(settings.CROWD_INFERENCE_MAX_SIDE))
                # Training uses 512x384 (4:3); ONNX export uses dynamic H/W
                cw, ch = max_side, int(384 * max_side / 512)
                pil_img = pil_img.resize((cw, ch), Image.BILINEAR)

                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ])
                input_tensor = transform(pil_img).unsqueeze(0)  # (1, 3, ch, cw)
                input_array = input_tensor.numpy()

                # Use ONNX Runtime if available (2-4x faster on CPU)
                density_map = None
                if self.crowd_onnx_session is not None:
                    outputs = self.crowd_onnx_session.run(None, {"input": input_array})
                    density_map = outputs[0]  # (1, 1, height, width)
                    density_sum = float(density_map.sum())
                else:
                    # Fallback: PyTorch
                    with torch.no_grad():
                        density_tensor = self.crowd_model(input_tensor)
                    density_map = density_tensor.cpu().numpy()
                    density_sum = float(density_tensor.sum().item())

                raw_density_map = np.squeeze(density_map)  # (H, W)

                # ── Scale factor ──
                # CSRNet was trained on ShanghaiTech (close-up ground-level).
                # It underestimates on downscaled/aerial views because the
                # density map resolution (48x64) loses small heads.
                # Scale factor compensates based on original frame vs model input.
                orig_h, orig_w = frame.shape[:2]
                orig_pixels = orig_h * orig_w
                model_pixels = ch * cw
                # Scale proportional to resolution ratio (capped at 10x)
                scale_factor = min(10.0, max(1.0, (orig_pixels / model_pixels) ** 0.5))

                raw_count = max(0, density_sum) * scale_factor

                # ── False-positive filter ──
                map_max = float(raw_density_map.max())
                map_mean = float(raw_density_map.mean()) if raw_density_map.size > 0 else 0.0
                map_std = float(raw_density_map.std()) if raw_density_map.size > 0 else 0.0

                peak_ratio = map_max / (map_mean + 1e-8)
                cv_ratio = map_std / (map_mean + 1e-8)

                logger.info(
                    "CSRNet raw=%.1f (scaled x%.1f), peak_ratio=%.1f, cv=%.1f, "
                    "max=%.4f, mean=%.4f",
                    raw_count, scale_factor, peak_ratio, cv_ratio,
                    map_max, map_mean,
                )

                # Reject only if density map is very uniform (texture noise)
                # Real crowds always have peaks (peak_ratio > 3) and variance (cv > 1)
                is_fake = (peak_ratio < 3.0 and cv_ratio < 1.0)

                if is_fake:
                    logger.info("CSRNet REJECTED as noise (peak=%.1f, cv=%.1f)", peak_ratio, cv_ratio)
                    person_count = 0
                    density = 0.0
                else:
                    person_count = max(0, int(raw_count))
                    density = min(1.0, person_count / 50.0)
                    logger.info("CSRNet ACCEPTED: count=%d, density=%.2f, scale=%.1fx",
                                person_count, density, scale_factor)

                # Normalize density map for heatmap visualization
                if raw_density_map.max() > raw_density_map.min():
                    density_normalized = ((raw_density_map - raw_density_map.min()) /
                                         (raw_density_map.max() - raw_density_map.min()) * 255).astype(np.uint8)
                else:
                    density_normalized = np.zeros_like(raw_density_map, dtype=np.uint8)

                return {
                    "count": person_count,
                    "density": round(density, 4),
                    "confidence": round(density, 4),
                    "density_map_normalized": density_normalized,
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
            "density_map_normalized": None,  # HOG doesn't provide density map
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
        # Company row from DB includes these keys (possibly None). Do not use
        # `x or default` — that replaced an explicit unset with DEFAULT_THRESHOLDS.
        if company_thresholds and "critical_threshold" in company_thresholds:
            ct = company_thresholds.get("critical_threshold")
            ht = company_thresholds.get("high_threshold")
            mt = company_thresholds.get("medium_threshold")
            if ct is None and ht is None and mt is None:
                ct, ht, mt = defaults
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
                thresholds["alert_on_levels"] = cs.alert_on_levels or "medium,high,critical"
                if cs.abandoned_seconds:
                    thresholds["abandoned_seconds"] = cs.abandoned_seconds
            else:
                thresholds["alert_on_levels"] = "medium,high,critical"
            enabled[module] = thresholds

        return enabled


detection_service = DetectionService()
