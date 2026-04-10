"""
Detection service for integrating with AI models.

Models used:
- weapon-box-bags-v3.pt (YOLO) — Bags, Box, Weapons
- face_detection.pt    (YOLOv8) — classes: {0: covered, 1: uncovered}
- csrnet_crowd.pth.tar (CSRNet) or sanet_partB_best.pth (SANet) — crowd density
- MediaPipe Pose       (optional) — violence / aggression detection
- MOG2 background subtraction    — abandoned object detection

Performance optimizations:
- ONNX Runtime for YOLO / crowd / violence ONNX (CUDA when ``onnxruntime-gpu`` + GPU present, else CPU)
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
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
import threading
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
    logger.info("ONNX Runtime loaded (%d CPU threads for CPU-only sessions)", cpu_cores)
except ImportError:
    ORT_AVAILABLE = False
    ort = None  # type: ignore
    cpu_cores = os.cpu_count() or 4
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


def _strip_module_prefix(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Remove ``module.`` prefix from DataParallel checkpoints."""
    out: Dict[str, Any] = {}
    for k, v in state_dict.items():
        nk = k[7:] if k.startswith("module.") else k
        out[nk] = v
    return out


def _resolve_ort_providers() -> List[str]:
    """
    Prefer CUDA when ``onnxruntime-gpu`` is installed and ``ONNX_PREFER_GPU`` is true.
    CPU-only wheels expose only ``CPUExecutionProvider``.
    """
    if not ORT_AVAILABLE or ort is None:
        return ["CPUExecutionProvider"]
    prefer = bool(getattr(settings, "ONNX_PREFER_GPU", True))
    if not prefer:
        return ["CPUExecutionProvider"]
    try:
        available = set(ort.get_available_providers())
    except Exception:
        return ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _make_ort_session(model_path: Path):
    """ONNX Runtime session: GPU (CUDA) when available, else CPU with thread pinning."""
    if not ORT_AVAILABLE or ort is None:
        raise RuntimeError("ONNX Runtime not available")
    providers = _resolve_ort_providers()
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if providers == ["CPUExecutionProvider"]:
        so.intra_op_num_threads = cpu_cores
        so.inter_op_num_threads = max(1, cpu_cores // 2)
    try:
        sess = ort.InferenceSession(
            str(model_path), sess_options=so, providers=providers
        )
    except Exception as e:
        if len(providers) > 1:
            logger.warning(
                "ONNX session with GPU failed (%s), retrying CPU-only for %s",
                e,
                model_path.name,
            )
            so_cpu = ort.SessionOptions()
            so_cpu.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            so_cpu.intra_op_num_threads = cpu_cores
            so_cpu.inter_op_num_threads = max(1, cpu_cores // 2)
            sess = ort.InferenceSession(
                str(model_path),
                sess_options=so_cpu,
                providers=["CPUExecutionProvider"],
            )
        else:
            raise
    active = sess.get_providers()
    logger.info("ONNX Runtime active providers for %s: %s", model_path.name, active)
    return sess


def _try_move_yolo_to_cuda(model: Any, model_name: str) -> None:
    """Move Ultralytics YOLO ``.pt`` model to CUDA when configured and available."""
    if model is None:
        return
    if not getattr(settings, "TORCH_PREFER_GPU", True):
        return
    if not torch.cuda.is_available():
        return
    try:
        model.to("cuda")
        logger.info("%s: Ultralytics model on CUDA", model_name)
    except Exception as e:
        logger.warning("%s: could not use CUDA (%s), using CPU", model_name, e)


# ---------------------------------------------------------------------------
# SANet (ShanghaiTech-style density) — matches sanet_partB_best checkpoints
# ---------------------------------------------------------------------------
class SANet(nn.Module):
    """SANet crowd density (Conv-BN-ReLU frontend + ASPP + density head)."""

    def __init__(self):
        super().__init__()
        self.frontend = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.aspp = nn.ModuleDict({
            "b0": nn.Sequential(
                nn.Conv2d(256, 256, 1, bias=False),
                nn.BatchNorm2d(256),
            ),
            "b1": nn.Sequential(
                nn.Conv2d(256, 256, 3, padding=6, dilation=6, bias=False),
                nn.BatchNorm2d(256),
            ),
            "b2": nn.Sequential(
                nn.Conv2d(256, 256, 3, padding=12, dilation=12, bias=False),
                nn.BatchNorm2d(256),
            ),
            "b3": nn.Sequential(
                nn.Conv2d(256, 256, 3, padding=18, dilation=18, bias=False),
                nn.BatchNorm2d(256),
            ),
            "gap": nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(256, 256, 1, bias=False),
                nn.BatchNorm2d(256),
            ),
            "proj": nn.Sequential(
                nn.Conv2d(1280, 256, 1, bias=False),
                nn.BatchNorm2d(256),
            ),
        })
        self.density_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.frontend(x)
        b0 = self.aspp["b0"](x)
        b1 = self.aspp["b1"](x)
        b2 = self.aspp["b2"](x)
        b3 = self.aspp["b3"](x)
        gap = self.aspp["gap"](x)
        gap = nn.functional.interpolate(
            gap, size=x.shape[2:], mode="bilinear", align_corners=False
        )
        x = torch.cat([b0, b1, b2, b3, gap], dim=1)
        x = self.aspp["proj"](x)
        return self.density_head(x)


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
        self.crowd_model = None        # CSRNet / SANet (or "onnx" sentinel when using ONNX)
        self.crowd_onnx_session = None # ONNX Runtime session for crowd density
        self._crowd_onnx_input_name: Optional[str] = None  # first ONNX input name (export may differ from "input")
        self._crowd_arch: str = "csrnet"
        self._crowd_density_scale: Optional[float] = None
        self._crowd_calib_ratio_samples: List[float] = []  # median over N frames for stable scale
        self._crowd_calibration_yolo = None  # YOLOv8n for SANet auto-calibration
        self._crowd_cal_yolo_attempts = 0  # max 2 tries (startup + first crowd frame)
        self._crowd_calib_lock = threading.Lock()
        self._crowd_skip_counter = 0
        self._last_crowd_result: Optional[Dict[str, Any]] = None
        self._crowd_future = None  # async crowd task (non-blocking collection in detect_all_parallel)
        self._crowd_disabled_warned = False  # avoid spamming when module is disabled by settings
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
            _try_move_yolo_to_cuda(model, model_name)
            logger.info("%s loaded from PyTorch: %s — classes: %s", model_name, pt_path, model.names)
            return model, None
        except Exception as e:
            logger.error("Failed to load %s: %s", model_name, e)
            return None, None

    def _load_crowd_model(self, crowd_path: Path):
        """Load CSRNet or SANet from settings (``CROWD_MODEL_ARCH``)."""
        arch = getattr(settings, "CROWD_MODEL_ARCH", "csrnet").lower().strip()
        if arch not in ("csrnet", "sanet"):
            logger.warning("Unknown CROWD_MODEL_ARCH=%s — using csrnet", arch)
            arch = "csrnet"
        self._crowd_arch = arch
        if arch == "sanet":
            self._load_sanet_onnx(crowd_path)
        else:
            self._load_csrnet_onnx(crowd_path)

    def _bind_crowd_onnx_session(self, session) -> None:
        """Store session and the actual input tensor name from the ONNX graph."""
        self.crowd_onnx_session = session
        self.crowd_model = "onnx"
        self._crowd_onnx_input_name = session.get_inputs()[0].name
        logger.info("Crowd ONNX input tensor: %s", self._crowd_onnx_input_name)

    def _load_csrnet_onnx(self, crowd_path: Path):
        """
        Load CSRNet. Prefer ONNX Runtime (faster on CPU than PyTorch);
        export once if ``.onnx`` is missing.
        """
        if crowd_path.suffix.lower() == ".onnx":
            onnx_path = crowd_path
        else:
            onnx_path = crowd_path.with_suffix(".onnx")

        if ORT_AVAILABLE and onnx_path.exists():
            try:
                self._bind_crowd_onnx_session(_make_ort_session(onnx_path))
                logger.info("CSRNet loaded from ONNX: %s", onnx_path)
                return
            except Exception as e:
                logger.warning("Failed to load CSRNet ONNX, falling back to PyTorch: %s", e)

        if crowd_path.suffix.lower() == ".onnx":
            self.crowd_model = None
            logger.error("CSRNet: ONNX load failed and no PyTorch weights at %s", crowd_path)
            return

        try:
            pt_model = CSRNet()
            checkpoint = torch.load(str(crowd_path), map_location="cpu", weights_only=False)
            pt_model.load_state_dict(checkpoint["model_state"])
            pt_model.eval()
            logger.info(
                "CSRNet loaded from PyTorch: %s (epoch %s, MAE %.2f)",
                crowd_path,
                checkpoint.get("epoch"),
                checkpoint.get("best_mae", 0),
            )
        except Exception as e:
            self.crowd_model = None
            logger.error("Failed to load CSRNet: %s", e)
            return

        if ORT_AVAILABLE and not onnx_path.exists():
            try:
                logger.info("Exporting CSRNet to ONNX (one-time operation)...")
                dummy = torch.randn(1, 3, 384, 512)
                torch.onnx.export(
                    pt_model,
                    dummy,
                    str(onnx_path),
                    opset_version=17,
                    input_names=["input"],
                    output_names=["density_map"],
                    dynamic_axes={"input": {2: "height", 3: "width"}},
                )
                self._bind_crowd_onnx_session(_make_ort_session(onnx_path))
                logger.info("CSRNet ONNX export complete: %s", onnx_path)
                return
            except Exception as e:
                logger.warning("CSRNet ONNX export failed, using PyTorch: %s", e)

        self.crowd_model = pt_model

    def _load_sanet_pytorch_weights(self, pt_path: Path) -> Optional[SANet]:
        """Load SANet ``state_dict`` from a ``.pth`` / ``.pt`` checkpoint."""
        try:
            pt_model = SANet()
            checkpoint = torch.load(str(pt_path), map_location="cpu", weights_only=False)
            if "state_dict" in checkpoint:
                sd = _strip_module_prefix(checkpoint["state_dict"])
            elif "model_state" in checkpoint:
                sd = _strip_module_prefix(checkpoint["model_state"])
            else:
                sd = _strip_module_prefix(checkpoint)
            pt_model.load_state_dict(sd, strict=True)
            pt_model.eval()
            mae = checkpoint.get("mae", checkpoint.get("best_mae", 0))
            logger.info("SANet loaded from PyTorch: %s (MAE %.2f)", pt_path, mae)
            return pt_model
        except Exception as e:
            logger.error("Failed to load SANet PyTorch weights %s: %s", pt_path, e)
            return None

    def _export_sanet_onnx_single_file(self, pt_model: SANet, out_path: Path) -> bool:
        """Write one self-contained ONNX (no ``.onnx.data``). Returns True if ORT session bound."""
        if not ORT_AVAILABLE:
            return False
        dummy = torch.randn(1, 3, 384, 512)
        tmp = out_path.with_suffix(".tmp.onnx")
        try:
            export_kw: Dict[str, Any] = {
                "opset_version": 17,
                "input_names": ["input"],
                "output_names": ["density_map"],
                "dynamic_axes": {"input": {2: "height", 3: "width"}},
            }
            try:
                torch.onnx.export(
                    pt_model, dummy, str(tmp), dynamo=False, **export_kw
                )
            except TypeError:
                torch.onnx.export(pt_model, dummy, str(tmp), **export_kw)
            tmp.replace(out_path)
            self._bind_crowd_onnx_session(_make_ort_session(out_path))
            logger.info("SANet single-file ONNX ready (ORT): %s", out_path)
            return True
        except Exception as e:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            logger.warning("SANet ONNX export failed: %s", e)
            return False

    def _load_sanet_onnx(self, crowd_path: Path):
        """Load SANet from ``.onnx`` or ``.pth`` (``state_dict``). Prefer ONNX on CPU."""
        if crowd_path.suffix.lower() == ".onnx":
            onnx_path = crowd_path
        else:
            onnx_path = crowd_path.with_suffix(".onnx")

        ext_data_path = onnx_path.parent / (onnx_path.name + ".data")
        pt_candidates = [onnx_path.with_suffix(".pth"), onnx_path.with_suffix(".pt")]
        pt_path = next((p for p in pt_candidates if p.is_file()), None)

        # 1) Try existing ONNX (must be single-file or accompanied by .onnx.data)
        if ORT_AVAILABLE and onnx_path.exists():
            try:
                self._bind_crowd_onnx_session(_make_ort_session(onnx_path))
                logger.info("SANet loaded from ONNX: %s", onnx_path)
                return
            except Exception as e:
                err_txt = str(e).lower()
                logger.warning("Failed to load SANet ONNX: %s", e)
                if (
                    ".data" in err_txt
                    or "external" in err_txt
                    or "file_size" in err_txt
                    or "cannot find the file" in err_txt
                ):
                    logger.error(
                        "Split ONNX needs weight file %s or PyTorch weights (.pth). "
                        "Place sanet_partB_best.pth beside the onnx, or run: "
                        "python scripts/export_sanet_onnx_from_pth.py",
                        ext_data_path,
                    )
                if not ext_data_path.exists() and onnx_path.exists():
                    logger.error("Missing external weights: %s", ext_data_path)

        # 2) ONNX failed or missing: load PyTorch checkpoint (same basename as .onnx, or explicit .pth path)
        pt_source: Optional[Path] = None
        if crowd_path.suffix.lower() in (".pth", ".pt", ".pth.tar"):
            pt_source = crowd_path
        elif pt_path is not None:
            pt_source = pt_path

        pt_model: Optional[SANet] = None
        if pt_source is not None and pt_source.is_file():
            pt_model = self._load_sanet_pytorch_weights(pt_source)

        if pt_model is None:
            self.crowd_model = None
            logger.error(
                "SANet: no usable weights. Add models/sanet_partB_best.pth (Kaggle checkpoint) next to "
                "the broken ONNX, or add the missing .onnx.data file, or run "
                "python scripts/export_sanet_onnx_from_pth.py",
            )
            return

        # 3) Rewrite a single-file ONNX from PyTorch so ORT works without .data
        if ORT_AVAILABLE and onnx_path.suffix.lower() == ".onnx":
            if self._export_sanet_onnx_single_file(pt_model, onnx_path):
                return

        # 4) No ORT export: keep PyTorch
        self.crowd_model = pt_model

    def _install_calibration_yolo(self) -> None:
        """Load YOLOv8n for SANet calibration (up to 2 attempts: startup + first crowd inference)."""
        if self._crowd_calibration_yolo is not None:
            return
        if self._crowd_arch != "sanet":
            return
        if not getattr(settings, "CROWD_AUTO_CALIBRATE", True):
            return
        if self._crowd_cal_yolo_attempts >= 2:
            return
        self._crowd_cal_yolo_attempts += 1
        cal_spec = getattr(
            settings, "CROWD_CALIBRATION_YOLO_PATH", "./models/yolov8n.pt"
        ).strip()
        cal_path = Path(cal_spec)
        if cal_path.is_file():
            load_target = str(cal_path.resolve())
        else:
            load_target = "yolov8n.pt"
            if cal_spec:
                logger.info(
                    "Crowd calibration: no file at %r — using %r (Ultralytics will download if needed)",
                    cal_spec,
                    load_target,
                )
        try:
            from ultralytics import YOLO

            self._crowd_calibration_yolo = YOLO(load_target)
            _try_move_yolo_to_cuda(self._crowd_calibration_yolo, "Crowd calibration YOLO")
            logger.info("Crowd calibration YOLO ready: %s", load_target)
        except Exception as e:
            logger.warning(
                "Calibration YOLO attempt %d/2 failed (%s): %s",
                self._crowd_cal_yolo_attempts,
                load_target,
                e,
            )

    def _load_crowd_calibration_yolo(self):
        """Schedule calibration YOLO at startup (may retry on first crowd frame via _install_calibration_yolo)."""
        self._install_calibration_yolo()

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

        # --- 3. Crowd density (CSRNet or SANet) ---
        crowd_path = Path(settings.CROWD_MODEL_PATH)
        if crowd_path.exists():
            self._load_crowd_model(crowd_path)
            self._load_crowd_calibration_yolo()
        else:
            logger.warning("Crowd model not found at %s — crowd density disabled", crowd_path)

        # --- 4. Conv3D Violence classifier (trained on Real Life Violence dataset) ---
        violence_onnx_path = Path("./models/violence_model.onnx")
        violence_pt_path = Path("./models/violence_model.pt")
        if ORT_AVAILABLE and violence_onnx_path.exists():
            try:
                self.violence_onnx_session = _make_ort_session(violence_onnx_path)
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
                    self.violence_onnx_session = _make_ort_session(violence_onnx_path)
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

    def _finalize_async_crowd_result(self, wait_for_first: bool = False) -> None:
        """
        Collect completed thread-pool crowd work into ``_last_crowd_result``.

        When ``wait_for_first`` is True and we have never stored a result yet, block up to
        ``CROWD_INFERENCE_TIMEOUT_SECONDS`` so the first analyzed frame (or a one-frame
        job) still receives a crowd row. Without this, async results only appeared on the
        *next* call to ``detect_all_parallel``.
        """
        if self._crowd_future is None:
            return
        fut = self._crowd_future
        if fut.done():
            try:
                self._last_crowd_result = fut.result()
            except Exception as e:
                logger.warning("Async crowd_density failed: %s", e)
            finally:
                self._crowd_future = None
            return
        if wait_for_first and self._last_crowd_result is None:
            timeout = float(getattr(settings, "CROWD_INFERENCE_TIMEOUT_SECONDS", 60))
            try:
                self._last_crowd_result = fut.result(timeout=timeout)
            except FuturesTimeoutError:
                logger.warning(
                    "Crowd inference timed out after %ss — no crowd result yet; will retry next frame",
                    timeout,
                )
            except Exception as e:
                logger.warning("Async crowd_density failed: %s", e)
            finally:
                self._crowd_future = None

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
        # Also do not skip when crowd_density is enabled: similarity would return [] before
        # crowd runs, so videos looked like they had no crowd at all.
        need_yolo = "weapon" in enabled_modules or "abandoned_object" in enabled_modules
        similar = self.is_frame_similar(frame)
        if similar and not need_yolo and "crowd_density" not in enabled_modules:
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
            self._crowd_disabled_warned = False
            prev_crowd_result = self._last_crowd_result
            # Pick up async result from the previous analyzed frame (if it finished).
            self._finalize_async_crowd_result(wait_for_first=False)

            crowd_every_n = max(
                1, int(getattr(settings, "CROWD_EVERY_N_ANALYSIS_FRAMES", 1))
            )
            self._crowd_skip_counter = (self._crowd_skip_counter + 1) % crowd_every_n
            should_run_crowd = (
                self._crowd_skip_counter == 0 or self._last_crowd_result is None
            )

            if should_run_crowd and self._crowd_future is None:
                # Submit crowd model asynchronously; do not block frame loop.
                crowd_frame = full_frame if full_frame is not None else frame
                self._crowd_future = self._inference_pool.submit(
                    self.calculate_crowd_density, crowd_frame, frame_timestamp
                )

            # Same-frame completion + first-result wait (single-frame / first analyzed frame).
            self._finalize_async_crowd_result(wait_for_first=True)

            # Reuse previous crowd result every frame (until next async result arrives).
            if self._last_crowd_result is not None:
                crowd_out = dict(self._last_crowd_result)
                is_fresh_result = (
                    prev_crowd_result is None or prev_crowd_result is not self._last_crowd_result
                )
                # Mark as cached only when we are reusing the exact previous value.
                if not is_fresh_result:
                    crowd_out["_crowd_cached"] = True
                all_results.append((DetectionType.CROWD_DENSITY, crowd_out))
        elif getattr(settings, "CROWD_DEBUG_LOG", False) and not self._crowd_disabled_warned:
            logger.warning(
                "[crowd debug] crowd_density is not enabled for this company/job. enabled_modules=%s",
                sorted(enabled_modules.keys()),
            )
            self._crowd_disabled_warned = True

        if "violence" in enabled_modules:
            futures["violence"] = self._inference_pool.submit(
                self.detect_violence, frame, previous_frames, frame_timestamp
            )

        # Step 3: Collect results
        for module, future in futures.items():
            try:
                timeout_s = 30
                result = future.result(timeout=timeout_s)

                if module == "weapon":
                    for det in result:
                        all_results.append((DetectionType.WEAPON, det))
                elif module == "abandoned_object":
                    for det in result:
                        all_results.append((DetectionType.ABANDONED_OBJECT, det))
                elif module == "mask_face":
                    for det in result:
                        all_results.append((DetectionType.MASK_FACE, det))
                elif module == "violence":
                    if result["is_violent"]:
                        all_results.append((DetectionType.VIOLENCE, result))
            except FuturesTimeoutError:
                future.cancel()
                logger.warning(
                    "Module %s timed out after %ss", module, timeout_s
                )
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
    # 5. Crowd density estimation  (CSRNet / SANet)
    # ==================================================================
    @staticmethod
    def _yolo_coco_person_count(results: Any) -> int:
        """Count COCO class 0 (person) like the SANet notebook loop."""
        if not results or len(results) == 0:
            return 0
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return 0
        n = 0
        for b in boxes:
            try:
                t = b.cls
                cid = int(t.item()) if hasattr(t, "item") else int(t[0])
            except (ValueError, TypeError, IndexError, RuntimeError):
                continue
            if cid == 0:
                n += 1
        return n

    def _calculate_sanet_crowd_density(
        self,
        frame: np.ndarray,
        frame_timestamp: datetime,
    ) -> Dict[str, Any]:
        """SANet: RGB/255; ONNX on CPU. Auto-calibration uses YOLO on the same resize as SANet unless full-frame."""
        h0, w0 = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        use_full_frame = getattr(settings, "CROWD_SANET_FULL_FRAME", False)
        notebook_cal = getattr(settings, "CROWD_SANET_NOTEBOOK_CALIBRATION", True)

        if use_full_frame:
            img = frame_rgb
            nw, nh = w0, h0
            yolo_cal_frame = frame
        else:
            max_side = max(32, int(settings.CROWD_INFERENCE_MAX_SIDE))
            scale_r = min(max_side / w0, max_side / h0)
            nw, nh = max(1, int(w0 * scale_r)), max(1, int(h0 * scale_r))
            img = cv2.resize(frame_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
            yolo_cal_frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)

        input_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

        density_map: Optional[np.ndarray] = None
        density_sum: float
        inp_name = self._crowd_onnx_input_name or "input"
        if self.crowd_onnx_session is not None:
            outputs = self.crowd_onnx_session.run(
                None, {inp_name: input_tensor.numpy()}
            )
            density_map = outputs[0]
            density_sum = float(np.sum(density_map))
        else:
            with torch.no_grad():
                density_tensor = self.crowd_model(input_tensor)
            density_map = density_tensor.cpu().numpy()
            density_sum = float(density_tensor.sum().item())

        raw_density_map = np.squeeze(density_map)

        self._install_calibration_yolo()

        cal_conf = float(getattr(settings, "CROWD_CALIBRATION_YOLO_CONF", 0.45))
        cal_imgsz = max(32, int(getattr(settings, "CROWD_CALIBRATION_YOLO_IMGSZ", 960)))
        n_cal_frames = max(1, int(getattr(settings, "CROWD_CALIBRATION_SAMPLE_FRAMES", 8)))
        expected_ct = getattr(settings, "CROWD_CALIBRATION_EXPECTED_COUNT", None)
        sum_ok_strict = float(density_sum) > 1e-5 and np.isfinite(density_sum)
        sum_ok_notebook = float(density_sum) > 0 and np.isfinite(density_sum)

        cal_final: Optional[float] = None
        cal_interim: List[float] = []
        with self._crowd_calib_lock:
            if getattr(settings, "CROWD_AUTO_CALIBRATE", True) and self._crowd_density_scale is None:
                # ── Notebook script: yolo(frame), cls==0, SCALE = persons / raw_sum (same resize as SANet) ──
                if notebook_cal and sum_ok_notebook:
                    if expected_ct is not None and int(expected_ct) > 0:
                        self._crowd_density_scale = float(int(expected_ct)) / float(density_sum)
                        logger.info(
                            "SANet calibration (expected count): count=%d raw_sum=%.4f scale=%.8f",
                            int(expected_ct),
                            density_sum,
                            self._crowd_density_scale,
                        )
                    elif self._crowd_calibration_yolo is not None:
                        try:
                            results = self._crowd_calibration_yolo(
                                yolo_cal_frame, verbose=False
                            )
                            person_count_yolo = self._yolo_coco_person_count(results)
                        except Exception as e:
                            logger.exception("SANet calibration YOLO inference failed: %s", e)
                            person_count_yolo = 0
                        self._crowd_density_scale = (
                            person_count_yolo / float(density_sum)
                            if density_sum > 0
                            else 1.0
                        )
                        logger.info(
                            "SANet auto-calibration: YOLO persons=%d raw_sum=%.4f scale=%.8f "
                            "calibrated=%.2f (sanet_input=%dx%d)",
                            person_count_yolo,
                            density_sum,
                            self._crowd_density_scale,
                            float(density_sum) * self._crowd_density_scale,
                            nw,
                            nh,
                        )
                # ── Multi-frame median (only when notebook calibration disabled) ──
                elif not notebook_cal and sum_ok_strict:
                    ratio: Optional[float] = None
                    if expected_ct is not None and int(expected_ct) > 0:
                        ratio = float(int(expected_ct)) / float(density_sum)
                    elif self._crowd_calibration_yolo is not None:
                        try:
                            results = self._crowd_calibration_yolo(
                                yolo_cal_frame,
                                verbose=False,
                                classes=[0],
                                conf=cal_conf,
                                imgsz=cal_imgsz,
                            )
                            person_count_yolo = len(results[0].boxes) if results[0].boxes is not None else 0
                        except Exception as e:
                            logger.exception("SANet multi-frame calibration YOLO failed: %s", e)
                            person_count_yolo = 0
                        ratio = person_count_yolo / float(density_sum)
                    if ratio is not None and np.isfinite(ratio) and 0 < ratio < 1e6:
                        self._crowd_calib_ratio_samples.append(float(ratio))
                        if len(self._crowd_calib_ratio_samples) >= n_cal_frames:
                            self._crowd_density_scale = float(
                                np.median(np.array(self._crowd_calib_ratio_samples, dtype=np.float64))
                            )
                            logger.info(
                                "SANet calibration (median n=%d): scale=%.8f samples=%s",
                                n_cal_frames,
                                self._crowd_density_scale,
                                [round(x, 8) for x in self._crowd_calib_ratio_samples],
                            )
                            self._crowd_calib_ratio_samples.clear()

            cal_final = self._crowd_density_scale
            cal_interim = list(self._crowd_calib_ratio_samples)

        if cal_final is not None:
            cal_scale = cal_final
        elif cal_interim:
            cal_scale = float(np.median(np.array(cal_interim, dtype=np.float64)))
        else:
            cal_scale = float(getattr(settings, "CROWD_CALIBRATION_SCALE", 1.0))
        raw_count = max(0.0, density_sum * cal_scale)
        person_count = max(0, int(round(raw_count)))
        density = min(1.0, person_count / 50.0)

        # Heatmap matches SANet notebook: resize density → full frame, clip, / max → uint8 (then JET in video_service)
        hm = cv2.resize(
            raw_density_map.astype(np.float32),
            (w0, h0),
            interpolation=cv2.INTER_LINEAR,
        )
        hm = np.clip(hm, 0, None)
        if hm.size > 0 and float(hm.max()) > 0:
            density_normalized = np.uint8(255.0 * hm / hm.max())
        else:
            density_normalized = np.zeros((h0, w0), dtype=np.uint8)

        return {
            "count": person_count,
            "density": round(density, 4),
            "confidence": round(density, 4),
            "density_map_normalized": density_normalized,
        }

    def calculate_crowd_density(
        self,
        frame: np.ndarray,
        frame_timestamp: datetime,
    ) -> Dict[str, Any]:
        """
        Estimate crowd count and density using CSRNet, SANet, or HOG fallback.

        Returns: dict with count, density, confidence, and density_map_normalized
        """
        if self.crowd_model is not None:
            try:
                if getattr(self, "_crowd_arch", "csrnet") == "sanet":
                    return self._calculate_sanet_crowd_density(
                        frame, frame_timestamp
                    )
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
                    in_name = self._crowd_onnx_input_name or "input"
                    outputs = self.crowd_onnx_session.run(None, {in_name: input_array})
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
        DetectionType.CROWD_DENSITY:    (None, 0.9, 0.7),    # Uses count/scaled or density (see classify_threat_level)
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
        crowd_count = float((metadata or {}).get("count", 0) or 0)
        crowd_scale = float(getattr(settings, "CROWD_THREAT_MAX_PEOPLE_SCALE", 100.0))
        use_people_count = bool(getattr(settings, "CROWD_THREAT_USE_PEOPLE_COUNT", True))
        if detection_type == DetectionType.CROWD_DENSITY:
            if use_people_count:
                value = min(1.0, crowd_count / crowd_scale) if crowd_scale > 0 else 0.0
            else:
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

        # Crowd supports two threshold styles when using people-count mode:
        # - Absolute people counts (e.g. 20 / 100 / 1000)
        # - Legacy 0..1 normalized values (converted by CROWD_THREAT_MAX_PEOPLE_SCALE)
        if detection_type == DetectionType.CROWD_DENSITY and use_people_count:
            def to_people_threshold(t: Optional[float]) -> Optional[float]:
                if t is None:
                    return None
                return float(t) if float(t) > 1.0 else float(t) * max(crowd_scale, 0.0)

            ct_people = to_people_threshold(ct)
            ht_people = to_people_threshold(ht)
            mt_people = to_people_threshold(mt)

            if ct_people is not None and crowd_count >= ct_people:
                return ThreatLevel.CRITICAL
            if ht_people is not None and crowd_count >= ht_people:
                return ThreatLevel.HIGH
            if mt_people is not None and crowd_count >= mt_people:
                return ThreatLevel.MEDIUM
            return ThreatLevel.LOW

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
