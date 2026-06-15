"""
Video processing service for handling video streams and files.
"""

import logging
from collections.abc import Generator
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)


class VideoService:
    """Service for processing video streams and files."""

    # Color mapping per detection type (BGR)
    DETECTION_COLORS = {
        "weapon": (0, 0, 255),  # Red
        "mask_face": (0, 165, 255),  # Orange
        "crowd_density": (255, 255, 0),  # Cyan
        "abandoned_object": (0, 255, 255),  # Yellow
        "violence": (128, 0, 128),  # Purple
    }

    def __init__(self):
        self.upload_dir = Path(settings.UPLOAD_DIR)
        self.evidence_dir = Path(settings.EVIDENCE_DIR)
        self.processed_dir = Path(settings.PROCESSED_VIDEO_DIR)

    def read_video_file(
        self, video_path: str
    ) -> Generator[tuple[np.ndarray, datetime], None, None]:
        """
        Read frames from a video file.

        Yields:
            Tuple of (frame, timestamp) where timestamp is calculated
            from the video's FPS so that frame positions are preserved.
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = 0
        video_start = datetime.now()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Calculate timestamp relative to video start using FPS
                from datetime import timedelta

                elapsed = timedelta(seconds=frame_count / fps)
                timestamp = video_start + elapsed
                frame_count += 1

                yield frame, timestamp
        finally:
            cap.release()

    def read_stream(self, stream_url: str) -> Generator[tuple[np.ndarray, datetime], None, None]:
        """
        Read frames from a live stream (RTSP, HTTP, etc.).

        Yields:
            Tuple of (frame, timestamp)
        """
        cap = cv2.VideoCapture(stream_url)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    import time

                    time.sleep(0.1)  # Wait before retrying
                    continue

                timestamp = datetime.now()
                yield frame, timestamp
        finally:
            cap.release()

    def save_snapshot(
        self,
        frame: np.ndarray,
        detection_id: int,
        prefix: str = "snapshot",
        bbox: list = None,
        label: str = None,
    ) -> str:
        """
        Save a frame as a snapshot image with bounding box drawn.

        Args:
            bbox: [x_norm, y_norm, w_norm, h_norm] in 0-1 normalized coords
            label: Text label to draw above the box (e.g. "Weapons 87%")

        Returns:
            Path to saved image
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        colors = {
            "weapon": (0, 0, 255),  # Red
            "mask_face": (255, 165, 0),  # Orange
            "crowd_density": (255, 255, 0),  # Cyan
            "abandoned_object": (0, 255, 255),  # Yellow
            "violence": (128, 0, 128),  # Purple
        }
        color = colors.get(prefix, (0, 255, 0))

        has_bbox = bbox and len(bbox) >= 4 and float(bbox[2]) > 1e-6 and float(bbox[3]) > 1e-6
        if has_bbox:
            x1 = int(bbox[0] * w)
            y1 = int(bbox[1] * h)
            x2 = int((bbox[0] + bbox[2]) * w)
            y2 = int((bbox[1] + bbox[3]) * h)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            if label:
                font_scale = 0.6
                thickness = 2
                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                )
                cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
                cv2.putText(
                    annotated,
                    label,
                    (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                )
        elif label:
            font_scale = 0.6
            thickness = 2
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            lx, ly = 8, th + 12
            cv2.rectangle(annotated, (lx - 4, ly - th - 8), (lx + tw + 4, ly + 4), color, -1)
            cv2.putText(
                annotated,
                label,
                (lx, ly),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness,
            )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{detection_id}_{timestamp}.jpg"
        filepath = self.evidence_dir / filename

        cv2.imwrite(str(filepath), annotated)
        return str(filepath)

    def extract_video_clip(
        self, video_path: str, start_frame: int, end_frame: int, output_path: str
    ) -> bool:
        """
        Extract a video clip from a larger video file.

        Returns:
            True if successful
        """
        try:
            cap = cv2.VideoCapture(video_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            for _ in range(start_frame, end_frame + 1):
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)

            cap.release()
            out.release()
            return True
        except Exception as e:
            print(f"Error extracting video clip: {e}")
            return False

    def get_video_info(self, video_path: str) -> dict:
        """Get video metadata: total frames, fps, width, height, duration."""
        cap = cv2.VideoCapture(video_path)
        info = {
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": cap.get(cv2.CAP_PROP_FPS) or 30.0,
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
        info["duration_seconds"] = info["total_frames"] / info["fps"] if info["fps"] else 0
        cap.release()
        return info

    @staticmethod
    def iou_normalized(a: list[float], b: list[float]) -> float:
        """IoU for boxes as [x, y, w, h] in 0–1 normalized coordinates."""
        if len(a) < 4 or len(b) < 4:
            return 0.0
        ax2, ay2 = a[0] + a[2], a[1] + a[3]
        bx2, by2 = b[0] + b[2], b[1] + b[3]
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        ua = a[2] * a[3] + b[2] * b[3] - inter
        return inter / ua if ua > 0 else 0.0

    def blur_faces_except(
        self,
        frame: np.ndarray,
        preserve_normalized: list[list[float]],
        face_boxes_normalized: list[list[float]],
        blur_ksize: int = 51,
        iou_keep: float = 0.12,
    ) -> np.ndarray:
        """
        Gaussian-blur each face region except those overlapping ``preserve_normalized``
        (subject faces / detections to keep sharp).
        """
        if not face_boxes_normalized:
            return frame
        out = frame.copy()
        h, w = out.shape[:2]
        k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        k = max(3, k)
        for fb in face_boxes_normalized:
            if len(fb) < 4:
                continue
            skip = False
            for pb in preserve_normalized:
                if len(pb) < 4:
                    continue
                if self.iou_normalized(fb, pb) >= iou_keep:
                    skip = True
                    break
            if skip:
                continue
            x1 = int(max(0, fb[0] * w))
            y1 = int(max(0, fb[1] * h))
            x2 = int(min(w, (fb[0] + fb[2]) * w))
            y2 = int(min(h, (fb[1] + fb[3]) * h))
            if x2 <= x1 or y2 <= y1:
                continue
            roi = out[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            out[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)
        return out

    def privacy_blur_for_snapshot(
        self,
        frame: np.ndarray,
        det_type_value: str,
        subject_bbox: Optional[list] = None,
        face_boxes_normalized: Optional[list[list[float]]] = None,
    ) -> np.ndarray:
        """Blur faces except the subject; for ``mask_face``, keep the detection bbox sharp."""
        if not getattr(settings, "PRIVACY_BLUR_NON_SUBJECT_FACES", True):
            return frame
        faces = face_boxes_normalized or []
        if not faces:
            return frame
        preserve: list[list[float]] = []
        bb = subject_bbox or []
        if det_type_value == "mask_face" and len(bb) >= 4 and float(bb[2]) > 0 and float(bb[3]) > 0:
            preserve = [bb]
        return self.blur_faces_except(
            frame,
            preserve,
            faces,
            blur_ksize=int(getattr(settings, "PRIVACY_FACE_BLUR_KSIZE", 51)),
            iou_keep=float(getattr(settings, "PRIVACY_FACE_IOU_KEEP", 0.12)),
        )

    def privacy_blur_for_video_frame(
        self,
        frame: np.ndarray,
        preserve_mask_face_bboxes: list[list[float]],
        face_boxes_normalized: Optional[list[list[float]]] = None,
    ) -> np.ndarray:
        """Blur all faces except those overlapping mask_face detection boxes on this frame."""
        if not getattr(settings, "PRIVACY_BLUR_IN_VIDEO_OUTPUT", True):
            return frame
        faces = face_boxes_normalized or []
        if not faces:
            return frame
        return self.blur_faces_except(
            frame,
            preserve_mask_face_bboxes,
            faces,
            blur_ksize=int(getattr(settings, "PRIVACY_FACE_BLUR_KSIZE", 51)),
            iou_keep=float(getattr(settings, "PRIVACY_FACE_IOU_KEEP", 0.12)),
        )

    def draw_detections_on_frame(
        self, frame: np.ndarray, detections: list, heatmap_overlay: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Draw bounding boxes and labels on a frame. Optionally overlay crowd density heatmap.

        Args:
            detections: list of dicts with keys:
                det_type (str), confidence (float), bbox [x,y,w,h] normalized,
                class_name (str, optional)
            heatmap_overlay: optional (height, width, 3) BGR density heatmap to blend
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # Apply heatmap overlay first (if crowd density detection exists)
        if heatmap_overlay is not None and heatmap_overlay.shape[:2] == (h, w):
            fw = float(getattr(settings, "CROWD_HEATMAP_FRAME_WEIGHT", 0.65))
            fw = min(max(fw, 0.0), 1.0)
            annotated = cv2.addWeighted(annotated, fw, heatmap_overlay, 1.0 - fw, 0)

        # Draw crowd count/density text from crowd_density detections
        for det in detections:
            if det.get("det_type") == "crowd_density":
                count = det.get("count", 0)
                density_val = det.get("density", 0.0)
                # Draw background box for text
                cv2.rectangle(annotated, (10, 10), (300, 90), (0, 0, 0), -1)
                cv2.rectangle(annotated, (10, 10), (300, 90), (0, 255, 255), 2)
                cv2.putText(
                    annotated,
                    f"People: {count}",
                    (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    annotated,
                    f"Density: {int(density_val * 100)}%",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                break  # only one crowd_density per frame

        # Draw bounding boxes for non-crowd detections
        for det in detections:
            det_type = det.get("det_type", "")

            # Skip crowd_density — we already drew the heatmap + text
            if det_type == "crowd_density":
                continue

            bbox = det.get("bbox", [])
            if not bbox or len(bbox) < 4:
                continue
            if float(bbox[2]) < 1e-6 or float(bbox[3]) < 1e-6:
                continue

            x1 = int(bbox[0] * w)
            y1 = int(bbox[1] * h)
            x2 = int((bbox[0] + bbox[2]) * w)
            y2 = int((bbox[1] + bbox[3]) * h)

            color = self.DETECTION_COLORS.get(det_type, (0, 255, 0))
            confidence = det.get("confidence", 0)
            class_name = det.get("class_name", det_type.replace("_", " ").title())

            # Draw box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Draw label background + text
            label = f"{class_name} {confidence:.0%}"
            font_scale = 0.6
            thickness = 2
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                annotated,
                label,
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness,
            )

            # Draw tracking line from bottom-center of bbox to bottom of frame
            center_x = (x1 + x2) // 2
            cv2.line(annotated, (center_x, y2), (center_x, h), color, 1, cv2.LINE_AA)

        return annotated

    def generate_crowd_heatmap(
        self,
        frame: np.ndarray,
        density_map_normalized: Optional[np.ndarray],
        count: int = 0,
        density: float = 0.0,
    ) -> Optional[np.ndarray]:
        """
        Generate a crowd density heatmap overlay from normalized density map.

        Args:
            frame: original frame (for resizing density map)
            density_map_normalized: (height, width) normalized density map (0-255)
            count: person count to display
            density: density ratio to display

        Returns:
            (frame_h, frame_w, 3) BGR heatmap overlay or None if input invalid
        """
        if density_map_normalized is None:
            return None

        h, w = frame.shape[:2]

        # Resize density map to match frame size (LINEAR matches SANet notebook resize)
        if density_map_normalized.shape != (h, w):
            density_resized = cv2.resize(
                density_map_normalized, (w, h), interpolation=cv2.INTER_LINEAR
            )
        else:
            density_resized = density_map_normalized

        cmap_name = getattr(settings, "CROWD_HEATMAP_COLORMAP", "jet").lower()
        cmap = {
            "jet": cv2.COLORMAP_JET,
            "hot": cv2.COLORMAP_HOT,
            "inferno": cv2.COLORMAP_INFERNO,
            "viridis": cv2.COLORMAP_VIRIDIS,
        }.get(cmap_name, cv2.COLORMAP_JET)
        heatmap_colored = cv2.applyColorMap(density_resized, cmap)

        # Add text overlay with crowd count and density percentage
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        thickness = 2
        color = (255, 255, 255)  # White text

        # Count label
        count_text = f"People: {count}"
        cv2.putText(heatmap_colored, count_text, (20, 50), font, font_scale, color, thickness)

        # Density percentage label
        density_percent = min(100, int(density * 100))
        density_text = f"Density: {density_percent}%"
        cv2.putText(heatmap_colored, density_text, (20, 100), font, font_scale, color, thickness)

        return heatmap_colored

    def create_video_writer(self, output_path: str, fps: float, width: int, height: int):
        """Create a VideoWriter for the output annotated video."""
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        return cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    def reencode_to_h264(self, input_path: str) -> str:
        """
        Re-encode an mp4v video to H.264 so browsers can play it inline.
        Replaces the original file. Returns the final path.
        """
        import os
        import shutil
        import subprocess

        # Find ffmpeg: system install or bundled via imageio-ffmpeg
        ffmpeg_bin = shutil.which("ffmpeg")
        if not ffmpeg_bin:
            try:
                import imageio_ffmpeg

                ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
            except (ImportError, Exception):
                ffmpeg_bin = None

        if not ffmpeg_bin:
            logger.warning("ffmpeg not available — video will not be browser-playable")
            return input_path

        temp_path = input_path + ".h264.mp4"
        try:
            subprocess.run(
                [
                    ffmpeg_bin,
                    "-y",
                    "-i",
                    input_path,
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    "23",
                    "-pix_fmt",
                    "yuv420p",  # maximum browser compat
                    "-movflags",
                    "+faststart",  # enables streaming/seeking
                    "-an",  # no audio track
                    temp_path,
                ],
                check=True,
                capture_output=True,
                timeout=300,
            )
            # Replace original with h264 version
            os.replace(temp_path, input_path)
            logger.info("Re-encoded %s to H.264 successfully", input_path)
        except Exception as e:
            logger.warning("ffmpeg re-encode failed: %s", e)
            if os.path.exists(temp_path):
                os.remove(temp_path)

        return input_path


video_service = VideoService()
