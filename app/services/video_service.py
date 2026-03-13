"""
Video processing service for handling video streams and files.
"""
import cv2
import numpy as np
from typing import Generator, Optional, Tuple
from pathlib import Path
from datetime import datetime
from app.config import settings


class VideoService:
    """Service for processing video streams and files."""

    # Color mapping per detection type (BGR)
    DETECTION_COLORS = {
        "weapon": (0, 0, 255),          # Red
        "mask_face": (0, 165, 255),     # Orange
        "crowd_density": (255, 255, 0), # Cyan
        "abandoned_object": (0, 255, 255), # Yellow
        "violence": (128, 0, 128),      # Purple
    }

    def __init__(self):
        self.upload_dir = Path(settings.UPLOAD_DIR)
        self.evidence_dir = Path(settings.EVIDENCE_DIR)
        self.processed_dir = Path(settings.PROCESSED_VIDEO_DIR)
    
    def read_video_file(self, video_path: str) -> Generator[Tuple[np.ndarray, datetime], None, None]:
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
    
    def read_stream(self, stream_url: str) -> Generator[Tuple[np.ndarray, datetime], None, None]:
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

        if bbox and len(bbox) >= 4:
            x1 = int(bbox[0] * w)
            y1 = int(bbox[1] * h)
            x2 = int((bbox[0] + bbox[2]) * w)
            y2 = int((bbox[1] + bbox[3]) * h)

            # Color by prefix type
            colors = {
                "weapon": (0, 0, 255),        # Red
                "mask_face": (255, 165, 0),    # Orange
                "crowd_density": (255, 255, 0),# Cyan
                "abandoned_object": (0, 255, 255), # Yellow
                "violence": (128, 0, 128),     # Purple
            }
            color = colors.get(prefix, (0, 255, 0))

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            if label:
                font_scale = 0.6
                thickness = 2
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
                cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{detection_id}_{timestamp}.jpg"
        filepath = self.evidence_dir / filename

        cv2.imwrite(str(filepath), annotated)
        return str(filepath)
    
    def extract_video_clip(
        self,
        video_path: str,
        start_frame: int,
        end_frame: int,
        output_path: str
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
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
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

    def draw_detections_on_frame(
        self, frame: np.ndarray, detections: list
    ) -> np.ndarray:
        """
        Draw bounding boxes and labels on a frame.

        Args:
            detections: list of dicts with keys:
                det_type (str), confidence (float), bbox [x,y,w,h] normalized,
                class_name (str, optional)
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        for det in detections:
            bbox = det.get("bbox", [])
            if not bbox or len(bbox) < 4:
                continue

            x1 = int(bbox[0] * w)
            y1 = int(bbox[1] * h)
            x2 = int((bbox[0] + bbox[2]) * w)
            y2 = int((bbox[1] + bbox[3]) * h)

            det_type = det.get("det_type", "")
            color = self.DETECTION_COLORS.get(det_type, (0, 255, 0))
            confidence = det.get("confidence", 0)
            class_name = det.get("class_name", det_type.replace("_", " ").title())

            # Draw box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Draw label background + text
            label = f"{class_name} {confidence:.0%}"
            font_scale = 0.6
            thickness = 2
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                annotated, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness,
            )

            # Draw tracking line from bottom-center of bbox to bottom of frame
            center_x = (x1 + x2) // 2
            cv2.line(annotated, (center_x, y2), (center_x, h), color, 1, cv2.LINE_AA)

        return annotated

    def create_video_writer(self, output_path: str, fps: float, width: int, height: int):
        """Create a VideoWriter for the output annotated video."""
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        return cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    def reencode_to_h264(self, input_path: str) -> str:
        """
        Re-encode an mp4v video to H.264 so browsers can play it inline.
        Replaces the original file. Returns the final path.
        """
        import subprocess
        import shutil
        import os

        # Check if ffmpeg is available
        if not shutil.which("ffmpeg"):
            # ffmpeg not installed — return original (will trigger download instead of play)
            return input_path

        temp_path = input_path + ".h264.mp4"
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", input_path,
                    "-c:v", "libx264",
                    "-preset", "fast",
                    "-crf", "23",
                    "-movflags", "+faststart",  # enables streaming/seeking
                    "-an",  # no audio track
                    temp_path,
                ],
                check=True,
                capture_output=True,
                timeout=300,
            )
            # Replace original with h264 version
            os.replace(temp_path, input_path)
        except Exception:
            # If ffmpeg fails, keep the original file
            if os.path.exists(temp_path):
                os.remove(temp_path)

        return input_path


video_service = VideoService()

