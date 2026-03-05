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
    
    def __init__(self):
        self.upload_dir = Path(settings.UPLOAD_DIR)
        self.evidence_dir = Path(settings.EVIDENCE_DIR)
    
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


video_service = VideoService()

