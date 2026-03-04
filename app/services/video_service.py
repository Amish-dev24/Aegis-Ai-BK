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
        prefix: str = "snapshot"
    ) -> str:
        """
        Save a frame as a snapshot image.
        
        Returns:
            Path to saved image
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{detection_id}_{timestamp}.jpg"
        filepath = self.evidence_dir / filename
        
        cv2.imwrite(str(filepath), frame)
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

