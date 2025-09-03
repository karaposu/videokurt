"""Frame extraction from video files using OpenCV."""

import cv2
import numpy as np
from typing import List, Generator, Optional, Tuple
from pathlib import Path
from .video_info import VideoInfo


class FrameExtractor:
    """Extract frames from video files efficiently.
    
    Supports multiple extraction strategies:
    - All frames
    - Every Nth frame
    - Specific frame indices
    - Time-based sampling
    """
    
    def __init__(self, video_path: str):
        """Initialize frame extractor.
        
        Args:
            video_path: Path to video file
            
        Raises:
            ValueError: If video file doesn't exist or can't be opened
        """
        self.video_path = Path(video_path)
        
        if not self.video_path.exists():
            raise ValueError(f"Video file not found: {video_path}")
        
        # Test if we can open the video
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        # Extract video info
        self.info = self._extract_info()
        
        # Close and reopen for actual extraction
        self.cap.release()
    
    @property
    def width(self) -> int:
        """Get video width."""
        return self.info.width
    
    @property
    def height(self) -> int:
        """Get video height."""
        return self.info.height
    
    @property
    def fps(self) -> float:
        """Get video FPS."""
        return self.info.fps
    
    @property
    def frame_count(self) -> int:
        """Get video frame count."""
        return self.info.frame_count
    
    def close(self):
        """Close and release video resources."""
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
            self.cap = None
        
    def _extract_info(self) -> VideoInfo:
        """Extract video metadata."""
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Verify actual frame count by testing last frame
        # Some videos report incorrect frame count
        if frame_count > 0:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
            ret, _ = self.cap.read()
            if not ret:
                # Last frame not readable, adjust count
                # Try a few frames back to find actual last frame
                for i in range(min(10, frame_count)):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 2 - i)
                    ret, _ = self.cap.read()
                    if ret:
                        frame_count = frame_count - 1 - i
                        break
            
            # Reset to beginning
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Calculate duration
        duration = frame_count / fps if fps > 0 else 0
        
        # Try to get codec
        fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        codec = None
        if fourcc > 0:
            codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        
        return VideoInfo(
            filepath=str(self.video_path),
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            duration_seconds=duration,
            codec=codec
        )
    
    def extract_all_frames(self) -> Generator[np.ndarray, None, None]:
        """Extract all frames from video.
        
        Yields frames one at a time to avoid memory issues.
        
        Yields:
            np.ndarray: Frame as BGR image
        """
        cap = cv2.VideoCapture(str(self.video_path))
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                yield frame
        finally:
            cap.release()
    
    def extract_every_nth_frame(self, n: int) -> Generator[Tuple[int, np.ndarray], None, None]:
        """Extract every Nth frame.
        
        Args:
            n: Frame interval (e.g., 5 means every 5th frame)
            
        Yields:
            Tuple of (frame_index, frame)
        """
        cap = cv2.VideoCapture(str(self.video_path))
        
        try:
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % n == 0:
                    yield (frame_idx, frame)
                
                frame_idx += 1
        finally:
            cap.release()
    
    def extract_frame_at_index(self, index: int) -> Optional[np.ndarray]:
        """Extract a specific frame by index.
        
        Args:
            index: Frame index (0-based)
            
        Returns:
            Frame or None if index out of bounds
        """
        if index < 0 or index >= self.info.frame_count:
            return None
        
        cap = cv2.VideoCapture(str(self.video_path))
        
        try:
            # Seek to frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ret, frame = cap.read()
            
            return frame if ret else None
        finally:
            cap.release()
    
    def extract_frames_at_indices(self, indices: List[int]) -> Generator[Tuple[int, np.ndarray], None, None]:
        """Extract frames at specific indices.
        
        Args:
            indices: List of frame indices to extract
            
        Yields:
            Tuple of (frame_index, frame)
        """
        cap = cv2.VideoCapture(str(self.video_path))
        
        try:
            # Remove duplicates and sort for efficient seeking
            unique_indices = sorted(set(indices))
            
            for idx in unique_indices:
                if idx < 0 or idx >= self.info.frame_count:
                    continue
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                
                if ret:
                    yield (idx, frame)
        finally:
            cap.release()
    
    def extract_frames_by_time(
        self, 
        start_seconds: float = 0, 
        end_seconds: Optional[float] = None,
        interval_seconds: float = 1.0
    ) -> Generator[Tuple[float, np.ndarray], None, None]:
        """Extract frames at time intervals.
        
        Args:
            start_seconds: Start time in seconds
            end_seconds: End time in seconds (None for entire video)
            interval_seconds: Time between extracted frames
            
        Yields:
            Tuple of (timestamp_seconds, frame)
        """
        cap = cv2.VideoCapture(str(self.video_path))
        
        # Clamp start time to valid range
        start_seconds = max(0.0, start_seconds)
        
        if end_seconds is None:
            end_seconds = self.info.duration_seconds
        
        try:
            current_time = start_seconds
            
            while current_time <= end_seconds + 0.001:  # Small epsilon for floating point
                # Convert time to frame number
                frame_idx = int(current_time * self.info.fps)
                
                # Handle edge case: last frame
                if frame_idx >= self.info.frame_count:
                    frame_idx = self.info.frame_count - 1
                    # Only extract if this is exactly at the end time
                    if abs(current_time - end_seconds) < 0.001:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                        ret, frame = cap.read()
                        if ret:
                            yield (current_time, frame)
                    break
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    yield (current_time, frame)
                
                current_time += interval_seconds
        finally:
            cap.release()
    
    def extract_frames_adaptive(
        self,
        min_interval: int = 1,
        max_interval: int = 30,
        activity_threshold: float = 0.1
    ) -> Generator[Tuple[int, np.ndarray, float], None, None]:
        """Extract frames adaptively based on activity.
        
        Extracts more frames during high activity, fewer during low activity.
        This is a simple version - can be enhanced with actual activity detection.
        
        Args:
            min_interval: Minimum frames between extractions
            max_interval: Maximum frames between extractions
            activity_threshold: Threshold for detecting activity
            
        Yields:
            Tuple of (frame_index, frame, activity_score)
        """
        cap = cv2.VideoCapture(str(self.video_path))
        
        try:
            prev_frame = None
            frame_idx = 0
            last_extracted_idx = -max_interval  # Force first frame extraction
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Calculate activity (simple frame difference)
                activity = 0.0
                if prev_frame is not None:
                    diff = cv2.absdiff(frame, prev_frame)
                    activity = np.mean(diff) / 255.0  # Normalize to 0-1
                
                # Determine if we should extract this frame
                frames_since_last = frame_idx - last_extracted_idx
                
                should_extract = (
                    frames_since_last >= max_interval or  # Max interval reached
                    (activity > activity_threshold and frames_since_last >= min_interval)  # High activity
                )
                
                if should_extract:
                    yield (frame_idx, frame, activity)
                    last_extracted_idx = frame_idx
                
                prev_frame = frame
                frame_idx += 1
        finally:
            cap.release()
    
    def get_frame_count(self) -> int:
        """Get total number of frames in video."""
        return self.info.frame_count
    
    def get_fps(self) -> float:
        """Get video frame rate."""
        return self.info.fps
    
    def get_duration(self) -> float:
        """Get video duration in seconds."""
        return self.info.duration_seconds