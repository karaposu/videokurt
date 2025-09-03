"""Frame differencingAdvanced analysis."""

import time
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import cv2

from .base import BaseAnalysis
from ..models import RawAnalysis


class FrameDiffAdvanced(BaseAnalysis):
    """Advanced frame differencing with multiple techniques."""
    
    METHOD_NAME = 'frame_diff_advanced'
    
    def __init__(self, downsample: float = 1.0, 
                 window_size: int = 5, accumulate: bool = True):
        """
        Args:
            downsample: Resolution scale
            window_size: Size of temporal window for running average
            accumulate: Whether to compute accumulated differences
        """
        super().__init__(downsample=downsample)
        self.window_size = window_size
        self.accumulate = accumulate
    
    def analyze(self, frames: List[np.ndarray]) -> RawAnalysis:
        """Compute advanced frame differences including triple diff and running average."""
        start_time = time.time()
        frames = self.preprocess_frames(frames)
        
        # Convert to grayscale
        gray_frames = []
        for frame in frames:
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            gray_frames.append(gray)
        
        if len(gray_frames) < 3:
            raise ValueError("Need at least 3 frames for advanced differencing")
        
        h, w = gray_frames[0].shape
        
        # Initialize outputs
        triple_diffs = []
        running_avg_diffs = []
        accumulated_diff = np.zeros((h, w), dtype=np.float32)
        running_avg = np.float32(gray_frames[0])
        
        # Process frames
        for i in range(2, len(gray_frames)):
            # Triple frame differencing (reduces noise)
            diff1 = cv2.absdiff(gray_frames[i-2], gray_frames[i-1])
            diff2 = cv2.absdiff(gray_frames[i-1], gray_frames[i])
            triple_diff = cv2.bitwise_and(diff1, diff2)
            triple_diffs.append(triple_diff)
            
            # Running average background subtraction
            cv2.accumulateWeighted(gray_frames[i], running_avg, 0.02)  # Learning rate
            background = np.uint8(running_avg)
            diff_background = cv2.absdiff(gray_frames[i], background)
            running_avg_diffs.append(diff_background)
            
            # Accumulated differences (motion history)
            if self.accumulate:
                simple_diff = cv2.absdiff(gray_frames[i-1], gray_frames[i])
                accumulated_diff = accumulated_diff * 0.95  # Decay factor
                accumulated_diff += simple_diff.astype(np.float32) / 255.0
        
        # Normalize accumulated diff
        accumulated_norm = np.uint8(np.clip(accumulated_diff * 255, 0, 255))
        
        return RawAnalysis(
            method=self.METHOD_NAME,
            data={
                'triple_diff': np.array(triple_diffs) if triple_diffs else None,
                'running_avg_diff': np.array(running_avg_diffs) if running_avg_diffs else None,
                'accumulated_diff': accumulated_norm if self.accumulate else None
            },
            parameters={
                'downsample': self.downsample,
                'window_size': self.window_size,
                'accumulate': self.accumulate
            },
            processing_time=time.time() - start_time,
            output_shapes={
                'triple_diff': np.array(triple_diffs).shape if triple_diffs else None,
                'running_avg_diff': np.array(running_avg_diffs).shape if running_avg_diffs else None,
                'accumulated_diff': accumulated_norm.shape if self.accumulate else None
            },
            dtype_info={
                'triple_diff': 'uint8',
                'running_avg_diff': 'uint8',
                'accumulated_diff': 'uint8'
            }
        )