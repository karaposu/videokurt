"""Frame differencing analysis."""

import time
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import cv2

from .base import BaseAnalysis
from ..models import RawAnalysis


class FrameDiff(BaseAnalysis):
    """Simple frame differencing analysis."""
    
    METHOD_NAME = 'frame_diff'
    
    def __init__(self, downsample: float = 1.0, threshold: float = 0.1):
        """
        Args:
            downsample: Resolution scale (0.5 = half resolution)
            threshold: Threshold for detecting changes
        """
        super().__init__(downsample=downsample)
        self.threshold = threshold
    
    def analyze(self, frames: List[np.ndarray]) -> RawAnalysis:
        """Compute pixel-wise differences between consecutive frames."""
        start_time = time.time()
        frames = self.preprocess_frames(frames)
        
        pixel_diffs = []
        for i in range(len(frames) - 1):
            # Convert to grayscale if needed
            if len(frames[i].shape) == 3:
                gray1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(frames[i+1], cv2.COLOR_BGR2GRAY)
            else:
                gray1 = frames[i]
                gray2 = frames[i+1]
            
            diff = cv2.absdiff(gray1, gray2)
            pixel_diffs.append(diff)
        
        pixel_diff_array = np.array(pixel_diffs)
        
        return RawAnalysis(
            method=self.METHOD_NAME,
            data={'pixel_diff': pixel_diff_array},
            parameters={
                'threshold': self.threshold,
                'downsample': self.downsample
            },
            processing_time=time.time() - start_time,
            output_shapes={'pixel_diff': pixel_diff_array.shape},
            dtype_info={'pixel_diff': str(pixel_diff_array.dtype)}
        )