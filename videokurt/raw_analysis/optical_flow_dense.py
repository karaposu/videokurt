"""OpticalFlowDense analysis."""

import time
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import cv2

from .base import BaseAnalysis
from ..models import RawAnalysis


class OpticalFlowDense(BaseAnalysis):
    """Farneback dense optical flow analysis."""
    
    METHOD_NAME = 'optical_flow_dense'
    
    def __init__(self, downsample: float = 0.25,  # Heavy computation, downsample!
                 pyr_scale: float = 0.5, levels: int = 3,
                 winsize: int = 15, iterations: int = 3):
        """
        Args:
            downsample: Resolution scale (default 0.25 for performance)
            pyr_scale: Pyramid scale factor
            levels: Number of pyramid levels
            winsize: Window size for averaging
            iterations: Number of iterations at each pyramid level
        """
        super().__init__(downsample=downsample)
        self.pyr_scale = pyr_scale
        self.levels = levels
        self.winsize = winsize
        self.iterations = iterations
    
    def analyze(self, frames: List[np.ndarray]) -> RawAnalysis:
        """Calculate motion vectors for every pixel between frames."""
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
        
        # Compute optical flow
        flow_fields = []
        for i in range(len(gray_frames) - 1):
            flow = cv2.calcOpticalFlowFarneback(
                gray_frames[i], gray_frames[i+1], None,
                self.pyr_scale, self.levels, self.winsize,
                self.iterations, 5, 1.2, 0
            )
            flow_fields.append(flow)
        
        flow_array = np.array(flow_fields)
        
        return RawAnalysis(
            method=self.METHOD_NAME,
            data={'flow_field': flow_array},
            parameters={
                'downsample': self.downsample,
                'pyr_scale': self.pyr_scale,
                'levels': self.levels,
                'winsize': self.winsize,
                'iterations': self.iterations
            },
            processing_time=time.time() - start_time,
            output_shapes={'flow_field': flow_array.shape},
            dtype_info={'flow_field': str(flow_array.dtype)}
        )