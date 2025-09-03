"""BackgroundMOG2 analysis."""

import time
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import cv2

from .base import BaseAnalysis
from ..models import RawAnalysis


class BackgroundMOG2(BaseAnalysis):
    """MOG2 background subtraction analysis."""
    
    METHOD_NAME = 'background_mog2'
    
    def __init__(self, downsample: float = 0.5,  # Often downsample for speed
                 history: int = 120, var_threshold: float = 16.0,
                 detect_shadows: bool = True):
        """
        Args:
            downsample: Resolution scale (default 0.5 for performance)
            history: Number of frames for background model
            var_threshold: Variance threshold for background model
            detect_shadows: Whether to detect shadows
        """
        super().__init__(downsample=downsample)
        self.history = history
        self.var_threshold = var_threshold
        self.detect_shadows = detect_shadows
    
    def analyze(self, frames: List[np.ndarray]) -> RawAnalysis:
        """Learn background model and detect foreground objects."""
        start_time = time.time()
        frames = self.preprocess_frames(frames)
        
        # Create MOG2 background subtractor
        mog2 = cv2.createBackgroundSubtractorMOG2(
            detectShadows=self.detect_shadows,
            varThreshold=self.var_threshold,
            history=self.history
        )
        
        foreground_masks = []
        
        for frame in frames:
            # Apply background subtraction
            mask = mog2.apply(frame)
            
            # Remove shadows if detected (shadows are 127, foreground is 255)
            if self.detect_shadows:
                mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)[1]
            
            # Apply morphological operations to reduce noise
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            foreground_masks.append(mask)
        
        foreground_array = np.array(foreground_masks)
        
        return RawAnalysis(
            method=self.METHOD_NAME,
            data={'foreground_mask': foreground_array},
            parameters={
                'downsample': self.downsample,
                'history': self.history,
                'var_threshold': self.var_threshold,
                'detect_shadows': self.detect_shadows
            },
            processing_time=time.time() - start_time,
            output_shapes={'foreground_mask': foreground_array.shape},
            dtype_info={'foreground_mask': str(foreground_array.dtype)}
        )