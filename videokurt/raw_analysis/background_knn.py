"""BackgroundKNN analysis."""

import time
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import cv2

from .base import BaseAnalysis
from ..models import RawAnalysis


class BackgroundKNN(BaseAnalysis):
    """KNN background subtraction analysis."""
    
    METHOD_NAME = 'background_knn'
    
    def __init__(self, downsample: float = 0.5,
                 history: int = 200, dist2_threshold: float = 400.0,
                 detect_shadows: bool = False):
        """
        Args:
            downsample: Resolution scale
            history: Number of frames for background model
            dist2_threshold: Distance threshold for KNN
            detect_shadows: Whether to detect shadows
        """
        super().__init__(downsample=downsample)
        self.history = history
        self.dist2_threshold = dist2_threshold
        self.detect_shadows = detect_shadows
    
    def analyze(self, frames: List[np.ndarray]) -> RawAnalysis:
        """K-nearest neighbors background model."""
        start_time = time.time()
        frames = self.preprocess_frames(frames)
        
        # Create KNN background subtractor
        knn = cv2.createBackgroundSubtractorKNN(
            detectShadows=self.detect_shadows,
            dist2Threshold=self.dist2_threshold,
            history=self.history
        )
        
        foreground_masks = []
        
        for frame in frames:
            # Apply background subtraction
            mask = knn.apply(frame)
            
            # Remove shadows if detected
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
                'dist2_threshold': self.dist2_threshold,
                'detect_shadows': self.detect_shadows
            },
            processing_time=time.time() - start_time,
            output_shapes={'foreground_mask': foreground_array.shape},
            dtype_info={'foreground_mask': str(foreground_array.dtype)}
        )