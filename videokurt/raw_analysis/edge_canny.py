"""EdgeCanny analysis."""

import time
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import cv2

from .base import BaseAnalysis
from ..models import RawAnalysis


class EdgeCanny(BaseAnalysis):
    """Canny edge detection analysis."""
    
    METHOD_NAME = 'edge_canny'
    
    def __init__(self, downsample: float = 1.0, 
                 low_threshold: int = 50, high_threshold: int = 150):
        """
        Args:
            downsample: Resolution scale
            low_threshold: Lower threshold for edge detection
            high_threshold: Upper threshold for edge detection
        """
        super().__init__(downsample=downsample)
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
    
    def analyze(self, frames: List[np.ndarray]) -> RawAnalysis:
        """Detect edges in each frame using Canny edge detector."""
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
        
        edge_maps = []
        gradient_magnitudes = []
        gradient_directions = []
        
        for gray in gray_frames:
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
            
            # Detect edges
            edges = cv2.Canny(blurred, self.low_threshold, self.high_threshold)
            edge_maps.append(edges)
            
            # Calculate gradients for additional info
            grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
            
            magnitude = np.sqrt(grad_x**2 + grad_y**2)
            direction = np.arctan2(grad_y, grad_x)
            
            gradient_magnitudes.append(magnitude.astype(np.float32))
            gradient_directions.append(direction.astype(np.float32))
        
        return RawAnalysis(
            method=self.METHOD_NAME,
            data={
                'edge_map': np.array(edge_maps),
                'gradient_magnitude': np.array(gradient_magnitudes),
                'gradient_direction': np.array(gradient_directions)
            },
            parameters={
                'downsample': self.downsample,
                'low_threshold': self.low_threshold,
                'high_threshold': self.high_threshold
            },
            processing_time=time.time() - start_time,
            output_shapes={
                'edge_map': np.array(edge_maps).shape,
                'gradient_magnitude': np.array(gradient_magnitudes).shape,
                'gradient_direction': np.array(gradient_directions).shape
            },
            dtype_info={
                'edge_map': 'uint8',
                'gradient_magnitude': 'float32',
                'gradient_direction': 'float32'
            }
        )


# =============================================================================
# LEVEL 2: Intermediate Analyses
# =============================================================================