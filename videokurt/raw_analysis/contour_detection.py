"""ContourDetection analysis."""

import time
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import cv2

from .base import BaseAnalysis
from ..models import RawAnalysis


class ContourDetection(BaseAnalysis):
    """Contour and shape detection analysis."""
    
    METHOD_NAME = 'contour_detection'
    
    def __init__(self, downsample: float = 0.5,  # Often downsample for contours
                 threshold: int = 127, max_contours: int = 100):
        """
        Args:
            downsample: Resolution scale (default 0.5 for performance)
            threshold: Binary threshold for contour detection
            max_contours: Maximum number of contours to keep per frame
        """
        super().__init__(downsample=downsample)
        self.threshold = threshold
        self.max_contours = max_contours
    
    def analyze(self, frames: List[np.ndarray]) -> RawAnalysis:
        """Find and track shape boundaries in each frame."""
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
        
        all_contours = []
        all_hierarchies = []
        
        # Process each frame pair
        for i in range(1, len(gray_frames)):
            # Calculate frame difference
            diff = cv2.absdiff(gray_frames[i-1], gray_frames[i])
            
            # Apply threshold to get binary mask
            _, thresh = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)
            
            # Apply morphological operations to clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)  # Close gaps
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)   # Remove noise
            
            # Find contours
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter and keep only significant contours
            significant_contours = []
            significant_hierarchy = []
            
            for j, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area > 100:  # Minimum area threshold
                    significant_contours.append(contour)
                    if hierarchy is not None:
                        significant_hierarchy.append(hierarchy[0][j])
            
            # Sort by area and keep only top N
            if len(significant_contours) > self.max_contours:
                areas = [cv2.contourArea(c) for c in significant_contours]
                indices = np.argsort(areas)[-self.max_contours:]
                significant_contours = [significant_contours[i] for i in indices]
                if significant_hierarchy:
                    significant_hierarchy = [significant_hierarchy[i] for i in indices]
            
            all_contours.append(significant_contours)
            all_hierarchies.append(np.array(significant_hierarchy) if significant_hierarchy else None)
        
        return RawAnalysis(
            method=self.METHOD_NAME,
            data={
                'contours': all_contours,
                'hierarchy': all_hierarchies
            },
            parameters={
                'downsample': self.downsample,
                'threshold': self.threshold,
                'max_contours': self.max_contours
            },
            processing_time=time.time() - start_time,
            output_shapes={
                'contours': f"List[{len(all_contours)}][varies]",
                'hierarchy': f"List[{len(all_hierarchies)}]"
            },
            dtype_info={
                'contours': 'List[np.ndarray]',
                'hierarchy': 'List[np.ndarray]'
            }
        )


# =============================================================================
# LEVEL 3: Advanced Analyses
# =============================================================================