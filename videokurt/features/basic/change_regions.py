"""Change region detection from frame differences."""

import numpy as np
import cv2
from typing import Dict, Any, List, Tuple

from ..base import BaseFeature


class ChangeRegions(BaseFeature):
    """Detect bounding boxes of changed regions."""
    
    FEATURE_NAME = 'change_regions'
    REQUIRED_ANALYSES = ['frame_diff']
    
    def __init__(self, threshold: float = 30.0, min_area: int = 100):
        """
        Args:
            threshold: Pixel difference threshold
            min_area: Minimum area for a change region
        """
        super().__init__()
        self.threshold = threshold
        self.min_area = min_area
    
    def compute(self, analysis_data: Dict[str, Any]) -> List[List[Tuple]]:
        """Compute bounding boxes of changed regions.
        
        Returns:
            List of bounding boxes per frame [(x, y, w, h), ...]
        """
        self.validate_inputs(analysis_data)
        
        frame_diff_analysis = analysis_data['frame_diff']
        pixel_diffs = frame_diff_analysis.data['pixel_diff']
        
        all_regions = []
        for diff in pixel_diffs:
            # Threshold the difference image
            binary = (diff > self.threshold).astype(np.uint8) * 255
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, 
                                          cv2.CHAIN_APPROX_SIMPLE)
            
            # Extract bounding boxes
            regions = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area >= self.min_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    regions.append((x, y, w, h))
            
            all_regions.append(regions)
        
        return all_regions