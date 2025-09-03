"""Binary activity detection from frame differences."""

import numpy as np
from typing import Dict, Any

from ..base import BasicFeature


class BinaryActivity(BasicFeature):
    """Convert frame differences to binary activity timeline."""
    
    FEATURE_NAME = 'binary_activity'
    REQUIRED_ANALYSES = ['frame_diff']
    
    def __init__(self, threshold: float = 30.0, activity_threshold: float = 0.1):
        """
        Args:
            threshold: Pixel difference threshold for activity
            activity_threshold: Percentage of pixels that must change
        """
        super().__init__()
        self.threshold = threshold
        self.activity_threshold = activity_threshold
    
    def _compute_basic(self, analysis_data: Dict[str, Any]) -> np.ndarray:
        """Compute binary activity from frame differences.
        
        Returns:
            Binary array where 1 = active, 0 = inactive
        """
        # Get frame difference data
        frame_diff_analysis = analysis_data['frame_diff']
        pixel_diffs = frame_diff_analysis.data['pixel_diff']
        
        # Compute activity for each frame
        binary_activity = []
        for diff in pixel_diffs:
            # Calculate percentage of pixels above threshold
            active_pixels = np.sum(diff > self.threshold)
            total_pixels = diff.size
            activity_ratio = active_pixels / total_pixels
            
            # Mark as active if enough pixels changed
            is_active = 1 if activity_ratio > self.activity_threshold else 0
            binary_activity.append(is_active)
        
        return np.array(binary_activity, dtype=np.uint8)