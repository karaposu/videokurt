"""Foreground ratio computation from background subtraction."""

import numpy as np
from typing import Dict, Any

from ..base import BasicFeature


class ForegroundRatio(BasicFeature):
    """Compute percentage of foreground pixels per frame."""
    
    FEATURE_NAME = 'foreground_ratio'
    REQUIRED_ANALYSES = ['background_mog2']  # or background_knn
    
    def __init__(self, min_value: int = 128):
        """
        Args:
            min_value: Minimum pixel value to consider as foreground
        """
        super().__init__()
        self.min_value = min_value
    
    def _compute_basic(self, analysis_data: Dict[str, Any]) -> np.ndarray:
        """Compute foreground ratio from background subtraction.
        
        Returns:
            Array of foreground ratios (0-1) per frame
        """
        # Try MOG2 first, then KNN
        if 'background_mog2' in analysis_data:
            bg_analysis = analysis_data['background_mog2']
        elif 'background_knn' in analysis_data:
            bg_analysis = analysis_data['background_knn']
        else:
            raise ValueError("Need background_mog2 or background_knn analysis")
        
        foreground_masks = bg_analysis.data['foreground_mask']
        
        ratios = []
        for mask in foreground_masks:
            # Count foreground pixels
            foreground_pixels = np.sum(mask >= self.min_value)
            total_pixels = mask.size
            ratio = foreground_pixels / total_pixels
            ratios.append(ratio)
        
        return np.array(ratios, dtype=np.float32)