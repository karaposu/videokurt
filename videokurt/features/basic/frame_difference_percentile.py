"""Frame difference percentile computation."""

import numpy as np
from typing import Dict, Any

from ..base import BasicFeature


class FrameDifferencePercentile(BasicFeature):
    """Compute percentile of pixel differences per frame."""
    
    FEATURE_NAME = 'frame_difference_percentile'
    REQUIRED_ANALYSES = ['frame_diff']
    
    def __init__(self, percentile: float = 95.0):
        """
        Args:
            percentile: Percentile to compute (e.g., 95 for 95th percentile)
        """
        super().__init__()
        self.percentile = percentile
    
    def _compute_basic(self, analysis_data: Dict[str, Any]) -> np.ndarray:
        """Compute percentile of pixel differences.
        
        Returns:
            Array of percentile values per frame
        """
        frame_diff_analysis = analysis_data['frame_diff']
        pixel_diffs = frame_diff_analysis.data['pixel_diff']
        
        percentiles = []
        for diff in pixel_diffs:
            # Compute specified percentile
            value = np.percentile(diff, self.percentile)
            percentiles.append(value)
        
        return np.array(percentiles, dtype=np.float32)