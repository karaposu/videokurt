"""Dominant flow vector extraction from optical flow."""

import numpy as np
from typing import Dict, Any

from ..base import BasicFeature


class DominantFlowVector(BasicFeature):
    """Extract single dominant motion vector per frame."""
    
    FEATURE_NAME = 'dominant_flow_vector'
    REQUIRED_ANALYSES = ['optical_flow_dense']
    
    def __init__(self, use_median: bool = False):
        """
        Args:
            use_median: Use median instead of mean for robustness
        """
        super().__init__()
        self.use_median = use_median
    
    def _compute_basic(self, analysis_data: Dict[str, Any]) -> np.ndarray:
        """Extract dominant flow vector from optical flow field.
        
        Returns:
            Array of dominant vectors (num_frames, 2) for [x, y]
        """
        flow_analysis = analysis_data['optical_flow_dense']
        flow_field = flow_analysis.data['flow_field']
        
        dominant_vectors = []
        for flow in flow_field:
            flow_x = flow[..., 0]
            flow_y = flow[..., 1]
            
            if self.use_median:
                # Use median for robustness to outliers
                dominant_x = np.median(flow_x)
                dominant_y = np.median(flow_y)
            else:
                # Use mean for average motion
                dominant_x = np.mean(flow_x)
                dominant_y = np.mean(flow_y)
            
            dominant_vectors.append([dominant_x, dominant_y])
        
        return np.array(dominant_vectors, dtype=np.float32)