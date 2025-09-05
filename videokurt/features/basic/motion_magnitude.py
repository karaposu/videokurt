"""Motion magnitude computation from optical flow."""

import numpy as np
from typing import Dict, Any

from ..base import BaseFeature


class MotionMagnitude(BaseFeature):
    """Compute total motion magnitude per frame from optical flow."""
    
    FEATURE_NAME = 'motion_magnitude'
    REQUIRED_ANALYSES = ['optical_flow_dense']
    
    def __init__(self, normalize: bool = False):
        """
        Args:
            normalize: Whether to normalize by frame size
        """
        super().__init__()
        self.normalize = normalize
    
    def compute(self, analysis_data: Dict[str, Any]) -> np.ndarray:
        """Compute motion magnitude from optical flow.
        
        Returns:
            Array of scalar motion magnitudes per frame
        """
        self.validate_inputs(analysis_data)
        
        # Get optical flow data
        flow_analysis = analysis_data['optical_flow_dense']
        flow_field = flow_analysis.data['flow_field']
        
        # Compute magnitude for each frame
        magnitudes = []
        for flow in flow_field:
            # Calculate magnitude at each pixel
            flow_x = flow[..., 0]
            flow_y = flow[..., 1]
            pixel_magnitudes = np.sqrt(flow_x**2 + flow_y**2)
            
            # Average magnitude across frame
            avg_magnitude = np.mean(pixel_magnitudes)
            
            if self.normalize:
                # Normalize by frame diagonal
                h, w = flow.shape[:2]
                diagonal = np.sqrt(h**2 + w**2)
                avg_magnitude = avg_magnitude / diagonal
            
            magnitudes.append(avg_magnitude)
        
        return np.array(magnitudes, dtype=np.float32)