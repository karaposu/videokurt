"""Motion direction histogram from optical flow."""

import numpy as np
from typing import Dict, Any

from ..base import BasicFeature


class MotionDirectionHistogram(BasicFeature):
    """Compute histogram of motion directions from optical flow."""
    
    FEATURE_NAME = 'motion_direction_histogram'
    REQUIRED_ANALYSES = ['optical_flow_dense']
    
    def __init__(self, num_bins: int = 8, magnitude_threshold: float = 1.0):
        """
        Args:
            num_bins: Number of direction bins (8 or 16 typical)
            magnitude_threshold: Minimum magnitude to include in histogram
        """
        super().__init__()
        self.num_bins = num_bins
        self.magnitude_threshold = magnitude_threshold
    
    def _compute_basic(self, analysis_data: Dict[str, Any]) -> np.ndarray:
        """Compute histogram of motion directions.
        
        Returns:
            Array of histograms (num_frames, num_bins)
        """
        flow_analysis = analysis_data['optical_flow_dense']
        flow_field = flow_analysis.data['flow_field']
        
        histograms = []
        for flow in flow_field:
            # Calculate angles and magnitudes
            flow_x = flow[..., 0]
            flow_y = flow[..., 1]
            
            magnitudes = np.sqrt(flow_x**2 + flow_y**2)
            angles = np.arctan2(flow_y, flow_x)
            
            # Filter by magnitude threshold
            valid_mask = magnitudes > self.magnitude_threshold
            valid_angles = angles[valid_mask]
            
            if len(valid_angles) > 0:
                # Convert angles to 0-2π range
                valid_angles = np.mod(valid_angles, 2 * np.pi)
                
                # Create histogram
                hist, _ = np.histogram(valid_angles, 
                                      bins=self.num_bins, 
                                      range=(0, 2 * np.pi))
                # Normalize
                hist = hist.astype(np.float32) / np.sum(hist) if np.sum(hist) > 0 else hist
            else:
                hist = np.zeros(self.num_bins, dtype=np.float32)
            
            histograms.append(hist)
        
        return np.array(histograms)