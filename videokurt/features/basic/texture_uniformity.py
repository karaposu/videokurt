"""Texture uniformity score from texture descriptors."""

import numpy as np
from typing import Dict, Any

from ..base import BaseFeature


class TextureUniformity(BaseFeature):
    """Compute texture uniformity score from texture analysis."""
    
    FEATURE_NAME = 'texture_uniformity'
    REQUIRED_ANALYSES = ['texture_descriptors']
    
    def __init__(self, window_size: int = 8):
        """
        Args:
            window_size: Window size for local uniformity computation
        """
        super().__init__()
        self.window_size = window_size
    
    def compute(self, analysis_data: Dict[str, Any]) -> np.ndarray:
        """Compute texture uniformity scores.
        
        Returns:
            Array of uniformity scores (0=textured, 1=uniform) per frame
        """
        self.validate_inputs(analysis_data)
        
        texture_analysis = analysis_data['texture_descriptors']
        texture_maps = texture_analysis.data['texture_features']
        
        uniformity_scores = []
        for texture_map in texture_maps:
            # Compute local variance
            h, w = texture_map.shape[:2]
            local_variances = []
            
            for i in range(0, h - self.window_size, self.window_size):
                for j in range(0, w - self.window_size, self.window_size):
                    window = texture_map[i:i+self.window_size, j:j+self.window_size]
                    variance = np.var(window)
                    local_variances.append(variance)
            
            if local_variances:
                # Average local variance (low variance = high uniformity)
                avg_variance = np.mean(local_variances)
                # Convert to uniformity score
                uniformity = 1.0 / (1.0 + avg_variance)
            else:
                uniformity = 1.0
            
            uniformity_scores.append(uniformity)
        
        return np.array(uniformity_scores, dtype=np.float32)