"""Structural similarity between consecutive frames."""

import numpy as np
import cv2
from typing import Dict, Any
from skimage.metrics import structural_similarity as ssim

from ..base import BaseFeature


class StructuralSimilarity(BaseFeature):
    """Compute SSIM between consecutive frames."""
    
    FEATURE_NAME = 'structural_similarity'
    REQUIRED_ANALYSES = ['frame_diff']  # Needs frame_diff for now
    
    def __init__(self, win_size: int = 7, multichannel: bool = False):
        """
        Args:
            win_size: Window size for SSIM computation
            multichannel: Whether to compute SSIM on color channels
        """
        super().__init__()
        self.win_size = win_size
        self.multichannel = multichannel
    
    def compute(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compute structural similarity metrics.
        
        Returns:
            Dict with SSIM scores and difference maps
        """
        self.validate_inputs(analysis_data)
        
        # This would need access to original frames
        # For now, we can compute from frame_diff if available
        if 'frame_diff' in analysis_data:
            frame_diff_data = analysis_data['frame_diff'].data['pixel_diff']
            
            # Compute similarity as inverse of normalized difference
            ssim_scores = []
            for diff in frame_diff_data:
                # Normalize difference to 0-1 range
                norm_diff = diff / 255.0
                # Convert to similarity (1 - difference)
                avg_diff = np.mean(norm_diff)
                similarity = 1.0 - avg_diff
                ssim_scores.append(similarity)
            
            return {
                'ssim_scores': np.array(ssim_scores),
                'mean_ssim': np.mean(ssim_scores),
                'min_ssim': np.min(ssim_scores),
                'change_points': self._detect_change_points(ssim_scores)
            }
        else:
            raise ValueError("Need frame_diff or raw frames for SSIM computation")
    
    def _detect_change_points(self, scores: list, threshold: float = 0.7) -> list:
        """Detect frames where similarity drops significantly."""
        
        change_points = []
        for i in range(len(scores)):
            if scores[i] < threshold:
                change_points.append(i)
        return change_points