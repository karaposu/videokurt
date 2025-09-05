"""Stability score computation from frame differences."""

import numpy as np
from typing import Dict, Any

from ..base import BaseFeature


class StabilityScore(BaseFeature):
    """Measure content stability over time windows."""
    
    FEATURE_NAME = 'stability_score'
    REQUIRED_ANALYSES = ['frame_diff']
    
    def __init__(self, window_size: int = 10, threshold: float = 5.0):
        """
        Args:
            window_size: Number of frames in temporal window
            threshold: Threshold for considering content stable
        """
        super().__init__()
        self.window_size = window_size
        self.threshold = threshold
    
    def compute(self, analysis_data: Dict[str, Any]) -> np.ndarray:
        """Compute stability scores over temporal windows.
        
        Returns:
            Array of stability scores (0=changing, 1=stable)
        """
        self.validate_inputs(analysis_data)
        
        frame_diff_analysis = analysis_data['frame_diff']
        pixel_diffs = frame_diff_analysis.data['pixel_diff']
        
        stability_scores = []
        for i in range(len(pixel_diffs)):
            # Get window of frames
            start = max(0, i - self.window_size // 2)
            end = min(len(pixel_diffs), i + self.window_size // 2)
            window = pixel_diffs[start:end]
            
            # Compute variance in window
            if len(window) > 0:
                avg_changes = [np.mean(diff) for diff in window]
                variance = np.var(avg_changes)
                
                # Convert to stability score (low variance = high stability)
                if variance < self.threshold:
                    stability = 1.0
                else:
                    stability = self.threshold / (variance + self.threshold)
            else:
                stability = 1.0
            
            stability_scores.append(stability)
        
        return np.array(stability_scores, dtype=np.float32)