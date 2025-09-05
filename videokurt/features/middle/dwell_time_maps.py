"""Dwell time maps showing how long regions remain active."""

import numpy as np
from typing import Dict, Any, List

from ..base import BaseFeature


class DwellTimeMaps(BaseFeature):
    """Create maps showing duration of activity at each location."""
    
    FEATURE_NAME = 'dwell_time_maps'
    REQUIRED_ANALYSES = ['frame_diff']  # or motion_heatmap
    
    def __init__(self, activity_threshold: float = 10.0, decay_factor: float = 0.95):
        """
        Args:
            activity_threshold: Threshold for considering pixel active
            decay_factor: Factor for temporal decay (1.0 = no decay)
        """
        super().__init__()
        self.activity_threshold = activity_threshold
        self.decay_factor = decay_factor
    
    def compute(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compute dwell time maps from activity data.
        
        Returns:
            Dict with dwell maps and statistics
        """
        self.validate_inputs(analysis_data)
        
        # Use motion_heatmap if available, otherwise frame_diff
        if 'motion_heatmap' in analysis_data:
            heatmap_data = analysis_data['motion_heatmap'].data
            cumulative_map = heatmap_data.get('cumulative', None)
            if cumulative_map is not None:
                # Already have a dwell-like map
                return {
                    'dwell_map': cumulative_map,
                    'max_dwell': np.max(cumulative_map),
                    'mean_dwell': np.mean(cumulative_map),
                    'active_pixels': np.sum(cumulative_map > 0)
                }
        
        # Compute from frame differences
        frame_diff_data = analysis_data['frame_diff'].data['pixel_diff']
        
        if len(frame_diff_data) == 0:
            return {
                'dwell_map': np.zeros((1, 1)),
                'max_dwell': 0,
                'mean_dwell': 0,
                'active_pixels': 0
            }
        
        # Initialize dwell map
        h, w = frame_diff_data[0].shape
        dwell_map = np.zeros((h, w), dtype=np.float32)
        current_activity = np.zeros((h, w), dtype=np.float32)
        
        # Track dwell times
        for diff in frame_diff_data:
            # Update activity
            active_mask = diff > self.activity_threshold
            
            # Increment dwell time for active pixels
            current_activity[active_mask] += 1
            
            # Reset dwell time for inactive pixels
            current_activity[~active_mask] *= self.decay_factor
            
            # Accumulate to dwell map
            dwell_map = np.maximum(dwell_map, current_activity)
        
        # Compute zones with longest dwell times
        dwell_zones = self._extract_dwell_zones(dwell_map)
        
        return {
            'dwell_map': dwell_map,
            'max_dwell': np.max(dwell_map),
            'mean_dwell': np.mean(dwell_map),
            'active_pixels': np.sum(dwell_map > 0),
            'dwell_zones': dwell_zones
        }
    
    def _extract_dwell_zones(self, dwell_map: np.ndarray, top_k: int = 5) -> List[Dict]:
        """Extract top zones with highest dwell times."""
        
        from scipy import ndimage
        
        # Threshold to get significant dwell areas
        threshold = np.percentile(dwell_map, 90)
        binary = dwell_map > threshold
        
        # Label connected components
        labeled, num_features = ndimage.label(binary)
        
        zones = []
        for i in range(1, min(num_features + 1, top_k + 1)):
            mask = labeled == i
            if np.any(mask):
                y_coords, x_coords = np.where(mask)
                zones.append({
                    'center': (int(np.mean(x_coords)), int(np.mean(y_coords))),
                    'area': int(np.sum(mask)),
                    'max_dwell': float(np.max(dwell_map[mask])),
                    'avg_dwell': float(np.mean(dwell_map[mask]))
                })
        
        # Sort by max dwell time
        zones.sort(key=lambda x: x['max_dwell'], reverse=True)
        
        return zones[:top_k]