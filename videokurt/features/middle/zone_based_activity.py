"""Zone-based activity analysis for predefined regions."""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional

from ..base import BaseFeature


class ZoneBasedActivity(BaseFeature):
    """Measure activity levels in predefined spatial zones."""
    
    FEATURE_NAME = 'zone_based_activity'
    REQUIRED_ANALYSES = ['frame_diff']  # or background_mog2
    
    def __init__(self, zones: Optional[List[Dict]] = None, 
                 grid_size: Tuple[int, int] = (3, 3)):
        """
        Args:
            zones: List of zone definitions with 'name' and 'bbox' (x, y, w, h)
            grid_size: If zones not provided, divide frame into grid (rows, cols)
        """
        super().__init__()
        self.zones = zones
        self.grid_size = grid_size
    
    def compute(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compute activity levels per zone.
        
        Returns:
            Dict with zone activity timelines and statistics
        """
        self.validate_inputs(analysis_data)
        
        # Get activity data
        if 'frame_diff' in analysis_data:
            activity_data = analysis_data['frame_diff'].data['pixel_diff']
        elif 'background_mog2' in analysis_data:
            activity_data = analysis_data['background_mog2'].data['foreground_mask']
        else:
            raise ValueError("Need frame_diff or background_mog2 analysis")
        
        if len(activity_data) == 0:
            return {
                'zone_activities': {},
                'most_active_zone': None,
                'activity_distribution': []
            }
        
        # Define zones if not provided
        h, w = activity_data[0].shape
        if self.zones is None:
            self.zones = self._create_grid_zones(h, w, self.grid_size)
        
        # Track activity per zone
        zone_activities = {zone['name']: [] for zone in self.zones}
        
        for frame_activity in activity_data:
            for zone in self.zones:
                x, y, zone_w, zone_h = zone['bbox']
                
                # Extract zone region
                x_end = min(x + zone_w, w)
                y_end = min(y + zone_h, h)
                zone_region = frame_activity[y:y_end, x:x_end]
                
                # Compute zone activity (mean intensity)
                if zone_region.size > 0:
                    activity_level = np.mean(zone_region)
                else:
                    activity_level = 0.0
                
                zone_activities[zone['name']].append(activity_level)
        
        # Convert to arrays
        for zone_name in zone_activities:
            zone_activities[zone_name] = np.array(zone_activities[zone_name])
        
        # Compute statistics
        zone_stats = {}
        for zone_name, activities in zone_activities.items():
            zone_stats[zone_name] = {
                'mean_activity': float(np.mean(activities)),
                'max_activity': float(np.max(activities)),
                'activity_ratio': float(np.sum(activities > 10) / len(activities))  # Frames with activity
            }
        
        # Find most active zone
        most_active = max(zone_stats.items(), 
                         key=lambda x: x[1]['mean_activity'])[0] if zone_stats else None
        
        return {
            'zone_activities': zone_activities,
            'zone_statistics': zone_stats,
            'most_active_zone': most_active,
            'activity_distribution': self._compute_distribution(zone_stats)
        }
    
    def _create_grid_zones(self, height: int, width: int, 
                          grid_size: Tuple[int, int]) -> List[Dict]:
        """Create grid-based zones."""
        # No need to validate here, already validated in compute()
        
        rows, cols = grid_size
        zone_h = height // rows
        zone_w = width // cols
        
        zones = []
        for i in range(rows):
            for j in range(cols):
                zones.append({
                    'name': f'zone_{i}_{j}',
                    'bbox': (j * zone_w, i * zone_h, zone_w, zone_h)
                })
        
        return zones
    
    def _compute_distribution(self, zone_stats: Dict) -> np.ndarray:
        """Compute normalized activity distribution across zones."""
        if not zone_stats:
            return np.array([])
        
        activities = [stats['mean_activity'] for stats in zone_stats.values()]
        total = sum(activities)
        
        if total > 0:
            return np.array([a / total for a in activities])
        else:
            return np.array([1.0 / len(activities)] * len(activities))