"""Spatial occupancy grid from motion data."""

import numpy as np
from typing import Dict, Any, Tuple

from ..base import MiddleFeature


class SpatialOccupancyGrid(MiddleFeature):
    """Create spatial occupancy grid showing activity distribution."""
    
    FEATURE_NAME = 'spatial_occupancy_grid'
    REQUIRED_ANALYSES = ['frame_diff']  # or background_mog2
    
    def __init__(self, grid_size: Tuple[int, int] = (10, 10), 
                 threshold: float = 10.0):
        """
        Args:
            grid_size: Grid dimensions (rows, cols)
            threshold: Activity threshold for occupancy
        """
        super().__init__()
        self.grid_size = grid_size
        self.threshold = threshold
    
    def _compute_middle(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compute spatial occupancy grid.
        
        Returns:
            Dict with occupancy grid and statistics
        """
        if 'frame_diff' in analysis_data:
            activity_data = analysis_data['frame_diff'].data['pixel_diff']
        elif 'background_mog2' in analysis_data:
            activity_data = analysis_data['background_mog2'].data['foreground_mask']
        else:
            raise ValueError("Need frame_diff or background_mog2")
        
        if len(activity_data) == 0:
            return {
                'occupancy_grid': np.zeros(self.grid_size),
                'temporal_occupancy': [],
                'max_occupancy': 0
            }
        
        h, w = activity_data[0].shape
        grid_h, grid_w = self.grid_size
        cell_h = h // grid_h
        cell_w = w // grid_w
        
        # Track occupancy over time
        temporal_occupancy = []
        cumulative_occupancy = np.zeros(self.grid_size)
        
        for frame_activity in activity_data:
            frame_grid = np.zeros(self.grid_size)
            
            for i in range(grid_h):
                for j in range(grid_w):
                    # Extract cell
                    y_start = i * cell_h
                    y_end = min((i + 1) * cell_h, h)
                    x_start = j * cell_w
                    x_end = min((j + 1) * cell_w, w)
                    
                    cell = frame_activity[y_start:y_end, x_start:x_end]
                    
                    # Compute occupancy
                    if cell.size > 0:
                        occupancy = np.mean(cell > self.threshold)
                        frame_grid[i, j] = occupancy
            
            temporal_occupancy.append(frame_grid)
            cumulative_occupancy += frame_grid
        
        # Normalize cumulative occupancy
        if len(activity_data) > 0:
            cumulative_occupancy /= len(activity_data)
        
        return {
            'occupancy_grid': cumulative_occupancy,
            'temporal_occupancy': np.array(temporal_occupancy),
            'max_occupancy': float(np.max(cumulative_occupancy)),
            'occupancy_distribution': cumulative_occupancy.flatten()
        }
