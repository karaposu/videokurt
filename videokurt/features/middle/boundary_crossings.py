"""Boundary crossing detection for motion tracking."""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional

from ..base import BaseFeature


class BoundaryCrossings(BaseFeature):
    """Detect when motion crosses predefined boundaries."""
    
    FEATURE_NAME = 'boundary_crossings'
    REQUIRED_ANALYSES = ['optical_flow_dense']  # or background_mog2
    
    def __init__(self, boundaries: Optional[List[Dict]] = None):
        """
        Args:
            boundaries: List of boundary definitions with 'type' and parameters
                       e.g., {'type': 'horizontal', 'y': 100}
                       or {'type': 'vertical', 'x': 200}
                       or {'type': 'line', 'p1': (x1, y1), 'p2': (x2, y2)}
        """
        super().__init__()
        self.boundaries = boundaries
    
    def compute(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect boundary crossing events.
        
        Returns:
            Dict with crossing events and statistics
        """
        self.validate_inputs(analysis_data)
        
        # Get motion data
        if 'optical_flow_dense' in analysis_data:
            return self._detect_flow_crossings(analysis_data['optical_flow_dense'])
        elif 'background_mog2' in analysis_data:
            return self._detect_blob_crossings(analysis_data['background_mog2'])
        else:
            raise ValueError("Need optical_flow_dense or background_mog2")
    
    def _detect_flow_crossings(self, flow_analysis) -> Dict[str, Any]:
        """Detect crossings from optical flow."""
        
        flow_field = flow_analysis.data['flow_field']
        
        if len(flow_field) == 0:
            return {
                'crossings': [],
                'num_crossings': 0,
                'crossing_rate': 0
            }
        
        h, w = flow_field[0].shape[:2]
        
        # Create default boundaries if not provided
        if self.boundaries is None:
            self.boundaries = self._create_default_boundaries(h, w)
        
        crossings = []
        
        for frame_idx, flow in enumerate(flow_field):
            # Create position grids
            y_grid, x_grid = np.mgrid[0:h, 0:w]
            
            # Calculate end positions after flow
            end_x = x_grid + flow[..., 0]
            end_y = y_grid + flow[..., 1]
            
            # Check each boundary
            for boundary in self.boundaries:
                crossing_mask = self._check_boundary_crossing(
                    x_grid, y_grid, end_x, end_y, boundary
                )
                
                num_crossings = np.sum(crossing_mask)
                if num_crossings > 0:
                    crossings.append({
                        'frame': frame_idx,
                        'boundary': boundary.get('name', str(boundary)),
                        'num_pixels': int(num_crossings),
                        'direction': self._get_crossing_direction(
                            x_grid[crossing_mask], y_grid[crossing_mask],
                            end_x[crossing_mask], end_y[crossing_mask], boundary
                        )
                    })
        
        # Compute statistics
        total_crossings = len(crossings)
        crossing_rate = total_crossings / len(flow_field) if len(flow_field) > 0 else 0
        
        # Group by boundary
        crossings_by_boundary = {}
        for crossing in crossings:
            boundary_name = crossing['boundary']
            if boundary_name not in crossings_by_boundary:
                crossings_by_boundary[boundary_name] = []
            crossings_by_boundary[boundary_name].append(crossing)
        
        return {
            'crossings': crossings,
            'num_crossings': total_crossings,
            'crossing_rate': crossing_rate,
            'crossings_by_boundary': crossings_by_boundary
        }
    
    def _detect_blob_crossings(self, bg_analysis) -> Dict[str, Any]:
        """Detect crossings from blob tracking."""
        import cv2
        
        foreground_masks = bg_analysis.data['foreground_mask']
        
        if len(foreground_masks) == 0:
            return {
                'crossings': [],
                'num_crossings': 0,
                'crossing_rate': 0
            }
        
        h, w = foreground_masks[0].shape
        
        # Create default boundaries if not provided
        if self.boundaries is None:
            self.boundaries = self._create_default_boundaries(h, w)
        
        crossings = []
        prev_centroids = {}
        
        for frame_idx, mask in enumerate(foreground_masks):
            # Find blobs
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                mask.astype(np.uint8), connectivity=8
            )
            
            current_centroids = {}
            
            # Track each blob
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area > 100:  # Minimum area threshold
                    centroid = centroids[i]
                    
                    # Match with previous frame
                    matched_id = None
                    for prev_id, prev_cent in prev_centroids.items():
                        dist = np.sqrt((centroid[0] - prev_cent[0])**2 + 
                                     (centroid[1] - prev_cent[1])**2)
                        if dist < 50:  # Maximum matching distance
                            matched_id = prev_id
                            break
                    
                    if matched_id is not None:
                        # Check boundary crossing
                        for boundary in self.boundaries:
                            if self._check_point_crossing(
                                prev_centroids[matched_id], centroid, boundary
                            ):
                                crossings.append({
                                    'frame': frame_idx,
                                    'boundary': boundary.get('name', str(boundary)),
                                    'blob_area': int(area)
                                })
                    
                    current_centroids[i] = centroid
            
            prev_centroids = current_centroids
        
        return {
            'crossings': crossings,
            'num_crossings': len(crossings),
            'crossing_rate': len(crossings) / len(foreground_masks) if len(foreground_masks) > 0 else 0
        }
    
    def _create_default_boundaries(self, height: int, width: int) -> List[Dict]:
        """Create default boundary lines."""
        return [
            {'type': 'horizontal', 'y': height // 3, 'name': 'top_third'},
            {'type': 'horizontal', 'y': 2 * height // 3, 'name': 'bottom_third'},
            {'type': 'vertical', 'x': width // 3, 'name': 'left_third'},
            {'type': 'vertical', 'x': 2 * width // 3, 'name': 'right_third'}
        ]
    
    def _check_boundary_crossing(self, x1, y1, x2, y2, boundary) -> np.ndarray:
        """Check if motion crosses boundary."""
        if boundary['type'] == 'horizontal':
            y_bound = boundary['y']
            return ((y1 < y_bound) & (y2 >= y_bound)) | ((y1 >= y_bound) & (y2 < y_bound))
        elif boundary['type'] == 'vertical':
            x_bound = boundary['x']
            return ((x1 < x_bound) & (x2 >= x_bound)) | ((x1 >= x_bound) & (x2 < x_bound))
        else:
            return np.zeros_like(x1, dtype=bool)
    
    def _check_point_crossing(self, p1: Tuple, p2: Tuple, boundary: Dict) -> bool:
        """Check if line segment crosses boundary."""
        if boundary['type'] == 'horizontal':
            y_bound = boundary['y']
            return (p1[1] < y_bound and p2[1] >= y_bound) or (p1[1] >= y_bound and p2[1] < y_bound)
        elif boundary['type'] == 'vertical':
            x_bound = boundary['x']
            return (p1[0] < x_bound and p2[0] >= x_bound) or (p1[0] >= x_bound and p2[0] < x_bound)
        return False
    
    def _get_crossing_direction(self, x1, y1, x2, y2, boundary) -> str:
        """Determine crossing direction."""
        if boundary['type'] == 'horizontal':
            if np.mean(y2 - y1) > 0:
                return 'downward'
            else:
                return 'upward'
        elif boundary['type'] == 'vertical':
            if np.mean(x2 - x1) > 0:
                return 'rightward'
            else:
                return 'leftward'
        return 'unknown'