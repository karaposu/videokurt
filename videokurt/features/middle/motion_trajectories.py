"""Motion trajectory extraction from optical flow."""

import numpy as np
from typing import Dict, Any, List, Tuple

from ..base import MiddleFeature


class MotionTrajectories(MiddleFeature):
    """Extract and analyze motion trajectories from flow fields."""
    
    FEATURE_NAME = 'motion_trajectories'
    REQUIRED_ANALYSES = ['optical_flow_sparse']  # or optical_flow_dense
    
    def __init__(self, min_trajectory_length: int = 5, 
                 max_trajectories: int = 100):
        """
        Args:
            min_trajectory_length: Minimum frames for valid trajectory
            max_trajectories: Maximum number of trajectories to track
        """
        super().__init__()
        self.min_trajectory_length = min_trajectory_length
        self.max_trajectories = max_trajectories
    
    def _compute_middle(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract motion trajectories from optical flow.
        
        Returns:
            Dict with trajectory paths and statistics
        """
        if 'optical_flow_sparse' in analysis_data:
            return self._extract_sparse_trajectories(analysis_data['optical_flow_sparse'])
        elif 'optical_flow_dense' in analysis_data:
            return self._extract_dense_trajectories(analysis_data['optical_flow_dense'])
        else:
            raise ValueError("Need optical_flow_sparse or optical_flow_dense")
    
    def _extract_sparse_trajectories(self, flow_analysis) -> Dict[str, Any]:
        """Extract trajectories from sparse optical flow."""
        tracked_points = flow_analysis.data.get('tracked_points', [])
        point_status = flow_analysis.data.get('point_status', [])
        
        if not tracked_points:
            return {
                'trajectories': [],
                'num_trajectories': 0,
                'avg_length': 0,
                'avg_displacement': 0
            }
        
        # Build trajectories
        trajectories = []
        current_trajectories = {}
        next_id = 0
        
        for frame_idx, (points, status) in enumerate(zip(tracked_points, point_status)):
            if points is None:
                continue
            
            for point_idx, (point, is_valid) in enumerate(zip(points, status)):
                if is_valid:
                    if point_idx not in current_trajectories:
                        current_trajectories[point_idx] = {
                            'id': next_id,
                            'points': [point],
                            'start_frame': frame_idx
                        }
                        next_id += 1
                    else:
                        current_trajectories[point_idx]['points'].append(point)
                else:
                    # End trajectory
                    if point_idx in current_trajectories:
                        traj = current_trajectories[point_idx]
                        if len(traj['points']) >= self.min_trajectory_length:
                            trajectories.append(traj)
                        del current_trajectories[point_idx]
        
        # Add remaining trajectories
        for traj in current_trajectories.values():
            if len(traj['points']) >= self.min_trajectory_length:
                trajectories.append(traj)
        
        # Limit number of trajectories
        trajectories = trajectories[:self.max_trajectories]
        
        # Compute statistics
        trajectory_stats = self._compute_trajectory_stats(trajectories)
        
        return {
            'trajectories': trajectories,
            'num_trajectories': len(trajectories),
            **trajectory_stats
        }
    
    def _extract_dense_trajectories(self, flow_analysis) -> Dict[str, Any]:
        """Extract trajectories from dense optical flow."""
        flow_field = flow_analysis.data['flow_field']
        
        if len(flow_field) == 0:
            return {
                'trajectories': [],
                'num_trajectories': 0,
                'avg_length': 0,
                'avg_displacement': 0
            }
        
        # Sample points to track
        h, w = flow_field[0].shape[:2]
        grid_size = 20  # Sample every 20 pixels
        
        trajectories = []
        
        # Create grid of starting points
        for y in range(0, h, grid_size):
            for x in range(0, w, grid_size):
                trajectory = {'points': [(x, y)], 'start_frame': 0}
                current_x, current_y = float(x), float(y)
                
                for flow in flow_field:
                    # Get flow at current position
                    ix, iy = int(current_x), int(current_y)
                    if 0 <= ix < w and 0 <= iy < h:
                        dx = flow[iy, ix, 0]
                        dy = flow[iy, ix, 1]
                        
                        # Update position
                        current_x += dx
                        current_y += dy
                        trajectory['points'].append((current_x, current_y))
                    else:
                        break
                
                if len(trajectory['points']) >= self.min_trajectory_length:
                    trajectories.append(trajectory)
                
                if len(trajectories) >= self.max_trajectories:
                    break
            if len(trajectories) >= self.max_trajectories:
                break
        
        # Compute statistics
        trajectory_stats = self._compute_trajectory_stats(trajectories)
        
        return {
            'trajectories': trajectories,
            'num_trajectories': len(trajectories),
            **trajectory_stats
        }
    
    def _compute_trajectory_stats(self, trajectories: List[Dict]) -> Dict[str, Any]:
        """Compute statistics about trajectories."""
        if not trajectories:
            return {
                'avg_length': 0,
                'avg_displacement': 0,
                'avg_speed': 0
            }
        
        lengths = []
        displacements = []
        speeds = []
        
        for traj in trajectories:
            points = traj['points']
            lengths.append(len(points))
            
            # Compute displacement
            if len(points) >= 2:
                start = points[0]
                end = points[-1]
                displacement = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                displacements.append(displacement)
                
                # Compute average speed
                total_dist = 0
                for i in range(1, len(points)):
                    dist = np.sqrt((points[i][0] - points[i-1][0])**2 + 
                                 (points[i][1] - points[i-1][1])**2)
                    total_dist += dist
                avg_speed = total_dist / len(points)
                speeds.append(avg_speed)
        
        return {
            'avg_length': float(np.mean(lengths)) if lengths else 0,
            'avg_displacement': float(np.mean(displacements)) if displacements else 0,
            'avg_speed': float(np.mean(speeds)) if speeds else 0
        }