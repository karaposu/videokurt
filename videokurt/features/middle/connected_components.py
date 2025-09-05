"""Connected components analysis."""

import numpy as np
import cv2
from typing import Dict, Any, List

from ..base import BaseFeature


class ConnectedComponents(BaseFeature):
    """Analyze connected components in binary masks."""
    
    FEATURE_NAME = 'connected_components'
    REQUIRED_ANALYSES = ['background_mog2']  # or edge_canny after thresholding
    
    def __init__(self, min_area: int = 50, max_components: int = 100):
        """
        Args:
            min_area: Minimum component area to consider
            max_components: Maximum components to track per frame
        """
        super().__init__()
        self.min_area = min_area
        self.max_components = max_components
    
    def compute(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze connected components.
        
        Returns:
            Dict with component statistics and properties
        """
        self.validate_inputs(analysis_data)
        
        # Get binary masks
        if 'background_mog2' in analysis_data:
            masks = analysis_data['background_mog2'].data['foreground_mask']
        elif 'background_knn' in analysis_data:
            masks = analysis_data['background_knn'].data['foreground_mask']
        elif 'edge_canny' in analysis_data:
            # Threshold edge maps to get binary
            edge_maps = analysis_data['edge_canny'].data['edge_map']
            masks = (edge_maps > 0).astype(np.uint8) * 255
        else:
            raise ValueError("Need background_mog2, background_knn, or edge_canny")
        
        component_data = []
        
        for frame_idx, mask in enumerate(masks):
            # Find connected components
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                mask.astype(np.uint8), connectivity=8
            )
            
            frame_components = []
            
            # Process components (skip background label 0)
            for i in range(1, min(num_labels, self.max_components + 1)):
                area = stats[i, cv2.CC_STAT_AREA]
                
                if area >= self.min_area:
                    component = {
                        'area': int(area),
                        'centroid': tuple(centroids[i].astype(int)),
                        'bbox': (
                            stats[i, cv2.CC_STAT_LEFT],
                            stats[i, cv2.CC_STAT_TOP],
                            stats[i, cv2.CC_STAT_WIDTH],
                            stats[i, cv2.CC_STAT_HEIGHT]
                        ),
                        'aspect_ratio': stats[i, cv2.CC_STAT_WIDTH] / max(1, stats[i, cv2.CC_STAT_HEIGHT]),
                        'density': area / (stats[i, cv2.CC_STAT_WIDTH] * stats[i, cv2.CC_STAT_HEIGHT])
                    }
                    frame_components.append(component)
            
            component_data.append({
                'frame': frame_idx,
                'num_components': len(frame_components),
                'components': frame_components,
                'total_area': sum(c['area'] for c in frame_components)
            })
        
        # Compute statistics
        num_components_timeline = [d['num_components'] for d in component_data]
        total_area_timeline = [d['total_area'] for d in component_data]
        
        # Find stable components (appear in multiple frames)
        stable_components = self._find_stable_components(component_data)
        
        return {
            'component_timeline': component_data,
            'num_components_timeline': np.array(num_components_timeline),
            'total_area_timeline': np.array(total_area_timeline),
            'avg_components': float(np.mean(num_components_timeline)) if num_components_timeline else 0,
            'max_components': int(np.max(num_components_timeline)) if num_components_timeline else 0,
            'stable_components': stable_components
        }
    
    def _find_stable_components(self, component_data: List[Dict], 
                               min_frames: int = 5) -> List[Dict]:
        """Find components that persist across frames."""
        
        if len(component_data) < min_frames:
            return []
        
        # Simple tracking based on centroid proximity
        stable = []
        tracked = {}
        next_id = 0
        
        for frame_data in component_data:
            current_tracked = {}
            
            for component in frame_data['components']:
                centroid = component['centroid']
                
                # Find closest tracked component
                matched_id = None
                min_dist = float('inf')
                
                for comp_id, prev_cent in tracked.items():
                    dist = np.sqrt((centroid[0] - prev_cent[0])**2 + 
                                 (centroid[1] - prev_cent[1])**2)
                    if dist < min_dist and dist < 50:  # 50 pixel threshold
                        min_dist = dist
                        matched_id = comp_id
                
                if matched_id is not None:
                    current_tracked[matched_id] = centroid
                    if matched_id not in stable:
                        stable[matched_id] = {
                            'id': matched_id,
                            'frames': 0,
                            'avg_area': 0
                        }
                    stable[matched_id]['frames'] += 1
                    stable[matched_id]['avg_area'] += component['area']
                else:
                    # New component
                    current_tracked[next_id] = centroid
                    next_id += 1
            
            tracked = current_tracked
        
        # Filter by minimum frames and compute averages
        stable_list = []
        for comp_id, data in stable.items():
            if data['frames'] >= min_frames:
                data['avg_area'] /= data['frames']
                stable_list.append(data)
        
        return stable_list
