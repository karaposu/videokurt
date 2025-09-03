"""Blob detection and tracking from background subtraction."""

import numpy as np
import cv2
from typing import Dict, Any, List, Tuple

from ..base import MiddleFeature


class BlobTracking(MiddleFeature):
    """Extract and track blobs from foreground masks."""
    
    FEATURE_NAME = 'blob_tracking'
    REQUIRED_ANALYSES = ['background_mog2']  # or background_knn
    
    def __init__(self, min_area: int = 100, max_area: int = 10000):
        """
        Args:
            min_area: Minimum blob area in pixels
            max_area: Maximum blob area in pixels
        """
        super().__init__()
        self.min_area = min_area
        self.max_area = max_area
    
    def _compute_middle(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract blob properties and simple tracking.
        
        Returns:
            Dict with blob counts, sizes, positions per frame
        """
        # Get foreground masks
        bg_analysis = analysis_data['background_mog2']
        foreground_masks = bg_analysis.data['foreground_mask']
        
        blob_data = {
            'counts': [],
            'sizes': [],
            'centroids': [],
            'bounding_boxes': []
        }
        
        for mask in foreground_masks:
            # Find connected components
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                mask.astype(np.uint8), connectivity=8
            )
            
            # Filter blobs by size (skip background label 0)
            valid_blobs = []
            valid_centroids = []
            valid_bboxes = []
            
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if self.min_area <= area <= self.max_area:
                    valid_blobs.append(area)
                    valid_centroids.append(centroids[i])
                    # Get bounding box (x, y, width, height)
                    bbox = (
                        stats[i, cv2.CC_STAT_LEFT],
                        stats[i, cv2.CC_STAT_TOP],
                        stats[i, cv2.CC_STAT_WIDTH],
                        stats[i, cv2.CC_STAT_HEIGHT]
                    )
                    valid_bboxes.append(bbox)
            
            blob_data['counts'].append(len(valid_blobs))
            blob_data['sizes'].append(valid_blobs)
            blob_data['centroids'].append(valid_centroids)
            blob_data['bounding_boxes'].append(valid_bboxes)
        
        # Add simple tracking (nearest neighbor between frames)
        trajectories = self._simple_tracking(blob_data['centroids'])
        blob_data['trajectories'] = trajectories
        
        return blob_data
    
    def _simple_tracking(self, centroids_list: List[List[Tuple]]) -> List[List[int]]:
        """Simple nearest neighbor tracking between frames."""
        if not centroids_list:
            return []
        
        trajectories = []
        prev_centroids = []
        prev_ids = []
        next_id = 0
        
        for centroids in centroids_list:
            current_ids = []
            
            if not prev_centroids:
                # First frame - assign new IDs
                current_ids = list(range(next_id, next_id + len(centroids)))
                next_id += len(centroids)
            else:
                # Match to previous frame
                for centroid in centroids:
                    if prev_centroids:
                        # Find nearest previous centroid
                        distances = [
                            np.sqrt((centroid[0] - prev[0])**2 + 
                                   (centroid[1] - prev[1])**2)
                            for prev in prev_centroids
                        ]
                        min_idx = np.argmin(distances)
                        
                        # Assign ID if close enough (within 50 pixels)
                        if distances[min_idx] < 50:
                            current_ids.append(prev_ids[min_idx])
                        else:
                            current_ids.append(next_id)
                            next_id += 1
                    else:
                        current_ids.append(next_id)
                        next_id += 1
            
            trajectories.append(current_ids)
            prev_centroids = centroids
            prev_ids = current_ids
        
        return trajectories