"""Blob stability (persistence) analysis."""

import numpy as np
import cv2
from typing import Dict, Any, List

from ..base import BaseFeature


class BlobStability(BaseFeature):
    """Measure blob persistence over time."""
    
    FEATURE_NAME = 'blob_stability'
    REQUIRED_ANALYSES = ['background_mog2']  # or background_knn
    
    def __init__(self, min_persistence: int = 5, min_area: int = 100):
        """
        Args:
            min_persistence: Minimum frames a blob must persist
            min_area: Minimum blob area to track
        """
        super().__init__()
        self.min_persistence = min_persistence
        self.min_area = min_area
    
    def compute(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Track blob persistence and stability.
        
        Returns:
            Dict with persistence scores and stable blob info
        """
        self.validate_inputs(analysis_data)
        
        bg_analysis = analysis_data['background_mog2']
        foreground_masks = bg_analysis.data['foreground_mask']
        
        blob_tracker = {}
        next_id = 0
        persistence_scores = []
        
        for frame_idx, mask in enumerate(foreground_masks):
            # Find blobs in current frame
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                mask.astype(np.uint8), connectivity=8
            )
            
            current_blobs = {}
            
            # Process each blob (skip background)
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area >= self.min_area:
                    centroid = centroids[i]
                    
                    # Find matching blob from previous frame
                    matched_id = None
                    min_dist = float('inf')
                    
                    for blob_id, blob_info in blob_tracker.items():
                        if blob_info['last_seen'] == frame_idx - 1:
                            dist = np.sqrt((centroid[0] - blob_info['centroid'][0])**2 + 
                                         (centroid[1] - blob_info['centroid'][1])**2)
                            if dist < min_dist and dist < 50:  # 50 pixel threshold
                                min_dist = dist
                                matched_id = blob_id
                    
                    if matched_id is not None:
                        # Update existing blob
                        blob_tracker[matched_id]['centroid'] = centroid
                        blob_tracker[matched_id]['last_seen'] = frame_idx
                        blob_tracker[matched_id]['persistence'] += 1
                        blob_tracker[matched_id]['areas'].append(area)
                        current_blobs[matched_id] = blob_tracker[matched_id]
                    else:
                        # New blob
                        new_id = next_id
                        next_id += 1
                        blob_tracker[new_id] = {
                            'centroid': centroid,
                            'first_seen': frame_idx,
                            'last_seen': frame_idx,
                            'persistence': 1,
                            'areas': [area]
                        }
                        current_blobs[new_id] = blob_tracker[new_id]
            
            # Compute frame stability score
            if current_blobs:
                persistences = [b['persistence'] for b in current_blobs.values()]
                stability_score = np.mean(persistences) / max(1, frame_idx + 1)
            else:
                stability_score = 0.0
            
            persistence_scores.append(stability_score)
        
        # Find stable blobs
        stable_blobs = []
        for blob_id, blob_info in blob_tracker.items():
            if blob_info['persistence'] >= self.min_persistence:
                stable_blobs.append({
                    'id': blob_id,
                    'persistence': blob_info['persistence'],
                    'avg_area': np.mean(blob_info['areas']),
                    'area_variance': np.var(blob_info['areas'])
                })
        
        return {
            'persistence_scores': np.array(persistence_scores),
            'stable_blob_count': len(stable_blobs),
            'stable_blobs': stable_blobs,
            'avg_persistence': np.mean([b['persistence'] for b in blob_tracker.values()]) if blob_tracker else 0
        }