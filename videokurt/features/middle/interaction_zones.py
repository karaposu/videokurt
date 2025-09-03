"""Interaction zones detection from blob overlaps."""

import numpy as np
import cv2
from typing import Dict, Any, List, Tuple

from ..base import MiddleFeature


class InteractionZones(MiddleFeature):
    """Detect zones where blobs interact or overlap."""
    
    FEATURE_NAME = 'interaction_zones'
    REQUIRED_ANALYSES = ['background_mog2']  # or background_knn
    
    def __init__(self, min_overlap: float = 0.1, min_area: int = 100):
        """
        Args:
            min_overlap: Minimum overlap ratio to consider interaction
            min_area: Minimum blob area to consider
        """
        super().__init__()
        self.min_overlap = min_overlap
        self.min_area = min_area
    
    def _compute_middle(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect blob interactions and overlap zones.
        
        Returns:
            Dict with interaction events and zones
        """
        bg_analysis = analysis_data['background_mog2']
        foreground_masks = bg_analysis.data['foreground_mask']
        
        interaction_events = []
        interaction_heatmap = None
        
        for frame_idx, mask in enumerate(foreground_masks):
            # Find blobs
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                mask.astype(np.uint8), connectivity=8
            )
            
            # Initialize heatmap on first frame
            if interaction_heatmap is None:
                interaction_heatmap = np.zeros_like(mask, dtype=np.float32)
            
            # Check for interactions between blobs
            blobs = []
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area >= self.min_area:
                    blobs.append({
                        'id': i,
                        'bbox': (stats[i, cv2.CC_STAT_LEFT],
                                stats[i, cv2.CC_STAT_TOP],
                                stats[i, cv2.CC_STAT_WIDTH],
                                stats[i, cv2.CC_STAT_HEIGHT]),
                        'area': area,
                        'centroid': centroids[i],
                        'mask': labels == i
                    })
            
            # Check pairwise interactions
            for i in range(len(blobs)):
                for j in range(i + 1, len(blobs)):
                    interaction = self._check_interaction(blobs[i], blobs[j])
                    
                    if interaction['overlap_ratio'] >= self.min_overlap:
                        interaction_events.append({
                            'frame': frame_idx,
                            'blob1_area': blobs[i]['area'],
                            'blob2_area': blobs[j]['area'],
                            'overlap_area': interaction['overlap_area'],
                            'overlap_ratio': interaction['overlap_ratio'],
                            'distance': interaction['distance']
                        })
                        
                        # Mark interaction zone in heatmap
                        overlap_mask = blobs[i]['mask'] & blobs[j]['mask']
                        interaction_heatmap[overlap_mask] += 1
        
        # Normalize heatmap
        if interaction_heatmap.max() > 0:
            interaction_heatmap = interaction_heatmap / interaction_heatmap.max()
        
        # Extract interaction hotspots
        hotspots = self._extract_hotspots(interaction_heatmap)
        
        return {
            'num_interactions': len(interaction_events),
            'interaction_events': interaction_events,
            'interaction_heatmap': interaction_heatmap,
            'hotspots': hotspots,
            'avg_overlap': np.mean([e['overlap_ratio'] for e in interaction_events]) if interaction_events else 0
        }
    
    def _check_interaction(self, blob1: Dict, blob2: Dict) -> Dict:
        """Check if two blobs interact."""
        # Calculate overlap
        overlap_mask = blob1['mask'] & blob2['mask']
        overlap_area = np.sum(overlap_mask)
        
        # Calculate overlap ratio
        min_area = min(blob1['area'], blob2['area'])
        overlap_ratio = overlap_area / min_area if min_area > 0 else 0
        
        # Calculate centroid distance
        distance = np.sqrt((blob1['centroid'][0] - blob2['centroid'][0])**2 + 
                          (blob1['centroid'][1] - blob2['centroid'][1])**2)
        
        return {
            'overlap_area': overlap_area,
            'overlap_ratio': overlap_ratio,
            'distance': distance
        }
    
    def _extract_hotspots(self, heatmap: np.ndarray, top_k: int = 3) -> List[Dict]:
        """Extract top interaction hotspots."""
        # Threshold heatmap
        threshold = np.percentile(heatmap[heatmap > 0], 75) if np.any(heatmap > 0) else 0
        binary = heatmap > threshold
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary.astype(np.uint8), connectivity=8
        )
        
        hotspots = []
        for i in range(1, min(num_labels, top_k + 1)):
            mask = labels == i
            hotspots.append({
                'center': tuple(centroids[i].astype(int)),
                'area': stats[i, cv2.CC_STAT_AREA],
                'intensity': float(np.max(heatmap[mask]))
            })
        
        # Sort by intensity
        hotspots.sort(key=lambda x: x['intensity'], reverse=True)
        
        return hotspots[:top_k]