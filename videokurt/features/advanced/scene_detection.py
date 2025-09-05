"""Scene boundary detection using multiple visual cues."""

import numpy as np
from typing import Dict, Any, List, Tuple

from ..base import BaseFeature


class SceneDetection(BaseFeature):
    """Detect scene boundaries (cuts, fades, transitions) from multiple features."""
    
    FEATURE_NAME = 'scene_detection'
    REQUIRED_ANALYSES = ['frame_diff', 'edge_canny']
    
    def __init__(self, cut_threshold: float = 0.5, 
                 fade_threshold: float = 0.3,
                 min_scene_length: int = 10):
        """
        Args:
            cut_threshold: Threshold for detecting hard cuts
            fade_threshold: Threshold for detecting fades
            min_scene_length: Minimum frames between scene changes
        """
        super().__init__()
        self.cut_threshold = cut_threshold
        self.fade_threshold = fade_threshold
        self.min_scene_length = min_scene_length
    
    def compute(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect scene boundaries using multiple visual patterns.
        
        Returns:
            Dict with scene boundaries, types, and confidence scores
        """
        # Get frame differences
        frame_diff = analysis_data['frame_diff'].data['pixel_diff']
        
        # Get edge changes
        edge_maps = analysis_data['edge_canny'].data['edge_map']
        
        # Compute normalized frame-to-frame changes
        frame_changes = []
        for diff in frame_diff:
            normalized_change = np.mean(diff) / 255.0
            frame_changes.append(normalized_change)
        frame_changes = np.array(frame_changes)
        
        # Compute edge density changes
        edge_changes = []
        prev_edges = None
        for edges in edge_maps:
            if prev_edges is not None:
                edge_diff = np.abs(edges.astype(float) - prev_edges.astype(float))
                edge_change = np.mean(edge_diff) / 255.0
                edge_changes.append(edge_change)
            prev_edges = edges
        edge_changes = np.array(edge_changes)
        
        # Detect different types of transitions
        cuts = self._detect_cuts(frame_changes, edge_changes)
        fades = self._detect_fades(frame_changes)
        
        # Merge and filter scenes
        all_boundaries = []
        for frame, confidence in cuts:
            all_boundaries.append({
                'frame': frame,
                'type': 'cut',
                'confidence': confidence
            })
        for start, end, confidence in fades:
            all_boundaries.append({
                'frame': start,
                'type': 'fade_out',
                'confidence': confidence
            })
            all_boundaries.append({
                'frame': end,
                'type': 'fade_in',
                'confidence': confidence
            })
        
        # Sort by frame number
        all_boundaries.sort(key=lambda x: x['frame'])
        
        # Filter by minimum scene length
        filtered_boundaries = self._filter_by_min_length(all_boundaries)
        
        # Extract scene segments
        scenes = self._extract_scenes(filtered_boundaries, len(frame_diff))
        
        return {
            'boundaries': filtered_boundaries,
            'scenes': scenes,
            'num_scenes': len(scenes),
            'avg_scene_length': np.mean([s['length'] for s in scenes]) if scenes else 0
        }
    
    def _detect_cuts(self, frame_changes: np.ndarray, 
                     edge_changes: np.ndarray) -> List[Tuple[int, float]]:
        """Detect hard cuts based on sudden changes."""
        cuts = []
        
        # Combine frame and edge changes (handle potential length mismatch)
        if len(edge_changes) > 0:
            min_len = min(len(frame_changes), len(edge_changes))
            combined_changes = (frame_changes[:min_len] + edge_changes[:min_len]) / 2
        else:
            combined_changes = frame_changes
        
        # Find peaks above threshold
        for i, change in enumerate(combined_changes):
            if change > self.cut_threshold:
                # Check if it's a local maximum
                is_peak = True
                if i > 0 and combined_changes[i-1] >= change:
                    is_peak = False
                if i < len(combined_changes) - 1 and combined_changes[i+1] >= change:
                    is_peak = False
                
                if is_peak:
                    cuts.append((i, float(change)))
        
        return cuts
    
    def _detect_fades(self, frame_changes: np.ndarray) -> List[Tuple[int, int, float]]:
        """Detect fade in/out based on gradual changes."""
        fades = []
        
        # Look for sustained low-level changes
        in_fade = False
        fade_start = 0
        fade_sum = 0
        
        for i, change in enumerate(frame_changes):
            if not in_fade and change > self.fade_threshold * 0.5:
                in_fade = True
                fade_start = i
                fade_sum = change
            elif in_fade:
                fade_sum += change
                if change < self.fade_threshold * 0.3 or i == len(frame_changes) - 1:
                    # Fade ended
                    if i - fade_start > 5:  # Minimum fade length
                        avg_change = fade_sum / (i - fade_start)
                        fades.append((fade_start, i, avg_change))
                    in_fade = False
        
        return fades
    
    def _filter_by_min_length(self, boundaries: List[Dict]) -> List[Dict]:
        """Filter out boundaries that are too close together."""
        if not boundaries:
            return []
        
        filtered = [boundaries[0]]
        for boundary in boundaries[1:]:
            if boundary['frame'] - filtered[-1]['frame'] >= self.min_scene_length:
                filtered.append(boundary)
        
        return filtered
    
    def _extract_scenes(self, boundaries: List[Dict], total_frames: int) -> List[Dict]:
        """Extract scene segments from boundaries."""
        scenes = []
        
        # Add first scene if needed
        if not boundaries or boundaries[0]['frame'] > 0:
            start = 0
            end = boundaries[0]['frame'] if boundaries else total_frames
            scenes.append({
                'start': start,
                'end': end,
                'length': end - start
            })
        
        # Add scenes between boundaries
        for i in range(len(boundaries) - 1):
            start = boundaries[i]['frame']
            end = boundaries[i + 1]['frame']
            scenes.append({
                'start': start,
                'end': end,
                'length': end - start
            })
        
        # Add last scene if needed
        if boundaries and boundaries[-1]['frame'] < total_frames:
            scenes.append({
                'start': boundaries[-1]['frame'],
                'end': total_frames,
                'length': total_frames - boundaries[-1]['frame']
            })
        
        return scenes