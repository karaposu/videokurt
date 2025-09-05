"""UI change detection for screen recordings and regular videos."""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from scipy import signal, stats
from scipy.ndimage import label, binary_dilation, binary_erosion
from scipy.spatial.distance import cosine

from ..base import BaseFeature


class UIChangeDetection(BaseFeature):
    """Detect UI changes using structural analysis and pattern recognition."""
    
    FEATURE_NAME = 'ui_change_detection'
    REQUIRED_ANALYSES = ['edge_canny', 'frame_diff', 'color_histogram']
    
    def __init__(self, 
                 change_threshold: float = 0.15,
                 structure_threshold: float = 0.2,
                 region_size_threshold: float = 0.05,
                 temporal_window: int = 5,
                 edge_density_bins: int = 20,
                 color_similarity_threshold: float = 0.85):
        """
        Args:
            change_threshold: Threshold for significant UI change
            structure_threshold: Threshold for structural changes
            region_size_threshold: Minimum region size (fraction of frame)
            temporal_window: Window for temporal consistency
            edge_density_bins: Number of bins for edge density histogram
            color_similarity_threshold: Threshold for color distribution similarity
        """
        super().__init__()
        self.change_threshold = change_threshold
        self.structure_threshold = structure_threshold
        self.region_size_threshold = region_size_threshold
        self.temporal_window = temporal_window
        self.edge_density_bins = edge_density_bins
        self.color_similarity_threshold = color_similarity_threshold
    
    def compute(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect UI changes using multi-modal analysis.
        
        Returns:
            Dict with UI change events, regions, and patterns
        """
        edge_maps = analysis_data['edge_canny'].data['edge_map']
        frame_diffs = analysis_data['frame_diff'].data['pixel_diff']
        color_hists = analysis_data['color_histogram'].data['histograms']
        
        if len(edge_maps) == 0:
            return self._empty_result()
        
        # Extract structural features
        structural_features = self._extract_structural_features(edge_maps)
        
        # Analyze color distribution changes
        color_changes = self._analyze_color_changes(color_hists)
        
        # Detect change regions with connected components
        change_regions = self._detect_change_regions(frame_diffs, edge_maps)
        
        # Classify UI changes with multi-modal fusion
        ui_changes = self._classify_ui_changes(
            structural_features, color_changes, change_regions, frame_diffs
        )
        
        # Apply temporal consistency filtering
        filtered_changes = self._temporal_filtering(ui_changes)
        
        # Group into UI events
        ui_events = self._group_into_events(filtered_changes)
        
        # Detect patterns
        patterns = self._detect_ui_patterns(ui_events)
        
        return {
            'ui_changes': filtered_changes,
            'ui_events': ui_events,
            'change_regions': self._summarize_regions(change_regions),
            'patterns': patterns,
            'structural_features': {
                'edge_density_curve': structural_features['edge_densities'],
                'layout_complexity': structural_features['layout_complexity']
            },
            'statistics': self._compute_statistics(filtered_changes, ui_events)
        }
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result structure."""
        return {
            'ui_changes': [],
            'ui_events': [],
            'change_regions': [],
            'patterns': {},
            'structural_features': {
                'edge_density_curve': [],
                'layout_complexity': []
            },
            'statistics': {
                'num_changes': 0,
                'change_rate': 0.0,
                'dominant_type': 'none'
            }
        }
    
    def _extract_structural_features(self, edge_maps: List[np.ndarray]) -> Dict[str, Any]:
        """Extract structural features from edge maps."""
        features = {
            'edge_densities': [],
            'edge_distributions': [],
            'layout_complexity': [],
            'grid_alignments': []
        }
        
        for edges in edge_maps:
            # Edge density
            density = np.mean(edges > 0)
            features['edge_densities'].append(density)
            
            # Edge distribution (spatial)
            h, w = edges.shape
            grid_h, grid_w = 4, 4  # 4x4 grid
            distribution = []
            
            for i in range(grid_h):
                for j in range(grid_w):
                    region = edges[
                        i*h//grid_h:(i+1)*h//grid_h,
                        j*w//grid_w:(j+1)*w//grid_w
                    ]
                    distribution.append(np.mean(region > 0))
            
            features['edge_distributions'].append(distribution)
            
            # Layout complexity (entropy of edge distribution)
            hist, _ = np.histogram(edges.flatten(), bins=self.edge_density_bins)
            hist = hist / (hist.sum() + 1e-10)
            entropy = -np.sum(hist * np.log(hist + 1e-10))
            features['layout_complexity'].append(entropy)
            
            # Grid alignment (detect rectangular structures)
            alignment_score = self._compute_grid_alignment(edges)
            features['grid_alignments'].append(alignment_score)
        
        return features
    
    def _compute_grid_alignment(self, edges: np.ndarray) -> float:
        """Compute grid alignment score for detecting UI layouts."""
        # Detect horizontal and vertical lines
        h_kernel = np.ones((1, 15))
        v_kernel = np.ones((15, 1))
        
        h_lines = signal.convolve2d(edges.astype(float), h_kernel, mode='same')
        v_lines = signal.convolve2d(edges.astype(float), v_kernel, mode='same')
        
        # Threshold to get strong lines
        h_threshold = np.percentile(h_lines, 90)
        v_threshold = np.percentile(v_lines, 90)
        
        h_strong = h_lines > h_threshold
        v_strong = v_lines > v_threshold
        
        # Score based on line regularity
        h_projection = np.sum(h_strong, axis=1)
        v_projection = np.sum(v_strong, axis=0)
        
        # Find peaks (potential grid lines)
        h_peaks = signal.find_peaks(h_projection, distance=20)[0]
        v_peaks = signal.find_peaks(v_projection, distance=20)[0]
        
        if len(h_peaks) > 1 and len(v_peaks) > 1:
            # Check spacing regularity
            h_spacings = np.diff(h_peaks)
            v_spacings = np.diff(v_peaks)
            
            h_regularity = 1.0 - np.std(h_spacings) / (np.mean(h_spacings) + 1e-6)
            v_regularity = 1.0 - np.std(v_spacings) / (np.mean(v_spacings) + 1e-6)
            
            return float(np.mean([h_regularity, v_regularity]))
        
        return 0.0
    
    def _analyze_color_changes(self, color_hists: List[np.ndarray]) -> List[Dict]:
        """Analyze color distribution changes between frames."""
        changes = []
        
        for i in range(1, len(color_hists)):
            prev_hist = color_hists[i-1].flatten()
            curr_hist = color_hists[i].flatten()
            
            # Normalize histograms
            prev_hist = prev_hist / (np.sum(prev_hist) + 1e-10)
            curr_hist = curr_hist / (np.sum(curr_hist) + 1e-10)
            
            # Multiple similarity metrics
            cosine_sim = 1.0 - cosine(prev_hist, curr_hist)
            bhattacharyya = np.sqrt(np.sum(np.sqrt(prev_hist * curr_hist)))
            chi_square = np.sum((prev_hist - curr_hist)**2 / (prev_hist + curr_hist + 1e-10))
            
            # Detect significant color shifts
            color_shift = self._detect_color_shift(prev_hist, curr_hist)
            
            changes.append({
                'frame': i,
                'cosine_similarity': cosine_sim,
                'bhattacharyya': bhattacharyya,
                'chi_square': chi_square,
                'color_shift': color_shift,
                'is_significant': cosine_sim < self.color_similarity_threshold
            })
        
        return changes
    
    def _detect_color_shift(self, prev_hist: np.ndarray, curr_hist: np.ndarray) -> Dict:
        """Detect the type and magnitude of color shift."""
        # Find dominant colors
        prev_peaks = signal.find_peaks(prev_hist, height=0.05)[0]
        curr_peaks = signal.find_peaks(curr_hist, height=0.05)[0]
        
        if len(prev_peaks) > 0 and len(curr_peaks) > 0:
            prev_dominant = prev_peaks[np.argmax(prev_hist[prev_peaks])]
            curr_dominant = curr_peaks[np.argmax(curr_hist[curr_peaks])]
            
            shift_magnitude = abs(curr_dominant - prev_dominant) / len(prev_hist)
            
            return {
                'has_shift': shift_magnitude > 0.1,
                'magnitude': float(shift_magnitude),
                'direction': 'lighter' if curr_dominant > prev_dominant else 'darker'
            }
        
        return {'has_shift': False, 'magnitude': 0.0, 'direction': 'none'}
    
    def _detect_change_regions(self, frame_diffs: List[np.ndarray], 
                               edge_maps: List[np.ndarray]) -> List[Dict]:
        """Detect and characterize regions of change."""
        regions = []
        
        for i, diff in enumerate(frame_diffs):
            if i + 1 >= len(edge_maps):
                break
            
            # Threshold difference map
            threshold = np.percentile(diff, 85)
            binary_diff = diff > threshold
            
            # Apply morphological operations
            binary_diff = binary_erosion(binary_diff, iterations=1)
            binary_diff = binary_dilation(binary_diff, iterations=2)
            
            # Find connected components
            labeled, num_features = label(binary_diff)
            
            frame_regions = []
            h, w = diff.shape
            
            for label_id in range(1, num_features + 1):
                mask = labeled == label_id
                area = np.sum(mask)
                
                # Skip small regions
                if area < h * w * self.region_size_threshold:
                    continue
                
                # Get bounding box
                coords = np.where(mask)
                y_min, y_max = coords[0].min(), coords[0].max()
                x_min, x_max = coords[1].min(), coords[1].max()
                
                # Characterize region
                region_edges = edge_maps[i+1][mask] if i+1 < len(edge_maps) else np.array([])
                edge_density = np.mean(region_edges > 0) if len(region_edges) > 0 else 0
                
                # Determine region type
                region_type = self._classify_region(
                    mask, (y_min, y_max, x_min, x_max), edge_density, h, w
                )
                
                frame_regions.append({
                    'bbox': (int(x_min), int(y_min), int(x_max), int(y_max)),
                    'area': float(area / (h * w)),
                    'edge_density': float(edge_density),
                    'type': region_type,
                    'centroid': (int((x_min + x_max) / 2), int((y_min + y_max) / 2))
                })
            
            if frame_regions:
                regions.append({
                    'frame': i + 1,
                    'regions': frame_regions,
                    'num_regions': len(frame_regions)
                })
        
        return regions
    
    def _classify_region(self, mask: np.ndarray, bbox: Tuple, 
                        edge_density: float, h: int, w: int) -> str:
        """Classify the type of UI region."""
        y_min, y_max, x_min, x_max = bbox
        region_w = x_max - x_min
        region_h = y_max - y_min
        
        # Position-based classification
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        
        # Check if centered (likely dialog/popup)
        if abs(center_x - w/2) < w * 0.1 and abs(center_y - h/2) < h * 0.1:
            if region_w < w * 0.7 and region_h < h * 0.7:
                return 'dialog'
        
        # Check if full-width (likely header/footer)
        if region_w > w * 0.9:
            if region_h < h * 0.2:
                if center_y < h * 0.3:
                    return 'header'
                elif center_y > h * 0.7:
                    return 'footer'
                else:
                    return 'banner'
        
        # Check if sidebar
        if region_h > h * 0.7 and region_w < w * 0.3:
            if center_x < w * 0.3:
                return 'left_sidebar'
            elif center_x > w * 0.7:
                return 'right_sidebar'
        
        # Check if corner notification
        if region_w < w * 0.3 and region_h < h * 0.3:
            corners = [
                (w * 0.15, h * 0.15, 'top_left'),
                (w * 0.85, h * 0.15, 'top_right'),
                (w * 0.15, h * 0.85, 'bottom_left'),
                (w * 0.85, h * 0.85, 'bottom_right')
            ]
            
            for cx, cy, corner_name in corners:
                if abs(center_x - cx) < w * 0.15 and abs(center_y - cy) < h * 0.15:
                    return f'{corner_name}_notification'
        
        # Default based on edge density
        if edge_density > 0.3:
            return 'content_panel'
        else:
            return 'overlay'
    
    def _classify_ui_changes(self, structural: Dict, color: List[Dict], 
                            regions: List[Dict], diffs: List[np.ndarray]) -> List[Dict]:
        """Classify UI changes using multi-modal fusion."""
        changes = []
        
        for i in range(1, len(structural['edge_densities'])):
            # Structural change
            edge_change = abs(structural['edge_densities'][i] - 
                            structural['edge_densities'][i-1])
            complexity_change = abs(structural['layout_complexity'][i] - 
                                  structural['layout_complexity'][i-1])
            
            # Color change
            color_change = color[i-1] if i-1 < len(color) else None
            
            # Pixel change
            pixel_change = np.mean(diffs[i-1]) / 255.0 if i-1 < len(diffs) else 0
            
            # Region analysis
            frame_regions = next((r for r in regions if r['frame'] == i), None)
            
            # Classify change type
            change_type, confidence = self._determine_change_type(
                edge_change, complexity_change, color_change, 
                pixel_change, frame_regions
            )
            
            if change_type != 'none':
                changes.append({
                    'frame': i,
                    'type': change_type,
                    'confidence': confidence,
                    'metrics': {
                        'edge_change': float(edge_change),
                        'complexity_change': float(complexity_change),
                        'pixel_change': float(pixel_change),
                        'color_similarity': float(color_change['cosine_similarity']) 
                            if color_change else 1.0
                    },
                    'regions': frame_regions['regions'] if frame_regions else []
                })
        
        return changes
    
    def _determine_change_type(self, edge_change: float, complexity_change: float,
                               color_change: Optional[Dict], pixel_change: float,
                               regions: Optional[Dict]) -> Tuple[str, float]:
        """Determine UI change type and confidence."""
        # Major transition
        if pixel_change > self.change_threshold and edge_change > self.structure_threshold:
            if complexity_change > 0.5:
                return 'screen_transition', 0.9
            else:
                return 'major_update', 0.85
        
        # Dialog/popup appearance
        if regions and len(regions['regions']) == 1:
            region = regions['regions'][0]
            if region['type'] in ['dialog', 'overlay']:
                return 'dialog_open', 0.8
            elif 'notification' in region['type']:
                return 'notification', 0.75
        
        # Sidebar toggle
        if regions:
            for region in regions['regions']:
                if 'sidebar' in region['type']:
                    return 'sidebar_toggle', 0.8
        
        # Content update
        if pixel_change > self.change_threshold * 0.5:
            if color_change and color_change['is_significant']:
                return 'theme_change', 0.7
            elif edge_change < self.structure_threshold * 0.5:
                return 'content_refresh', 0.6
            else:
                return 'layout_adjustment', 0.65
        
        # Animation/transition
        if pixel_change > 0.05 and pixel_change < self.change_threshold * 0.5:
            return 'animation_frame', 0.5
        
        return 'none', 0.0
    
    def _temporal_filtering(self, changes: List[Dict]) -> List[Dict]:
        """Apply temporal consistency filtering."""
        if len(changes) <= self.temporal_window:
            return changes
        
        filtered = []
        
        for i, change in enumerate(changes):
            # Look at surrounding frames
            window_start = max(0, i - self.temporal_window // 2)
            window_end = min(len(changes), i + self.temporal_window // 2 + 1)
            window = changes[window_start:window_end]
            
            # Check consistency
            similar_changes = [c for c in window 
                             if c['type'] == change['type'] or 
                             abs(c['frame'] - change['frame']) <= 2]
            
            if len(similar_changes) >= 2 or change['confidence'] > 0.7:
                filtered.append(change)
        
        return filtered
    
    def _group_into_events(self, changes: List[Dict]) -> List[Dict]:
        """Group related changes into UI events."""
        if not changes:
            return []
        
        events = []
        current_event = None
        
        for change in changes:
            if current_event is None:
                current_event = {
                    'type': change['type'],
                    'start_frame': change['frame'],
                    'end_frame': change['frame'],
                    'confidence': change['confidence'],
                    'changes': [change]
                }
            elif (change['frame'] - current_event['end_frame'] <= 5 and
                  self._are_related_changes(change['type'], current_event['type'])):
                current_event['end_frame'] = change['frame']
                current_event['confidence'] = max(current_event['confidence'], 
                                                 change['confidence'])
                current_event['changes'].append(change)
            else:
                events.append(current_event)
                current_event = {
                    'type': change['type'],
                    'start_frame': change['frame'],
                    'end_frame': change['frame'],
                    'confidence': change['confidence'],
                    'changes': [change]
                }
        
        if current_event:
            events.append(current_event)
        
        return events
    
    def _are_related_changes(self, type1: str, type2: str) -> bool:
        """Check if two change types are related."""
        related_groups = [
            {'screen_transition', 'major_update', 'layout_adjustment'},
            {'dialog_open', 'overlay', 'dialog_close'},
            {'sidebar_toggle', 'layout_adjustment'},
            {'content_refresh', 'animation_frame'},
            {'notification', 'overlay'}
        ]
        
        for group in related_groups:
            if type1 in group and type2 in group:
                return True
        
        return type1 == type2
    
    def _detect_ui_patterns(self, events: List[Dict]) -> Dict[str, Any]:
        """Detect patterns in UI behavior."""
        patterns = {
            'modal_usage': 0,
            'navigation_frequency': 0,
            'animation_prevalence': 0,
            'stability_score': 1.0,
            'interaction_patterns': []
        }
        
        if not events:
            return patterns
        
        # Count pattern occurrences
        event_types = [e['type'] for e in events]
        
        patterns['modal_usage'] = sum(1 for t in event_types 
                                     if t in ['dialog_open', 'notification'])
        patterns['navigation_frequency'] = sum(1 for t in event_types 
                                              if t in ['screen_transition', 'major_update'])
        patterns['animation_prevalence'] = sum(1 for t in event_types 
                                              if t == 'animation_frame')
        
        # Stability score (inverse of change frequency)
        total_frames = events[-1]['end_frame'] if events else 1
        change_frames = sum(e['end_frame'] - e['start_frame'] + 1 for e in events)
        patterns['stability_score'] = 1.0 - (change_frames / total_frames)
        
        # Detect interaction patterns
        for i in range(1, len(events)):
            prev_event = events[i-1]
            curr_event = events[i]
            
            # Common interaction patterns
            if prev_event['type'] == 'content_refresh' and curr_event['type'] == 'dialog_open':
                patterns['interaction_patterns'].append({
                    'pattern': 'action_confirmation',
                    'frame': curr_event['start_frame']
                })
            elif prev_event['type'] == 'sidebar_toggle' and curr_event['type'] == 'content_refresh':
                patterns['interaction_patterns'].append({
                    'pattern': 'navigation_response',
                    'frame': curr_event['start_frame']
                })
        
        return patterns
    
    def _summarize_regions(self, regions: List[Dict]) -> List[Dict]:
        """Summarize detected regions for output."""
        summary = []
        
        for frame_regions in regions[:50]:  # Limit output size
            summary.append({
                'frame': frame_regions['frame'],
                'num_regions': frame_regions['num_regions'],
                'types': list(set(r['type'] for r in frame_regions['regions']))
            })
        
        return summary
    
    def _compute_statistics(self, changes: List[Dict], events: List[Dict]) -> Dict:
        """Compute UI change statistics."""
        if not changes:
            return {
                'num_changes': 0,
                'num_events': 0,
                'change_rate': 0.0,
                'dominant_type': 'none',
                'avg_confidence': 0.0
            }
        
        type_counts = {}
        for change in changes:
            type_counts[change['type']] = type_counts.get(change['type'], 0) + 1
        
        return {
            'num_changes': len(changes),
            'num_events': len(events),
            'change_rate': len(changes) / (changes[-1]['frame'] + 1) if changes else 0,
            'dominant_type': max(type_counts.items(), key=lambda x: x[1])[0] if type_counts else 'none',
            'avg_confidence': float(np.mean([c['confidence'] for c in changes])),
            'type_distribution': type_counts
        }
