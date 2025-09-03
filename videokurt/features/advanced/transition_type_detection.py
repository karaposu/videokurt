"""Transition type detection using advanced temporal and spatial analysis."""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from scipy import signal, stats
from scipy.ndimage import gaussian_filter, morphology
from scipy.interpolate import interp1d
import cv2

from ..base import AdvancedFeature


class TransitionTypeDetection(AdvancedFeature):
    """Detect and classify video transitions with detailed characteristics."""
    
    FEATURE_NAME = 'transition_type_detection'
    REQUIRED_ANALYSES = ['frame_diff', 'edge_canny', 'color_histogram', 'optical_flow_dense']
    
    def __init__(self, 
                 transition_threshold: float = 0.25,
                 temporal_window: int = 15,
                 spatial_resolution: int = 16,
                 fade_smoothness_threshold: float = 0.8,
                 wipe_directional_threshold: float = 0.7,
                 morph_kernel_size: int = 3):
        """
        Args:
            transition_threshold: Threshold for detecting transitions
            temporal_window: Window size for temporal analysis
            spatial_resolution: Grid resolution for spatial analysis
            fade_smoothness_threshold: Smoothness threshold for fade detection
            wipe_directional_threshold: Directionality threshold for wipes
            morph_kernel_size: Kernel size for morphological operations
        """
        super().__init__()
        self.transition_threshold = transition_threshold
        self.temporal_window = temporal_window
        self.spatial_resolution = spatial_resolution
        self.fade_smoothness_threshold = fade_smoothness_threshold
        self.wipe_directional_threshold = wipe_directional_threshold
        self.morph_kernel_size = morph_kernel_size
    
    def _compute_advanced(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect and classify transitions using comprehensive analysis.
        
        Returns:
            Dict with transition events, types, and characteristics
        """
        frame_diffs = analysis_data['frame_diff'].data['pixel_diff']
        edge_maps = analysis_data['edge_canny'].data['edge_map']
        color_hists = analysis_data['color_histogram'].data['histogram']
        flow_field = analysis_data['optical_flow_dense'].data['flow_field']
        
        if len(frame_diffs) == 0:
            return self._empty_result()
        
        # Detect transition candidates
        candidates = self._detect_transition_candidates(
            frame_diffs, edge_maps, color_hists
        )
        
        # Analyze each candidate in detail
        transitions = []
        for candidate in candidates:
            transition = self._analyze_transition(
                candidate, frame_diffs, edge_maps, color_hists, flow_field
            )
            if transition:
                transitions.append(transition)
        
        # Merge and refine transitions
        refined_transitions = self._refine_transitions(transitions)
        
        # Detect composite transitions
        composite_transitions = self._detect_composite_transitions(refined_transitions)
        
        # Analyze transition quality
        quality_metrics = self._analyze_transition_quality(refined_transitions)
        
        # Detect patterns
        patterns = self._detect_transition_patterns(refined_transitions)
        
        return {
            'transitions': refined_transitions,
            'composite_transitions': composite_transitions,
            'quality_metrics': quality_metrics,
            'patterns': patterns,
            'statistics': self._compute_statistics(refined_transitions)
        }
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result structure."""
        return {
            'transitions': [],
            'composite_transitions': [],
            'quality_metrics': {},
            'patterns': {},
            'statistics': {
                'num_transitions': 0,
                'dominant_type': 'none',
                'avg_duration': 0
            }
        }
    
    def _detect_transition_candidates(self, frame_diffs: List[np.ndarray],
                                     edge_maps: List[np.ndarray],
                                     color_hists: List[np.ndarray]) -> List[Dict]:
        """Detect potential transition points using multiple cues."""
        candidates = []
        
        # Compute change metrics
        diff_metrics = []
        edge_metrics = []
        color_metrics = []
        
        for i in range(len(frame_diffs)):
            # Frame difference metric
            diff_metric = np.mean(frame_diffs[i]) / 255.0
            diff_metrics.append(diff_metric)
            
            # Edge change metric
            if i > 0 and i < len(edge_maps):
                edge_change = np.mean(np.abs(
                    edge_maps[i].astype(float) - edge_maps[i-1].astype(float)
                )) / 255.0
                edge_metrics.append(edge_change)
            else:
                edge_metrics.append(0)
            
            # Color histogram change
            if i > 0 and i < len(color_hists):
                hist_diff = np.sum(np.abs(
                    color_hists[i] - color_hists[i-1]
                )) / np.sum(color_hists[i] + color_hists[i-1] + 1e-10)
                color_metrics.append(hist_diff)
            else:
                color_metrics.append(0)
        
        # Find peaks in combined metrics
        combined_metric = np.array(diff_metrics) + \
                         np.array(edge_metrics) * 0.5 + \
                         np.array(color_metrics) * 0.5
        
        # Smooth to reduce noise
        if len(combined_metric) > 3:
            combined_metric = gaussian_filter(combined_metric, sigma=1.0)
        
        # Find peaks
        peaks, properties = signal.find_peaks(
            combined_metric,
            height=self.transition_threshold,
            distance=5
        )
        
        # Create candidates
        for idx, peak in enumerate(peaks):
            # Determine window
            start = max(0, peak - self.temporal_window // 2)
            end = min(len(frame_diffs), peak + self.temporal_window // 2)
            
            candidates.append({
                'center_frame': peak,
                'start_frame': start,
                'end_frame': end,
                'strength': float(combined_metric[peak]),
                'peak_properties': {
                    'height': float(properties['peak_heights'][idx]) if 'peak_heights' in properties else 0,
                    'width': float(properties.get('widths', [0])[idx]) if 'widths' in properties and idx < len(properties.get('widths', [])) else 0
                }
            })
        
        return candidates
    
    def _analyze_transition(self, candidate: Dict,
                          frame_diffs: List[np.ndarray],
                          edge_maps: List[np.ndarray],
                          color_hists: List[np.ndarray],
                          flow_field: List[np.ndarray]) -> Optional[Dict]:
        """Analyze a transition candidate in detail."""
        start = candidate['start_frame']
        end = candidate['end_frame']
        center = candidate['center_frame']
        
        # Extract windows
        diff_window = frame_diffs[start:end]
        edge_window = edge_maps[start:end] if start < len(edge_maps) else []
        color_window = color_hists[start:end] if start < len(color_hists) else []
        flow_window = flow_field[start:end] if start < len(flow_field) else []
        
        if len(diff_window) < 3:
            return None
        
        # Classify transition type
        trans_type, confidence, characteristics = self._classify_transition_type(
            diff_window, edge_window, color_window, flow_window
        )
        
        # Compute duration
        actual_start, actual_end = self._find_transition_boundaries(
            diff_window, start
        )
        
        transition = {
            'type': trans_type,
            'start_frame': actual_start,
            'end_frame': actual_end,
            'center_frame': center,
            'duration': actual_end - actual_start,
            'confidence': confidence,
            'strength': candidate['strength'],
            'characteristics': characteristics
        }
        
        return transition
    
    def _classify_transition_type(self, diff_window: List[np.ndarray],
                                 edge_window: List[np.ndarray],
                                 color_window: List[np.ndarray],
                                 flow_window: List[np.ndarray]) -> Tuple[str, float, Dict]:
        """Classify transition type using multi-modal analysis."""
        # Analyze temporal profile
        temporal_profile = [np.mean(d) for d in diff_window]
        
        # Check for cut (instantaneous change)
        cut_score, cut_frame = self._detect_cut(temporal_profile)
        if cut_score > 0.8:
            return 'cut', cut_score, {'cut_frame': cut_frame, 'sharpness': cut_score}
        
        # Check for fade
        fade_score, fade_type = self._detect_fade(temporal_profile, color_window)
        if fade_score > 0.7:
            return fade_type, fade_score, {'smoothness': fade_score}
        
        # Check for dissolve
        dissolve_score = self._detect_dissolve(diff_window, edge_window)
        if dissolve_score > 0.6:
            return 'dissolve', dissolve_score, {'blend_quality': dissolve_score}
        
        # Check for wipe
        wipe_score, wipe_direction = self._detect_wipe(diff_window, flow_window)
        if wipe_score > 0.6:
            return f'wipe_{wipe_direction}', wipe_score, {
                'direction': wipe_direction,
                'speed': self._compute_wipe_speed(diff_window, wipe_direction)
            }
        
        # Check for special transitions
        special_type, special_score = self._detect_special_transitions(
            diff_window, edge_window, flow_window
        )
        if special_score > 0.5:
            return special_type, special_score, {}
        
        # Default to cut if uncertain
        return 'cut', 0.5, {}
    
    def _detect_cut(self, temporal_profile: List[float]) -> Tuple[float, int]:
        """Detect hard cut transition."""
        if len(temporal_profile) < 2:
            return 0.0, 0
        
        # Find sharpest change
        changes = np.diff(temporal_profile)
        if len(changes) == 0:
            return 0.0, 0
        
        max_change_idx = np.argmax(np.abs(changes))
        max_change = abs(changes[max_change_idx])
        
        # Check if change is sharp (occurs in 1-2 frames)
        if max_change > np.mean(temporal_profile) * 2:
            # Check surrounding frames
            before = temporal_profile[max(0, max_change_idx-2):max_change_idx]
            after = temporal_profile[max_change_idx+2:min(len(temporal_profile), max_change_idx+4)]
            
            if before and after:
                before_avg = np.mean(before)
                after_avg = np.mean(after)
                
                # Low activity before and after = cut
                if before_avg < max_change * 0.2 and after_avg < max_change * 0.2:
                    return float(min(1.0, max_change / np.max(temporal_profile))), max_change_idx
        
        return 0.0, 0
    
    def _detect_fade(self, temporal_profile: List[float],
                    color_window: List[np.ndarray]) -> Tuple[float, str]:
        """Detect fade transition (in/out/through black)."""
        if len(temporal_profile) < 5:
            return 0.0, 'fade'
        
        # Normalize profile
        profile = np.array(temporal_profile)
        if np.max(profile) > 0:
            profile = profile / np.max(profile)
        
        # Fit polynomial to check smoothness
        x = np.arange(len(profile))
        try:
            poly = np.polyfit(x, profile, 3)
            fitted = np.polyval(poly, x)
            
            # Compute fit quality
            residual = np.mean(np.abs(profile - fitted))
            smoothness = 1.0 - residual
            
            if smoothness > self.fade_smoothness_threshold:
                # Determine fade type
                start_val = profile[0]
                end_val = profile[-1]
                mid_val = profile[len(profile)//2]
                
                # Check color to determine if fade to/from black
                if color_window:
                    start_brightness = np.mean(color_window[0])
                    end_brightness = np.mean(color_window[-1])
                    
                    if start_brightness < 0.1 and end_val > start_val:
                        return smoothness, 'fade_in'
                    elif end_brightness < 0.1 and start_val > end_val:
                        return smoothness, 'fade_out'
                    elif mid_val < start_val * 0.5 and mid_val < end_val * 0.5:
                        return smoothness, 'fade_through_black'
                
                # Default fade type based on profile
                if end_val > start_val * 1.5:
                    return smoothness, 'fade_in'
                elif start_val > end_val * 1.5:
                    return smoothness, 'fade_out'
                else:
                    return smoothness * 0.8, 'fade'
        except:
            pass
        
        return 0.0, 'fade'
    
    def _detect_dissolve(self, diff_window: List[np.ndarray],
                        edge_window: List[np.ndarray]) -> float:
        """Detect dissolve/cross-fade transition."""
        if len(diff_window) < 5:
            return 0.0
        
        # Dissolve characterized by gradual blend with overlapping content
        scores = []
        
        for i in range(1, len(diff_window) - 1):
            # Check for double edges (overlapping content)
            if i < len(edge_window) - 1:
                edge_density = np.mean(edge_window[i] > 0)
                prev_density = np.mean(edge_window[i-1] > 0)
                next_density = np.mean(edge_window[i+1] > 0)
                
                # Peak in edge density suggests overlapping content
                if edge_density > prev_density * 1.2 and edge_density > next_density * 1.2:
                    scores.append(1.0)
                else:
                    scores.append(0.0)
            
            # Check for transparency-like blending
            diff = diff_window[i]
            h, w = diff.shape
            
            # Sample patches
            patch_vars = []
            patch_size = min(h, w) // 4
            for y in range(0, h - patch_size, patch_size):
                for x in range(0, w - patch_size, patch_size):
                    patch = diff[y:y+patch_size, x:x+patch_size]
                    patch_vars.append(np.var(patch))
            
            # High variance across patches suggests blending
            if patch_vars:
                blend_score = np.mean(patch_vars) / (np.mean(diff) + 1e-6)
                scores.append(min(1.0, blend_score / 100))
        
        return float(np.mean(scores)) if scores else 0.0
    
    def _detect_wipe(self, diff_window: List[np.ndarray],
                    flow_window: List[np.ndarray]) -> Tuple[float, str]:
        """Detect wipe transition with direction."""
        if len(diff_window) < 3:
            return 0.0, 'unknown'
        
        directions = []
        scores = []
        
        for i, diff in enumerate(diff_window):
            h, w = diff.shape
            
            # Compute directional gradients
            h_gradient = np.mean(diff, axis=0)  # Horizontal profile
            v_gradient = np.mean(diff, axis=1)  # Vertical profile
            
            # Detect edge of wipe
            h_edge = self._detect_wipe_edge(h_gradient)
            v_edge = self._detect_wipe_edge(v_gradient)
            
            if h_edge['score'] > v_edge['score']:
                # Horizontal wipe
                if i > 0:
                    prev_diff = diff_window[i-1]
                    prev_gradient = np.mean(prev_diff, axis=0)
                    prev_edge = self._detect_wipe_edge(prev_gradient)
                    
                    if prev_edge['position'] < h_edge['position']:
                        directions.append('left_to_right')
                    else:
                        directions.append('right_to_left')
                    scores.append(h_edge['score'])
            elif v_edge['score'] > 0.5:
                # Vertical wipe
                if i > 0:
                    prev_diff = diff_window[i-1]
                    prev_gradient = np.mean(prev_diff, axis=1)
                    prev_edge = self._detect_wipe_edge(prev_gradient)
                    
                    if prev_edge['position'] < v_edge['position']:
                        directions.append('top_to_bottom')
                    else:
                        directions.append('bottom_to_top')
                    scores.append(v_edge['score'])
            
            # Check for diagonal wipes using flow
            if i < len(flow_window) and flow_window[i].size > 0:
                flow_angle = self._compute_dominant_flow_angle(flow_window[i])
                if abs(flow_angle - np.pi/4) < 0.2:
                    directions.append('diagonal_tl_br')
                    scores.append(0.7)
                elif abs(flow_angle + np.pi/4) < 0.2:
                    directions.append('diagonal_tr_bl')
                    scores.append(0.7)
        
        if scores:
            # Most common direction
            if directions:
                from collections import Counter
                direction_counts = Counter(directions)
                dominant_direction = direction_counts.most_common(1)[0][0]
                return float(np.mean(scores)), dominant_direction
        
        return 0.0, 'unknown'
    
    def _detect_wipe_edge(self, gradient: np.ndarray) -> Dict:
        """Detect edge position in wipe gradient."""
        if len(gradient) < 3:
            return {'score': 0.0, 'position': 0}
        
        # Find steepest part of gradient
        diff = np.diff(gradient)
        if len(diff) == 0:
            return {'score': 0.0, 'position': 0}
        
        max_diff_idx = np.argmax(np.abs(diff))
        max_diff = abs(diff[max_diff_idx])
        
        # Check if it's a clear edge
        if max_diff > np.std(gradient) * 2:
            # Compute sharpness
            window = gradient[max(0, max_diff_idx-2):min(len(gradient), max_diff_idx+3)]
            if len(window) > 0:
                sharpness = max_diff / (np.mean(np.abs(window)) + 1e-6)
                return {
                    'score': min(1.0, sharpness / 5),
                    'position': max_diff_idx
                }
        
        return {'score': 0.0, 'position': 0}
    
    def _compute_dominant_flow_angle(self, flow: np.ndarray) -> float:
        """Compute dominant flow direction angle."""
        if flow.size == 0:
            return 0.0
        
        flow_x = np.mean(flow[..., 0])
        flow_y = np.mean(flow[..., 1])
        return float(np.arctan2(flow_y, flow_x))
    
    def _compute_wipe_speed(self, diff_window: List[np.ndarray],
                           direction: str) -> float:
        """Compute wipe transition speed."""
        positions = []
        
        for diff in diff_window:
            h, w = diff.shape
            
            if 'left' in direction or 'right' in direction:
                gradient = np.mean(diff, axis=0)
                edge = self._detect_wipe_edge(gradient)
                if edge['score'] > 0.5:
                    positions.append(edge['position'] / w)
            elif 'top' in direction or 'bottom' in direction:
                gradient = np.mean(diff, axis=1)
                edge = self._detect_wipe_edge(gradient)
                if edge['score'] > 0.5:
                    positions.append(edge['position'] / h)
        
        if len(positions) > 1:
            # Speed is change in position per frame
            speeds = np.diff(positions)
            return float(np.mean(np.abs(speeds)))
        
        return 0.0
    
    def _detect_special_transitions(self, diff_window: List[np.ndarray],
                                   edge_window: List[np.ndarray],
                                   flow_window: List[np.ndarray]) -> Tuple[str, float]:
        """Detect special transition types."""
        # Iris transition (circular wipe)
        iris_score = self._detect_iris_transition(diff_window)
        if iris_score > 0.6:
            return 'iris', iris_score
        
        # Push transition (sliding)
        push_score = self._detect_push_transition(flow_window)
        if push_score > 0.6:
            return 'push', push_score
        
        # Zoom transition
        zoom_score = self._detect_zoom_transition(flow_window)
        if zoom_score > 0.6:
            return 'zoom', zoom_score
        
        # Morph transition
        morph_score = self._detect_morph_transition(edge_window)
        if morph_score > 0.5:
            return 'morph', morph_score
        
        return 'unknown', 0.0
    
    def _detect_iris_transition(self, diff_window: List[np.ndarray]) -> float:
        """Detect iris (circular) transition."""
        scores = []
        
        for diff in diff_window:
            h, w = diff.shape
            center_y, center_x = h // 2, w // 2
            
            # Create radial profile
            y_coords, x_coords = np.ogrid[:h, :w]
            distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
            max_dist = np.sqrt(center_x**2 + center_y**2)
            
            # Bin by distance
            n_bins = 10
            radial_profile = []
            for i in range(n_bins):
                r_min = i * max_dist / n_bins
                r_max = (i + 1) * max_dist / n_bins
                mask = (distances >= r_min) & (distances < r_max)
                if np.any(mask):
                    radial_profile.append(np.mean(diff[mask]))
            
            if len(radial_profile) > 5:
                # Check for circular pattern
                gradient = np.gradient(radial_profile)
                if np.max(np.abs(gradient)) > np.mean(radial_profile) * 0.5:
                    scores.append(1.0)
                else:
                    scores.append(0.0)
        
        return float(np.mean(scores)) if scores else 0.0
    
    def _detect_push_transition(self, flow_window: List[np.ndarray]) -> float:
        """Detect push/slide transition."""
        if len(flow_window) < 3:
            return 0.0
        
        scores = []
        for flow in flow_window:
            if flow.size == 0:
                continue
            
            # Check for uniform motion
            flow_x = flow[..., 0]
            flow_y = flow[..., 1]
            
            # Compute consistency
            std_x = np.std(flow_x)
            std_y = np.std(flow_y)
            mean_x = np.mean(np.abs(flow_x))
            mean_y = np.mean(np.abs(flow_y))
            
            if mean_x > 0 or mean_y > 0:
                consistency = 1.0 - (std_x + std_y) / (mean_x + mean_y + 1e-6)
                
                # High consistency + high magnitude = push
                magnitude = np.sqrt(mean_x**2 + mean_y**2)
                if magnitude > 2.0 and consistency > 0.7:
                    scores.append(consistency)
        
        return float(np.mean(scores)) if scores else 0.0
    
    def _detect_zoom_transition(self, flow_window: List[np.ndarray]) -> float:
        """Detect zoom transition."""
        if len(flow_window) < 3:
            return 0.0
        
        scores = []
        for flow in flow_window:
            if flow.size == 0:
                continue
            
            h, w = flow.shape[:2]
            center_x, center_y = w // 2, h // 2
            
            # Check for radial flow pattern
            radial_scores = []
            for y in range(0, h, h//8):
                for x in range(0, w, w//8):
                    dx = x - center_x
                    dy = y - center_y
                    r = np.sqrt(dx**2 + dy**2)
                    
                    if r > 10:
                        # Radial unit vector
                        rx = dx / r
                        ry = dy / r
                        
                        # Check if flow aligns with radial direction
                        flow_mag = np.sqrt(flow[y, x, 0]**2 + flow[y, x, 1]**2)
                        if flow_mag > 0:
                            radial_component = (flow[y, x, 0] * rx + flow[y, x, 1] * ry) / flow_mag
                            radial_scores.append(abs(radial_component))
            
            if radial_scores:
                scores.append(np.mean(radial_scores))
        
        return float(np.mean(scores)) if scores else 0.0
    
    def _detect_morph_transition(self, edge_window: List[np.ndarray]) -> float:
        """Detect morph transition."""
        if len(edge_window) < 5:
            return 0.0
        
        # Morph characterized by gradual edge transformation
        edge_changes = []
        for i in range(1, len(edge_window)):
            if edge_window[i].size > 0 and edge_window[i-1].size > 0:
                # Compute edge similarity
                intersection = np.logical_and(edge_window[i], edge_window[i-1])
                union = np.logical_or(edge_window[i], edge_window[i-1])
                
                if np.sum(union) > 0:
                    similarity = np.sum(intersection) / np.sum(union)
                    edge_changes.append(1.0 - similarity)
        
        if edge_changes:
            # Gradual change = morph
            if np.std(edge_changes) < 0.2 and np.mean(edge_changes) > 0.1:
                return float(1.0 - np.std(edge_changes))
        
        return 0.0
    
    def _find_transition_boundaries(self, diff_window: List[np.ndarray],
                                   global_start: int) -> Tuple[int, int]:
        """Find actual transition boundaries."""
        if not diff_window:
            return global_start, global_start
        
        # Compute activity profile
        activity = [np.mean(d) for d in diff_window]
        
        if not activity:
            return global_start, global_start
        
        # Find significant activity region
        threshold = np.max(activity) * 0.2
        
        start_idx = 0
        for i, a in enumerate(activity):
            if a > threshold:
                start_idx = i
                break
        
        end_idx = len(activity) - 1
        for i in range(len(activity) - 1, -1, -1):
            if activity[i] > threshold:
                end_idx = i
                break
        
        return global_start + start_idx, global_start + end_idx
    
    def _refine_transitions(self, transitions: List[Dict]) -> List[Dict]:
        """Refine and merge overlapping transitions."""
        if not transitions:
            return []
        
        # Sort by start frame
        sorted_trans = sorted(transitions, key=lambda x: x['start_frame'])
        
        refined = []
        current = sorted_trans[0]
        
        for trans in sorted_trans[1:]:
            # Check for overlap
            if trans['start_frame'] <= current['end_frame'] + 5:
                # Merge transitions
                if trans['confidence'] > current['confidence']:
                    # Keep better transition but extend timeframe
                    trans['start_frame'] = min(trans['start_frame'], current['start_frame'])
                    trans['end_frame'] = max(trans['end_frame'], current['end_frame'])
                    current = trans
                else:
                    # Extend current transition
                    current['end_frame'] = max(current['end_frame'], trans['end_frame'])
            else:
                refined.append(current)
                current = trans
        
        refined.append(current)
        
        return refined
    
    def _detect_composite_transitions(self, transitions: List[Dict]) -> List[Dict]:
        """Detect composite/compound transitions."""
        composites = []
        
        for i in range(len(transitions) - 1):
            trans1 = transitions[i]
            trans2 = transitions[i + 1]
            
            # Check if transitions are close and different types
            gap = trans2['start_frame'] - trans1['end_frame']
            
            if gap < 10 and trans1['type'] != trans2['type']:
                # Possible composite transition
                composite = {
                    'type': f"{trans1['type']}+{trans2['type']}",
                    'start_frame': trans1['start_frame'],
                    'end_frame': trans2['end_frame'],
                    'components': [trans1, trans2],
                    'confidence': min(trans1['confidence'], trans2['confidence'])
                }
                composites.append(composite)
        
        return composites
    
    def _analyze_transition_quality(self, transitions: List[Dict]) -> Dict:
        """Analyze quality metrics of transitions."""
        if not transitions:
            return {
                'avg_smoothness': 0,
                'avg_duration': 0,
                'consistency': 0
            }
        
        smoothness_scores = []
        durations = []
        
        for trans in transitions:
            # Smoothness based on type and confidence
            if trans['type'] in ['fade_in', 'fade_out', 'dissolve']:
                smoothness = trans['confidence']
            elif trans['type'] == 'cut':
                smoothness = 1.0 - trans['confidence'] * 0.5  # Cuts are less smooth
            else:
                smoothness = 0.5
            
            smoothness_scores.append(smoothness)
            durations.append(trans['duration'])
        
        # Consistency (similarity of transition types)
        type_counts = {}
        for trans in transitions:
            type_counts[trans['type']] = type_counts.get(trans['type'], 0) + 1
        
        consistency = max(type_counts.values()) / len(transitions) if transitions else 0
        
        return {
            'avg_smoothness': float(np.mean(smoothness_scores)),
            'avg_duration': float(np.mean(durations)),
            'consistency': float(consistency),
            'duration_variance': float(np.var(durations))
        }
    
    def _detect_transition_patterns(self, transitions: List[Dict]) -> Dict:
        """Detect patterns in transition usage."""
        patterns = {
            'has_rhythm': False,
            'has_style_consistency': False,
            'transition_frequency': 0,
            'dominant_style': 'none'
        }
        
        if not transitions:
            return patterns
        
        # Check for rhythmic patterns (regular intervals)
        if len(transitions) > 2:
            intervals = []
            for i in range(1, len(transitions)):
                interval = transitions[i]['start_frame'] - transitions[i-1]['end_frame']
                intervals.append(interval)
            
            if intervals:
                # Low variance in intervals = rhythmic
                interval_var = np.var(intervals)
                interval_mean = np.mean(intervals)
                if interval_mean > 0:
                    patterns['has_rhythm'] = interval_var / interval_mean < 0.5
        
        # Style consistency
        type_groups = {}
        for trans in transitions:
            base_type = trans['type'].split('_')[0]  # Group by base type
            type_groups[base_type] = type_groups.get(base_type, 0) + 1
        
        if type_groups:
            dominant = max(type_groups.items(), key=lambda x: x[1])
            patterns['dominant_style'] = dominant[0]
            patterns['has_style_consistency'] = dominant[1] / len(transitions) > 0.6
        
        # Transition frequency
        if transitions:
            total_frames = transitions[-1]['end_frame'] - transitions[0]['start_frame']
            if total_frames > 0:
                patterns['transition_frequency'] = len(transitions) / total_frames
        
        return patterns
    
    def _compute_statistics(self, transitions: List[Dict]) -> Dict:
        """Compute transition statistics."""
        if not transitions:
            return {
                'num_transitions': 0,
                'dominant_type': 'none',
                'avg_duration': 0,
                'type_distribution': {}
            }
        
        # Type distribution
        type_counts = {}
        total_duration = 0
        
        for trans in transitions:
            type_counts[trans['type']] = type_counts.get(trans['type'], 0) + 1
            total_duration += trans['duration']
        
        dominant = max(type_counts.items(), key=lambda x: x[1])[0]
        
        return {
            'num_transitions': len(transitions),
            'dominant_type': dominant,
            'avg_duration': total_duration / len(transitions),
            'type_distribution': type_counts,
            'total_transition_frames': total_duration
        }