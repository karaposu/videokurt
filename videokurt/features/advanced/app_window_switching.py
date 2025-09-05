"""App/window switching detection for screen recordings and videos."""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from scipy import signal, stats
from scipy.spatial.distance import cosine, euclidean
from scipy.ndimage import label, gaussian_filter
import cv2

from ..base import BaseFeature


class AppWindowSwitching(BaseFeature):
    """Detect app/window switching using visual signatures and transition patterns."""
    
    FEATURE_NAME = 'app_window_switching'
    REQUIRED_ANALYSES = ['frame_diff', 'edge_canny', 'color_histogram', 'dct_transform']
    
    def __init__(self, 
                 switch_threshold: float = 0.4,
                 signature_threshold: float = 0.35,
                 transition_duration: int = 15,
                 min_stability_frames: int = 10,
                 signature_components: int = 64,
                 use_perceptual_hash: bool = True):
        """
        Args:
            switch_threshold: Threshold for detecting window switch
            signature_threshold: Threshold for signature dissimilarity
            transition_duration: Max frames for a transition
            min_stability_frames: Min frames to consider stable state
            signature_components: Number of DCT components for signature
            use_perceptual_hash: Use perceptual hashing for comparison
        """
        super().__init__()
        self.switch_threshold = switch_threshold
        self.signature_threshold = signature_threshold
        self.transition_duration = transition_duration
        self.min_stability_frames = min_stability_frames
        self.signature_components = signature_components
        self.use_perceptual_hash = use_perceptual_hash
    
    def compute(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect app/window switching using multiple visual cues.
        
        Returns:
            Dict with switch events, app signatures, and statistics
        """
        frame_diffs = analysis_data['frame_diff'].data['pixel_diff']
        edge_maps = analysis_data['edge_canny'].data['edge_map']
        color_hists = analysis_data['color_histogram'].data['histograms']
        dct_coeffs = analysis_data['dct_transform'].data['dct_coefficients']
        
        if len(edge_maps) == 0:
            return self._empty_result()
        
        # Compute visual signatures for each frame
        signatures = self._compute_visual_signatures(
            edge_maps, color_hists, dct_coeffs
        )
        
        # Detect transition points
        transitions = self._detect_transitions(frame_diffs, signatures)
        
        # Identify stable states between transitions
        stable_states = self._identify_stable_states(signatures, transitions)
        
        # Classify switch types
        switch_events = self._classify_switches(
            transitions, stable_states, signatures, frame_diffs
        )
        
        # Extract app/window signatures
        app_signatures = self._extract_app_signatures(stable_states, signatures)
        
        # Detect patterns
        patterns = self._detect_switch_patterns(switch_events)
        
        return {
            'switch_events': switch_events,
            'stable_states': stable_states,
            'app_signatures': app_signatures,
            'patterns': patterns,
            'statistics': self._compute_statistics(switch_events, stable_states)
        }
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result structure."""
        return {
            'switch_events': [],
            'stable_states': [],
            'app_signatures': [],
            'patterns': {},
            'statistics': {
                'num_switches': 0,
                'num_apps': 0,
                'avg_app_duration': 0.0,
                'switch_frequency': 0.0
            }
        }
    
    def _compute_visual_signatures(self, edge_maps: List[np.ndarray],
                                  color_hists: List[np.ndarray],
                                  dct_coeffs: List[np.ndarray]) -> List[np.ndarray]:
        """Compute comprehensive visual signatures for each frame."""
        signatures = []
        
        for i in range(len(edge_maps)):
            # Edge-based signature
            edge_sig = self._compute_edge_signature(edge_maps[i])
            
            # Color distribution signature
            color_sig = self._compute_color_signature(
                color_hists[i] if i < len(color_hists) else None
            )
            
            # Frequency domain signature
            dct_sig = self._compute_dct_signature(
                dct_coeffs[i] if i < len(dct_coeffs) else None
            )
            
            # Perceptual hash if enabled
            if self.use_perceptual_hash:
                phash = self._compute_perceptual_hash(edge_maps[i])
            else:
                phash = np.zeros(64)
            
            # Combine signatures
            combined = np.concatenate([
                edge_sig,
                color_sig,
                dct_sig,
                phash
            ])
            
            signatures.append(combined)
        
        return signatures
    
    def _compute_edge_signature(self, edges: np.ndarray) -> np.ndarray:
        """Compute edge-based structural signature."""
        h, w = edges.shape
        
        # Grid-based edge density (4x4 grid)
        grid_size = 4
        grid_features = []
        
        for i in range(grid_size):
            for j in range(grid_size):
                region = edges[
                    i*h//grid_size:(i+1)*h//grid_size,
                    j*w//grid_size:(j+1)*w//grid_size
                ]
                grid_features.append(np.mean(region > 0))
        
        # Edge orientation histogram
        # Compute gradients
        gx = cv2.Sobel(edges.astype(float), cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(edges.astype(float), cv2.CV_64F, 0, 1, ksize=3)
        
        # Compute angles
        angles = np.arctan2(gy, gx)
        magnitudes = np.sqrt(gx**2 + gy**2)
        
        # Weighted orientation histogram (8 bins)
        hist, _ = np.histogram(angles, bins=8, range=(-np.pi, np.pi),
                              weights=magnitudes)
        hist = hist / (np.sum(hist) + 1e-10)
        
        # Line detection features
        line_features = self._detect_line_features(edges)
        
        return np.concatenate([grid_features, hist, line_features])
    
    def _detect_line_features(self, edges: np.ndarray) -> np.ndarray:
        """Detect line-based features for UI structure."""
        h, w = edges.shape
        
        # Horizontal and vertical projections
        h_proj = np.sum(edges > 0, axis=1) / w
        v_proj = np.sum(edges > 0, axis=0) / h
        
        # Find dominant lines
        h_peaks = signal.find_peaks(h_proj, height=0.1)[0]
        v_peaks = signal.find_peaks(v_proj, height=0.1)[0]
        
        features = [
            len(h_peaks) / h,  # Horizontal line density
            len(v_peaks) / w,  # Vertical line density
            np.std(np.diff(h_peaks)) if len(h_peaks) > 1 else 0,  # H-line regularity
            np.std(np.diff(v_peaks)) if len(v_peaks) > 1 else 0,  # V-line regularity
        ]
        
        return np.array(features)
    
    def _compute_color_signature(self, color_hist: Optional[np.ndarray]) -> np.ndarray:
        """Compute color distribution signature."""
        if color_hist is None:
            return np.zeros(16)
        
        # Flatten and normalize
        hist = color_hist.flatten()
        hist = hist / (np.sum(hist) + 1e-10)
        
        # Reduce dimensionality by binning
        n_bins = 16
        reduced = np.zeros(n_bins)
        bin_size = len(hist) // n_bins
        
        for i in range(n_bins):
            start = i * bin_size
            end = start + bin_size if i < n_bins - 1 else len(hist)
            reduced[i] = np.sum(hist[start:end])
        
        # Add color statistics
        dominant_colors = signal.find_peaks(hist, height=0.05)[0]
        color_stats = [
            len(dominant_colors) / len(hist),  # Color diversity
            np.max(hist),  # Peak color strength
            stats.entropy(hist + 1e-10),  # Color entropy
        ]
        
        return np.concatenate([reduced, color_stats])
    
    def _compute_dct_signature(self, dct_coeffs: Optional[np.ndarray]) -> np.ndarray:
        """Compute DCT-based frequency signature."""
        if dct_coeffs is None:
            return np.zeros(self.signature_components)
        
        # Take top-left coefficients (low frequencies)
        size = int(np.sqrt(self.signature_components))
        if len(dct_coeffs.shape) == 2:
            sig = dct_coeffs[:size, :size].flatten()
        else:
            sig = dct_coeffs[:self.signature_components]
        
        # Normalize
        sig = sig / (np.linalg.norm(sig) + 1e-10)
        
        # Pad if necessary
        if len(sig) < self.signature_components:
            sig = np.pad(sig, (0, self.signature_components - len(sig)))
        
        return sig[:self.signature_components]
    
    def _compute_perceptual_hash(self, image: np.ndarray) -> np.ndarray:
        """Compute perceptual hash for robust comparison."""
        # Resize to 32x32
        resized = cv2.resize(image.astype(float), (32, 32))
        
        # Compute DCT
        dct = cv2.dct(resized)
        
        # Take 8x8 low frequency components
        dct_low = dct[:8, :8]
        
        # Compute median
        median = np.median(dct_low)
        
        # Generate binary hash
        hash_bits = (dct_low > median).flatten()
        
        return hash_bits.astype(float)
    
    def _detect_transitions(self, frame_diffs: List[np.ndarray],
                           signatures: List[np.ndarray]) -> List[Dict]:
        """Detect transition points between different visual states."""
        transitions = []
        
        # Compute signature distances
        sig_distances = []
        for i in range(1, len(signatures)):
            dist = 1.0 - cosine(signatures[i], signatures[i-1])
            sig_distances.append(dist)
        
        # Smooth distances
        if len(sig_distances) > 3:
            sig_distances = gaussian_filter(sig_distances, sigma=1.0)
        
        # Find peaks in signature distance
        if len(sig_distances) > 0:
            peaks, properties = signal.find_peaks(
                sig_distances, 
                height=self.signature_threshold,
                distance=self.min_stability_frames
            )
            
            for peak_idx, peak in enumerate(peaks):
                frame_idx = peak + 1  # Adjust for offset
                
                # Analyze transition characteristics
                if frame_idx - 1 < len(frame_diffs):
                    pixel_change = np.mean(frame_diffs[frame_idx - 1]) / 255.0
                else:
                    pixel_change = 0
                
                # Determine transition type
                trans_type = self._classify_transition(
                    pixel_change,
                    sig_distances[peak],
                    frame_idx,
                    frame_diffs
                )
                
                transitions.append({
                    'frame': frame_idx,
                    'type': trans_type,
                    'signature_distance': float(sig_distances[peak]),
                    'pixel_change': float(pixel_change),
                    'confidence': float(properties['peak_heights'][peak_idx])
                })
        
        return transitions
    
    def _classify_transition(self, pixel_change: float, sig_dist: float,
                            frame: int, diffs: List[np.ndarray]) -> str:
        """Classify the type of transition."""
        # Check for fade/dissolve
        if frame >= 2 and frame < len(diffs) + 1:
            prev_changes = [np.mean(diffs[max(0, frame-3+i)]) / 255.0 
                          for i in range(min(3, len(diffs)))]
            
            if all(c > 0.1 for c in prev_changes):
                if np.std(prev_changes) < 0.1:
                    return 'fade'
                else:
                    return 'dissolve'
        
        # Check for hard cut
        if pixel_change > 0.7 and sig_dist > 0.5:
            return 'hard_cut'
        
        # Check for slide/push
        if pixel_change > 0.3 and pixel_change < 0.7:
            return 'slide'
        
        # Default
        return 'soft_transition'
    
    def _identify_stable_states(self, signatures: List[np.ndarray],
                               transitions: List[Dict]) -> List[Dict]:
        """Identify stable visual states between transitions."""
        stable_states = []
        
        # Add boundaries
        transition_frames = [0] + [t['frame'] for t in transitions] + [len(signatures)]
        
        for i in range(len(transition_frames) - 1):
            start = transition_frames[i]
            end = transition_frames[i + 1]
            
            # Skip if too short
            if end - start < self.min_stability_frames:
                continue
            
            # Compute state signature
            state_sigs = signatures[start:end]
            mean_sig = np.mean(state_sigs, axis=0)
            
            # Compute stability (low variance)
            if len(state_sigs) > 1:
                sig_vars = [np.var(s) for s in state_sigs]
                stability = 1.0 / (1.0 + np.mean(sig_vars))
            else:
                stability = 0.5
            
            stable_states.append({
                'start_frame': start,
                'end_frame': end - 1,
                'duration': end - start,
                'signature': mean_sig,
                'stability': float(stability),
                'state_id': i
            })
        
        return stable_states
    
    def _classify_switches(self, transitions: List[Dict], states: List[Dict],
                          signatures: List[np.ndarray], 
                          diffs: List[np.ndarray]) -> List[Dict]:
        """Classify switch events based on transitions and states."""
        switches = []
        
        for i, trans in enumerate(transitions):
            # Find surrounding states
            prev_state = None
            next_state = None
            
            for state in states:
                if state['end_frame'] < trans['frame']:
                    prev_state = state
                elif state['start_frame'] >= trans['frame']:
                    next_state = state
                    break
            
            if prev_state and next_state:
                # Compare state signatures
                sig_similarity = 1.0 - cosine(
                    prev_state['signature'],
                    next_state['signature']
                )
                
                # Determine switch type
                switch_type = self._determine_switch_type(
                    trans, sig_similarity, prev_state, next_state
                )
                
                switches.append({
                    'frame': trans['frame'],
                    'type': switch_type,
                    'transition_type': trans['type'],
                    'from_state': prev_state['state_id'],
                    'to_state': next_state['state_id'],
                    'confidence': trans['confidence'],
                    'signature_change': 1.0 - sig_similarity
                })
        
        return switches
    
    def _determine_switch_type(self, transition: Dict, similarity: float,
                              prev_state: Dict, next_state: Dict) -> str:
        """Determine the type of app/window switch."""
        # Complete app change
        if similarity < 0.3:
            if transition['type'] == 'hard_cut':
                return 'app_switch'
            else:
                return 'app_transition'
        
        # Window change within same app
        elif similarity < 0.6:
            return 'window_switch'
        
        # Tab/view change
        elif similarity < 0.8:
            return 'tab_switch'
        
        # Minor update
        else:
            return 'view_update'
    
    def _extract_app_signatures(self, states: List[Dict],
                               signatures: List[np.ndarray]) -> List[Dict]:
        """Extract unique app/window signatures."""
        if not states:
            return []
        
        # Cluster similar states
        app_signatures = []
        processed = set()
        
        for i, state in enumerate(states):
            if i in processed:
                continue
            
            # Find similar states
            similar_states = [i]
            for j, other_state in enumerate(states[i+1:], i+1):
                if j not in processed:
                    similarity = 1.0 - cosine(
                        state['signature'],
                        other_state['signature']
                    )
                    if similarity > 0.7:
                        similar_states.append(j)
                        processed.add(j)
            
            # Create app signature
            app_states = [states[idx] for idx in similar_states]
            combined_sig = np.mean([s['signature'] for s in app_states], axis=0)
            
            app_signatures.append({
                'app_id': len(app_signatures),
                'signature': combined_sig,
                'state_indices': similar_states,
                'total_frames': sum(s['duration'] for s in app_states),
                'occurrences': len(similar_states)
            })
        
        return app_signatures
    
    def _detect_switch_patterns(self, switches: List[Dict]) -> Dict[str, Any]:
        """Detect patterns in switching behavior."""
        patterns = {
            'switch_frequency': 0,
            'dominant_switch_type': 'none',
            'rapid_switching': False,
            'cycling_pattern': False,
            'switch_intervals': []
        }
        
        if not switches:
            return patterns
        
        # Switch frequency
        total_frames = switches[-1]['frame'] if switches else 1
        patterns['switch_frequency'] = len(switches) / total_frames
        
        # Dominant type
        type_counts = {}
        for switch in switches:
            type_counts[switch['type']] = type_counts.get(switch['type'], 0) + 1
        patterns['dominant_switch_type'] = max(type_counts.items(), 
                                              key=lambda x: x[1])[0]
        
        # Switch intervals
        intervals = []
        for i in range(1, len(switches)):
            intervals.append(switches[i]['frame'] - switches[i-1]['frame'])
        patterns['switch_intervals'] = intervals
        
        # Rapid switching detection
        if intervals:
            patterns['rapid_switching'] = np.mean(intervals) < 30
        
        # Cycling pattern detection
        if len(switches) >= 4:
            # Check if returning to same states
            state_sequence = [s['to_state'] for s in switches]
            patterns['cycling_pattern'] = self._detect_cycling(state_sequence)
        
        return patterns
    
    def _detect_cycling(self, sequence: List[int]) -> bool:
        """Detect if there's a cycling pattern in state transitions."""
        if len(sequence) < 4:
            return False
        
        # Look for repeated subsequences
        for cycle_len in range(2, len(sequence) // 2 + 1):
            for start in range(len(sequence) - cycle_len * 2 + 1):
                pattern = sequence[start:start + cycle_len]
                next_pattern = sequence[start + cycle_len:start + cycle_len * 2]
                
                if pattern == next_pattern:
                    return True
        
        return False
    
    def _compute_statistics(self, switches: List[Dict], 
                          states: List[Dict]) -> Dict:
        """Compute switching statistics."""
        if not switches:
            return {
                'num_switches': 0,
                'num_apps': 0,
                'avg_app_duration': 0.0,
                'switch_frequency': 0.0,
                'avg_stability': 0.0
            }
        
        # Count unique apps/windows
        unique_states = set()
        for switch in switches:
            unique_states.add(switch['from_state'])
            unique_states.add(switch['to_state'])
        
        # Average duration
        avg_duration = np.mean([s['duration'] for s in states]) if states else 0
        
        # Average stability
        avg_stability = np.mean([s['stability'] for s in states]) if states else 0
        
        return {
            'num_switches': len(switches),
            'num_apps': len(unique_states),
            'avg_app_duration': float(avg_duration),
            'switch_frequency': len(switches) / (switches[-1]['frame'] + 1),
            'avg_stability': float(avg_stability)
        }