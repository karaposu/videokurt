"""Repetitive pattern classification using advanced signal processing and pattern recognition."""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from scipy import signal, stats
from scipy.fft import fft, fftfreq
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import euclidean, cosine
import warnings

from ..base import BaseFeature


class RepetitivePatternClassification(BaseFeature):
    """Classify and analyze repetitive visual patterns using multi-modal analysis."""
    
    FEATURE_NAME = 'repetitive_pattern_classification'
    REQUIRED_ANALYSES = ['frequency_fft', 'frame_diff', 'optical_flow_dense', 
                         'edge_canny', 'color_histogram']
    
    def __init__(self, 
                 min_repetitions: int = 3,
                 frequency_resolution: int = 256,
                 temporal_window: int = 120,
                 spatial_grid_size: int = 8,
                 similarity_threshold: float = 0.7,
                 use_phase_analysis: bool = True,
                 pattern_memory: int = 10):
        """
        Args:
            min_repetitions: Minimum repetitions to consider pattern
            frequency_resolution: FFT resolution for frequency analysis
            temporal_window: Window size for temporal pattern analysis
            spatial_grid_size: Grid size for spatial pattern analysis
            similarity_threshold: Threshold for pattern similarity
            use_phase_analysis: Use phase information in frequency analysis
            pattern_memory: Number of patterns to remember for matching
        """
        super().__init__()
        self.min_repetitions = min_repetitions
        self.frequency_resolution = frequency_resolution
        self.temporal_window = temporal_window
        self.spatial_grid_size = spatial_grid_size
        self.similarity_threshold = similarity_threshold
        self.use_phase_analysis = use_phase_analysis
        self.pattern_memory = pattern_memory
    
    def compute(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Classify repetitive patterns using comprehensive analysis.
        
        Returns:
            Dict with pattern classifications, periodicities, and characteristics
        """
        fft_data = analysis_data['frequency_fft'].data
        frame_diffs = analysis_data['frame_diff'].data['pixel_diff']
        flow_field = analysis_data['optical_flow_dense'].data['flow_field']
        edge_maps = analysis_data['edge_canny'].data['edge_map']
        color_hists = analysis_data['color_histogram'].data['histograms']
        
        if len(frame_diffs) == 0:
            return self._empty_result()
        
        # Extract temporal patterns
        temporal_patterns = self._extract_temporal_patterns(
            frame_diffs, flow_field, edge_maps, color_hists
        )
        
        # Analyze frequency domain patterns
        frequency_patterns = self._analyze_frequency_patterns(fft_data)
        
        # Detect spatial repetition patterns
        spatial_patterns = self._detect_spatial_patterns(
            edge_maps, frame_diffs
        )
        
        # Detect motion repetition patterns
        motion_patterns = self._detect_motion_patterns(flow_field)
        
        # Identify specific pattern types
        pattern_classifications = self._classify_patterns(
            temporal_patterns, frequency_patterns, spatial_patterns, motion_patterns
        )
        
        # Detect pattern cycles and phases
        cycles = self._detect_pattern_cycles(temporal_patterns, frequency_patterns)
        
        # Analyze pattern stability
        stability = self._analyze_pattern_stability(
            temporal_patterns, pattern_classifications
        )
        
        return {
            'pattern_classifications': pattern_classifications,
            'temporal_patterns': temporal_patterns,
            'frequency_patterns': frequency_patterns,
            'spatial_patterns': spatial_patterns,
            'motion_patterns': motion_patterns,
            'cycles': cycles,
            'stability': stability,
            'statistics': self._compute_statistics(pattern_classifications)
        }
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result structure."""
        return {
            'pattern_classifications': [],
            'temporal_patterns': {},
            'frequency_patterns': {},
            'spatial_patterns': {},
            'motion_patterns': {},
            'cycles': [],
            'stability': {},
            'statistics': {
                'num_patterns': 0,
                'dominant_type': 'none',
                'avg_period': 0
            }
        }
    
    def _extract_temporal_patterns(self, frame_diffs: List[np.ndarray],
                                  flow_field: List[np.ndarray],
                                  edge_maps: List[np.ndarray],
                                  color_hists: List[np.ndarray]) -> Dict:
        """Extract temporal patterns from multiple modalities."""
        patterns = {
            'activity': [],
            'motion': [],
            'structure': [],
            'color': [],
            'combined': []
        }
        
        # Activity pattern from frame differences
        patterns['activity'] = [np.mean(diff) for diff in frame_diffs]
        
        # Motion pattern from optical flow
        if flow_field:
            patterns['motion'] = []
            for flow in flow_field:
                if flow.size > 0:
                    mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                    patterns['motion'].append(np.mean(mag))
                else:
                    patterns['motion'].append(0)
        
        # Structural pattern from edges
        if edge_maps:
            patterns['structure'] = [np.mean(edges > 0) for edges in edge_maps]
        
        # Color pattern from histograms
        if color_hists:
            patterns['color'] = []
            for hist in color_hists:
                if hist.size > 0:
                    # Compute color centroid
                    hist_norm = hist / (np.sum(hist) + 1e-10)
                    centroid = np.sum(np.arange(len(hist_norm)) * hist_norm)
                    patterns['color'].append(centroid)
                else:
                    patterns['color'].append(0)
        
        # Combined pattern using PCA
        if patterns['activity'] and patterns['motion']:
            combined = np.column_stack([
                patterns['activity'][:min(len(patterns['activity']), len(patterns['motion']))],
                patterns['motion'][:min(len(patterns['activity']), len(patterns['motion']))]
            ])
            patterns['combined'] = np.mean(combined, axis=1).tolist()
        else:
            patterns['combined'] = patterns['activity']
        
        # Analyze each pattern for repetition
        for key in patterns:
            if patterns[key]:
                patterns[key] = {
                    'values': patterns[key],
                    'autocorrelation': self._compute_autocorrelation(patterns[key]),
                    'periodicity': self._detect_periodicity(patterns[key]),
                    'regularity': self._compute_regularity(patterns[key])
                }
        
        return patterns
    
    def _compute_autocorrelation(self, signal_data: List[float]) -> np.ndarray:
        """Compute normalized autocorrelation of signal."""
        if len(signal_data) < 3:
            return np.array([])
        
        signal_array = np.array(signal_data)
        signal_array = signal_array - np.mean(signal_array)
        
        # Compute autocorrelation
        autocorr = np.correlate(signal_array, signal_array, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Normalize
        if autocorr[0] > 0:
            autocorr = autocorr / autocorr[0]
        
        return autocorr[:min(len(autocorr), self.temporal_window)]
    
    def _detect_periodicity(self, signal_data: List[float]) -> Dict:
        """Detect periodicity in signal using multiple methods."""
        if len(signal_data) < self.min_repetitions * 2:
            return {'has_period': False, 'period': 0, 'confidence': 0}
        
        signal_array = np.array(signal_data)
        
        # Method 1: Autocorrelation peaks
        autocorr = self._compute_autocorrelation(signal_data)
        if len(autocorr) > 1:
            peaks, properties = signal.find_peaks(autocorr[1:], height=0.3)
            if len(peaks) > 0:
                period_ac = peaks[0] + 1
                confidence_ac = properties['peak_heights'][0]
            else:
                period_ac = 0
                confidence_ac = 0
        else:
            period_ac = 0
            confidence_ac = 0
        
        # Method 2: FFT
        if len(signal_array) > 4:
            fft_vals = np.fft.fft(signal_array)
            fft_abs = np.abs(fft_vals)[1:len(fft_vals)//2]
            
            if len(fft_abs) > 0 and np.max(fft_abs) > 0:
                dominant_freq_idx = np.argmax(fft_abs)
                period_fft = len(signal_array) / (dominant_freq_idx + 1)
                confidence_fft = fft_abs[dominant_freq_idx] / np.sum(fft_abs)
            else:
                period_fft = 0
                confidence_fft = 0
        else:
            period_fft = 0
            confidence_fft = 0
        
        # Method 3: Peak detection
        peaks, _ = signal.find_peaks(signal_array, prominence=np.std(signal_array) * 0.5)
        if len(peaks) > 1:
            peak_intervals = np.diff(peaks)
            if len(peak_intervals) > 0:
                period_peaks = np.median(peak_intervals)
                confidence_peaks = 1.0 - np.std(peak_intervals) / (np.mean(peak_intervals) + 1e-6)
            else:
                period_peaks = 0
                confidence_peaks = 0
        else:
            period_peaks = 0
            confidence_peaks = 0
        
        # Combine methods
        periods = [period_ac, period_fft, period_peaks]
        confidences = [confidence_ac, confidence_fft, confidence_peaks]
        
        # Select best result
        best_idx = np.argmax(confidences)
        best_period = periods[best_idx]
        best_confidence = confidences[best_idx]
        
        return {
            'has_period': best_confidence > 0.3 and best_period > 0,
            'period': float(best_period),
            'confidence': float(best_confidence),
            'methods': {
                'autocorrelation': {'period': period_ac, 'confidence': confidence_ac},
                'fft': {'period': period_fft, 'confidence': confidence_fft},
                'peaks': {'period': period_peaks, 'confidence': confidence_peaks}
            }
        }
    
    def _compute_regularity(self, signal_data: List[float]) -> float:
        """Compute regularity score of signal."""
        if len(signal_data) < 3:
            return 0.0
        
        signal_array = np.array(signal_data)
        
        # Compute differences
        diffs = np.diff(signal_array)
        if len(diffs) == 0:
            return 1.0
        
        # Regularity based on consistency of changes
        if np.std(diffs) > 0:
            regularity = 1.0 / (1.0 + np.std(diffs) / (np.mean(np.abs(diffs)) + 1e-6))
        else:
            regularity = 1.0
        
        return float(regularity)
    
    def _analyze_frequency_patterns(self, fft_data: Dict) -> Dict:
        """Analyze patterns in frequency domain."""
        patterns = {
            'dominant_frequencies': [],
            'harmonic_structure': {},
            'frequency_stability': 0.0
        }
        
        if 'frequency_spectrum' not in fft_data:
            return patterns
        
        spectrum = fft_data['frequency_spectrum']
        
        if len(spectrum.shape) > 1:
            # Average across spatial dimensions
            avg_spectrum = np.mean(spectrum, axis=tuple(range(1, len(spectrum.shape))))
        else:
            avg_spectrum = spectrum
        
        if len(avg_spectrum) < 2:
            return patterns
        
        # Find dominant frequencies
        freqs = fftfreq(len(avg_spectrum), d=1.0)[:len(avg_spectrum)//2]
        power = np.abs(avg_spectrum[:len(avg_spectrum)//2])
        
        # Normalize power spectrum
        if np.max(power[1:]) > 0:
            power_norm = power[1:] / np.max(power[1:])
        else:
            power_norm = power[1:]
        
        # Find peaks
        peaks, properties = signal.find_peaks(power_norm, height=0.2, distance=5)
        
        for idx, peak in enumerate(peaks):
            if peak > 0:
                patterns['dominant_frequencies'].append({
                    'frequency': float(freqs[peak + 1]),
                    'period': float(1.0 / freqs[peak + 1]) if freqs[peak + 1] > 0 else 0,
                    'power': float(power_norm[peak]),
                    'bandwidth': self._compute_bandwidth(power_norm, peak)
                })
        
        # Detect harmonic structure
        if len(patterns['dominant_frequencies']) > 1:
            patterns['harmonic_structure'] = self._detect_harmonics(
                patterns['dominant_frequencies']
            )
        
        # Frequency stability (concentration of power)
        if len(power_norm) > 0:
            entropy = stats.entropy(power_norm + 1e-10)
            patterns['frequency_stability'] = float(1.0 / (1.0 + entropy))
        
        return patterns
    
    def _compute_bandwidth(self, spectrum: np.ndarray, peak_idx: int) -> float:
        """Compute bandwidth around frequency peak."""
        if peak_idx >= len(spectrum):
            return 0.0
        
        peak_val = spectrum[peak_idx]
        half_power = peak_val / np.sqrt(2)
        
        # Find 3dB points
        left_idx = peak_idx
        right_idx = peak_idx
        
        while left_idx > 0 and spectrum[left_idx] > half_power:
            left_idx -= 1
        
        while right_idx < len(spectrum) - 1 and spectrum[right_idx] > half_power:
            right_idx += 1
        
        bandwidth = (right_idx - left_idx) / len(spectrum)
        return float(bandwidth)
    
    def _detect_harmonics(self, frequencies: List[Dict]) -> Dict:
        """Detect harmonic relationships between frequencies."""
        if len(frequencies) < 2:
            return {}
        
        harmonics = {
            'has_harmonics': False,
            'fundamental': None,
            'harmonic_ratios': []
        }
        
        # Sort by frequency
        sorted_freqs = sorted(frequencies, key=lambda x: x['frequency'])
        
        # Check for harmonic relationships
        fundamental = sorted_freqs[0]
        harmonic_ratios = []
        
        for freq in sorted_freqs[1:]:
            if fundamental['frequency'] > 0:
                ratio = freq['frequency'] / fundamental['frequency']
                
                # Check if ratio is close to integer
                nearest_int = round(ratio)
                if abs(ratio - nearest_int) < 0.1 and nearest_int > 1:
                    harmonic_ratios.append({
                        'frequency': freq['frequency'],
                        'harmonic_number': nearest_int,
                        'deviation': float(abs(ratio - nearest_int))
                    })
        
        if harmonic_ratios:
            harmonics['has_harmonics'] = True
            harmonics['fundamental'] = fundamental
            harmonics['harmonic_ratios'] = harmonic_ratios
        
        return harmonics
    
    def _detect_spatial_patterns(self, edge_maps: List[np.ndarray],
                                frame_diffs: List[np.ndarray]) -> Dict:
        """Detect spatial repetition patterns."""
        patterns = {
            'grid_patterns': [],
            'texture_repetition': [],
            'spatial_periodicity': {}
        }
        
        if not edge_maps:
            return patterns
        
        for idx, edges in enumerate(edge_maps[:min(len(edge_maps), 50)]):  # Limit processing
            if edges.size == 0:
                continue
            
            # Detect grid patterns
            grid_pattern = self._detect_grid_pattern(edges)
            if grid_pattern['has_grid']:
                patterns['grid_patterns'].append({
                    'frame': idx,
                    **grid_pattern
                })
            
            # Detect texture repetition
            if idx < len(frame_diffs):
                texture = self._detect_texture_repetition(frame_diffs[idx])
                if texture['has_repetition']:
                    patterns['texture_repetition'].append({
                        'frame': idx,
                        **texture
                    })
        
        # Analyze spatial periodicity
        if edge_maps:
            patterns['spatial_periodicity'] = self._analyze_spatial_periodicity(
                edge_maps[:min(len(edge_maps), 30)]
            )
        
        return patterns
    
    def _detect_grid_pattern(self, edges: np.ndarray) -> Dict:
        """Detect grid-like patterns in edges."""
        h, w = edges.shape
        
        # Compute projections
        h_projection = np.sum(edges > 0, axis=1)
        v_projection = np.sum(edges > 0, axis=0)
        
        # Find regularly spaced peaks
        h_peaks, _ = signal.find_peaks(h_projection, distance=h//20, height=w*0.1)
        v_peaks, _ = signal.find_peaks(v_projection, distance=w//20, height=h*0.1)
        
        grid_info = {
            'has_grid': False,
            'h_spacing': 0,
            'v_spacing': 0,
            'regularity': 0
        }
        
        # Check for regular spacing
        if len(h_peaks) > 2:
            h_spacings = np.diff(h_peaks)
            h_regularity = 1.0 - np.std(h_spacings) / (np.mean(h_spacings) + 1e-6)
            
            if h_regularity > 0.7:
                grid_info['h_spacing'] = float(np.mean(h_spacings))
                grid_info['regularity'] = h_regularity
        
        if len(v_peaks) > 2:
            v_spacings = np.diff(v_peaks)
            v_regularity = 1.0 - np.std(v_spacings) / (np.mean(v_spacings) + 1e-6)
            
            if v_regularity > 0.7:
                grid_info['v_spacing'] = float(np.mean(v_spacings))
                grid_info['regularity'] = max(grid_info['regularity'], v_regularity)
        
        grid_info['has_grid'] = grid_info['regularity'] > 0.7
        
        return grid_info
    
    def _detect_texture_repetition(self, image: np.ndarray) -> Dict:
        """Detect repetitive texture patterns."""
        if image.size == 0:
            return {'has_repetition': False}
        
        h, w = image.shape[:2] if len(image.shape) >= 2 else (0, 0)
        
        if h < 32 or w < 32:
            return {'has_repetition': False}
        
        # Use autocorrelation for texture analysis
        # Downsample for efficiency
        img_small = image[::4, ::4]
        
        # 2D autocorrelation
        autocorr_2d = signal.correlate2d(img_small, img_small, mode='same')
        
        # Find peaks in autocorrelation
        peaks = signal.find_peaks_2d(autocorr_2d, min_distance=5, threshold_rel=0.5)
        
        if len(peaks[0]) > 4:  # Multiple peaks indicate repetition
            # Analyze peak spacing
            peak_coords = np.column_stack(peaks)
            
            # Compute pairwise distances
            distances = []
            for i in range(len(peak_coords)):
                for j in range(i+1, min(i+5, len(peak_coords))):
                    dist = euclidean(peak_coords[i], peak_coords[j])
                    distances.append(dist)
            
            if distances:
                # Check for regular spacing
                distances = np.array(distances)
                regularity = 1.0 - np.std(distances) / (np.mean(distances) + 1e-6)
                
                if regularity > 0.5:
                    return {
                        'has_repetition': True,
                        'spacing': float(np.mean(distances) * 4),  # Account for downsampling
                        'regularity': float(regularity)
                    }
        
        return {'has_repetition': False}
    
    def _analyze_spatial_periodicity(self, images: List[np.ndarray]) -> Dict:
        """Analyze spatial periodicity across frames."""
        if not images:
            return {}
        
        # Compute spatial frequency for each frame
        spatial_freqs = []
        
        for img in images:
            if img.size == 0:
                continue
            
            # 2D FFT
            fft_2d = np.fft.fft2(img)
            fft_abs = np.abs(fft_2d)
            
            # Radial average
            h, w = fft_abs.shape
            cy, cx = h // 2, w // 2
            
            radial_profile = []
            max_radius = min(cy, cx)
            
            for r in range(1, max_radius, 2):
                y, x = np.ogrid[-cy:h-cy, -cx:w-cx]
                mask = (x**2 + y**2 >= r**2) & (x**2 + y**2 < (r+2)**2)
                radial_profile.append(np.mean(fft_abs[mask]))
            
            if radial_profile:
                spatial_freqs.append(radial_profile)
        
        if spatial_freqs:
            # Average profiles
            max_len = max(len(p) for p in spatial_freqs)
            padded = [p + [0]*(max_len - len(p)) for p in spatial_freqs]
            avg_profile = np.mean(padded, axis=0)
            
            # Find peaks in radial profile
            peaks, _ = signal.find_peaks(avg_profile, prominence=np.std(avg_profile))
            
            if len(peaks) > 0:
                return {
                    'has_spatial_period': True,
                    'dominant_wavelength': float(2 * max_radius / (peaks[0] + 1)),
                    'num_periodic_components': len(peaks)
                }
        
        return {'has_spatial_period': False}
    
    def _detect_motion_patterns(self, flow_field: List[np.ndarray]) -> Dict:
        """Detect repetitive motion patterns."""
        patterns = {
            'oscillatory_motion': [],
            'circular_motion': [],
            'back_forth_motion': []
        }
        
        if len(flow_field) < self.min_repetitions * 2:
            return patterns
        
        # Analyze motion in regions
        for i in range(0, len(flow_field) - self.temporal_window, 10):
            window = flow_field[i:i+self.temporal_window]
            
            if not window or window[0].size == 0:
                continue
            
            h, w = window[0].shape[:2]
            
            # Divide into regions
            region_size = min(h, w) // self.spatial_grid_size
            
            for y in range(0, h - region_size, region_size):
                for x in range(0, w - region_size, region_size):
                    # Extract region motion over time
                    region_flows = []
                    for flow in window:
                        region = flow[y:y+region_size, x:x+region_size]
                        if region.size > 0:
                            mean_flow = np.mean(region, axis=(0, 1))
                            region_flows.append(mean_flow)
                    
                    if len(region_flows) < 3:
                        continue
                    
                    region_flows = np.array(region_flows)
                    
                    # Detect oscillatory motion
                    oscillation = self._detect_oscillatory_motion(region_flows)
                    if oscillation['is_oscillatory']:
                        patterns['oscillatory_motion'].append({
                            'start_frame': i,
                            'region': (x, y, x+region_size, y+region_size),
                            **oscillation
                        })
                    
                    # Detect circular motion
                    circular = self._detect_circular_motion(region_flows)
                    if circular['is_circular']:
                        patterns['circular_motion'].append({
                            'start_frame': i,
                            'region': (x, y, x+region_size, y+region_size),
                            **circular
                        })
                    
                    # Detect back-and-forth motion
                    back_forth = self._detect_back_forth_motion(region_flows)
                    if back_forth['is_back_forth']:
                        patterns['back_forth_motion'].append({
                            'start_frame': i,
                            'region': (x, y, x+region_size, y+region_size),
                            **back_forth
                        })
        
        return patterns
    
    def _detect_oscillatory_motion(self, flows: np.ndarray) -> Dict:
        """Detect oscillatory motion pattern."""
        if len(flows) < 4:
            return {'is_oscillatory': False}
        
        # Compute magnitude and angle over time
        magnitudes = np.linalg.norm(flows, axis=1)
        angles = np.arctan2(flows[:, 1], flows[:, 0])
        
        # Check for periodic magnitude
        mag_periodicity = self._detect_periodicity(magnitudes.tolist())
        
        # Check for alternating angles
        angle_changes = np.diff(angles)
        sign_changes = np.sum(np.diff(np.sign(angle_changes)) != 0)
        
        is_oscillatory = (mag_periodicity['has_period'] and 
                         sign_changes > len(angle_changes) * 0.5)
        
        return {
            'is_oscillatory': is_oscillatory,
            'period': mag_periodicity['period'] if is_oscillatory else 0,
            'amplitude': float(np.std(magnitudes)) if is_oscillatory else 0
        }
    
    def _detect_circular_motion(self, flows: np.ndarray) -> Dict:
        """Detect circular motion pattern."""
        if len(flows) < 8:
            return {'is_circular': False}
        
        # Compute angles
        angles = np.arctan2(flows[:, 1], flows[:, 0])
        
        # Unwrap angles
        angles_unwrapped = np.unwrap(angles)
        
        # Check for monotonic increase/decrease (rotation)
        angle_diff = angles_unwrapped[-1] - angles_unwrapped[0]
        
        # Check if completed at least one rotation
        if abs(angle_diff) > 2 * np.pi:
            # Check for constant angular velocity
            angular_velocities = np.diff(angles_unwrapped)
            velocity_std = np.std(angular_velocities)
            velocity_mean = np.mean(angular_velocities)
            
            if abs(velocity_mean) > 0 and velocity_std / abs(velocity_mean) < 0.3:
                return {
                    'is_circular': True,
                    'angular_velocity': float(velocity_mean),
                    'num_rotations': float(abs(angle_diff) / (2 * np.pi))
                }
        
        return {'is_circular': False}
    
    def _detect_back_forth_motion(self, flows: np.ndarray) -> Dict:
        """Detect back-and-forth motion pattern."""
        if len(flows) < 4:
            return {'is_back_forth': False}
        
        # Project onto dominant direction
        mean_flow = np.mean(flows, axis=0)
        if np.linalg.norm(mean_flow) < 0.1:
            # Try to find dominant direction from variance
            flow_cov = np.cov(flows.T)
            eigvals, eigvecs = np.linalg.eig(flow_cov)
            dominant_dir = eigvecs[:, np.argmax(eigvals)]
        else:
            dominant_dir = mean_flow / np.linalg.norm(mean_flow)
        
        # Project flows onto dominant direction
        projections = np.dot(flows, dominant_dir)
        
        # Count direction changes
        direction_changes = np.sum(np.diff(np.sign(projections)) != 0)
        
        # Check for regular back-and-forth
        if direction_changes >= self.min_repetitions * 2 - 1:
            # Check for regular intervals
            sign_changes_idx = np.where(np.diff(np.sign(projections)) != 0)[0]
            
            if len(sign_changes_idx) > 1:
                intervals = np.diff(sign_changes_idx)
                regularity = 1.0 - np.std(intervals) / (np.mean(intervals) + 1e-6)
                
                if regularity > 0.5:
                    return {
                        'is_back_forth': True,
                        'frequency': float(len(sign_changes_idx) / len(flows)),
                        'regularity': float(regularity),
                        'amplitude': float(np.std(projections))
                    }
        
        return {'is_back_forth': False}
    
    def _classify_patterns(self, temporal: Dict, frequency: Dict,
                          spatial: Dict, motion: Dict) -> List[Dict]:
        """Classify detected patterns into categories."""
        classifications = []
        
        # Temporal pattern classification
        for key, pattern in temporal.items():
            if isinstance(pattern, dict) and 'periodicity' in pattern:
                if pattern['periodicity']['has_period']:
                    period = pattern['periodicity']['period']
                    
                    # Classify by period length
                    if period < 5:
                        pattern_type = 'flicker'
                    elif period < 15:
                        pattern_type = 'rapid_cycle'
                    elif period < 60:
                        pattern_type = 'regular_cycle'
                    else:
                        pattern_type = 'slow_cycle'
                    
                    classifications.append({
                        'type': pattern_type,
                        'source': f'temporal_{key}',
                        'period': period,
                        'confidence': pattern['periodicity']['confidence'],
                        'characteristics': {
                            'regularity': pattern.get('regularity', 0),
                            'modal': key
                        }
                    })
        
        # Frequency pattern classification
        if frequency.get('dominant_frequencies'):
            for freq in frequency['dominant_frequencies']:
                if freq['period'] > 0:
                    classifications.append({
                        'type': 'frequency_pattern',
                        'source': 'frequency_domain',
                        'period': freq['period'],
                        'confidence': freq['power'],
                        'characteristics': {
                            'frequency': freq['frequency'],
                            'bandwidth': freq['bandwidth']
                        }
                    })
        
        # Spatial pattern classification
        if spatial.get('grid_patterns'):
            for grid in spatial['grid_patterns']:
                classifications.append({
                    'type': 'spatial_grid',
                    'source': 'spatial',
                    'period': (grid.get('h_spacing', 0) + grid.get('v_spacing', 0)) / 2,
                    'confidence': grid['regularity'],
                    'characteristics': grid
                })
        
        # Motion pattern classification
        for motion_type in ['oscillatory_motion', 'circular_motion', 'back_forth_motion']:
            if motion.get(motion_type):
                for pattern in motion[motion_type]:
                    classifications.append({
                        'type': motion_type.replace('_', ' '),
                        'source': 'motion',
                        'period': pattern.get('period', 0),
                        'confidence': 0.8,  # High confidence for detected motion patterns
                        'characteristics': pattern
                    })
        
        return classifications
    
    def _detect_pattern_cycles(self, temporal: Dict, frequency: Dict) -> List[Dict]:
        """Detect complete pattern cycles."""
        cycles = []
        
        # Detect cycles from temporal patterns
        for key, pattern in temporal.items():
            if isinstance(pattern, dict) and 'periodicity' in pattern:
                if pattern['periodicity']['has_period']:
                    period = pattern['periodicity']['period']
                    values = pattern.get('values', [])
                    
                    if values and period > 0:
                        num_cycles = len(values) / period
                        
                        # Find cycle boundaries
                        cycle_starts = []
                        for i in range(int(num_cycles)):
                            start_idx = int(i * period)
                            if start_idx < len(values):
                                cycle_starts.append(start_idx)
                        
                        if len(cycle_starts) >= self.min_repetitions:
                            cycles.append({
                                'source': key,
                                'period': period,
                                'num_cycles': int(num_cycles),
                                'cycle_starts': cycle_starts,
                                'completeness': num_cycles / int(num_cycles)
                            })
        
        return cycles
    
    def _analyze_pattern_stability(self, temporal: Dict, 
                                  classifications: List[Dict]) -> Dict:
        """Analyze stability of detected patterns."""
        stability = {
            'overall_stability': 0,
            'pattern_consistency': 0,
            'phase_stability': 0
        }
        
        if not classifications:
            return stability
        
        # Analyze consistency of pattern periods
        periods = [c['period'] for c in classifications if c['period'] > 0]
        if periods:
            # Check if periods are harmonically related
            base_period = min(periods)
            harmonic_scores = []
            
            for period in periods:
                ratio = period / base_period
                nearest_int = round(ratio)
                if nearest_int > 0:
                    deviation = abs(ratio - nearest_int) / nearest_int
                    harmonic_scores.append(1.0 - min(deviation, 1.0))
            
            stability['pattern_consistency'] = float(np.mean(harmonic_scores))
        
        # Analyze phase stability
        for key, pattern in temporal.items():
            if isinstance(pattern, dict) and 'autocorrelation' in pattern:
                autocorr = pattern['autocorrelation']
                if len(autocorr) > 1:
                    # Phase stability from autocorrelation decay
                    decay_rate = np.polyfit(np.arange(len(autocorr)), autocorr, 1)[0]
                    stability['phase_stability'] = float(max(0, 1.0 + decay_rate))
                    break
        
        # Overall stability
        stability['overall_stability'] = float(
            np.mean([stability['pattern_consistency'], stability['phase_stability']])
        )
        
        return stability
    
    def _compute_statistics(self, classifications: List[Dict]) -> Dict:
        """Compute pattern statistics."""
        if not classifications:
            return {
                'num_patterns': 0,
                'dominant_type': 'none',
                'avg_period': 0,
                'avg_confidence': 0
            }
        
        # Type distribution
        type_counts = {}
        periods = []
        confidences = []
        
        for pattern in classifications:
            pattern_type = pattern['type']
            type_counts[pattern_type] = type_counts.get(pattern_type, 0) + 1
            
            if pattern['period'] > 0:
                periods.append(pattern['period'])
            confidences.append(pattern['confidence'])
        
        dominant = max(type_counts.items(), key=lambda x: x[1])[0] if type_counts else 'none'
        
        return {
            'num_patterns': len(classifications),
            'dominant_type': dominant,
            'avg_period': float(np.mean(periods)) if periods else 0,
            'avg_confidence': float(np.mean(confidences)) if confidences else 0,
            'type_distribution': type_counts
        }