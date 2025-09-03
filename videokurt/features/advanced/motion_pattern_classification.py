"""Motion pattern classification using advanced trajectory analysis."""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from scipy import signal, stats
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA
import warnings

from ..base import AdvancedFeature


class MotionPatternClassification(AdvancedFeature):
    """Classify complex motion patterns using trajectory and flow analysis."""
    
    FEATURE_NAME = 'motion_pattern_classification'
    REQUIRED_ANALYSES = ['optical_flow_dense', 'optical_flow_sparse']
    
    def __init__(self, 
                 window_size: int = 30,
                 trajectory_min_length: int = 10,
                 pattern_threshold: float = 0.7,
                 spatial_bins: int = 8,
                 temporal_smoothing: float = 0.5,
                 use_fourier_analysis: bool = True):
        """
        Args:
            window_size: Window size for pattern analysis
            trajectory_min_length: Minimum trajectory length to consider
            pattern_threshold: Confidence threshold for pattern classification
            spatial_bins: Number of spatial bins for motion distribution
            temporal_smoothing: Temporal smoothing factor
            use_fourier_analysis: Use Fourier analysis for periodicity
        """
        super().__init__()
        self.window_size = window_size
        self.trajectory_min_length = trajectory_min_length
        self.pattern_threshold = pattern_threshold
        self.spatial_bins = spatial_bins
        self.temporal_smoothing = temporal_smoothing
        self.use_fourier_analysis = use_fourier_analysis
    
    def _compute_advanced(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Classify motion patterns using multi-scale analysis.
        
        Returns:
            Dict with pattern classifications, trajectories, and dynamics
        """
        dense_flow = analysis_data['optical_flow_dense'].data['flow_field']
        sparse_data = analysis_data['optical_flow_sparse'].data
        
        if len(dense_flow) == 0:
            return self._empty_result()
        
        # Extract motion trajectories from sparse flow
        trajectories = self._extract_trajectories(sparse_data)
        
        # Analyze dense flow patterns
        flow_patterns = self._analyze_flow_patterns(dense_flow)
        
        # Classify motion in sliding windows
        window_classifications = self._classify_motion_windows(
            dense_flow, trajectories
        )
        
        # Detect global motion patterns
        global_patterns = self._detect_global_patterns(
            flow_patterns, trajectories
        )
        
        # Analyze motion dynamics
        dynamics = self._analyze_motion_dynamics(dense_flow, trajectories)
        
        # Detect specific pattern types
        specific_patterns = self._detect_specific_patterns(
            trajectories, flow_patterns
        )
        
        return {
            'window_classifications': window_classifications,
            'global_patterns': global_patterns,
            'trajectories': self._summarize_trajectories(trajectories),
            'dynamics': dynamics,
            'specific_patterns': specific_patterns,
            'statistics': self._compute_statistics(
                window_classifications, global_patterns
            )
        }
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result structure."""
        return {
            'window_classifications': [],
            'global_patterns': {},
            'trajectories': [],
            'dynamics': {},
            'specific_patterns': {},
            'statistics': {
                'dominant_pattern': 'static',
                'pattern_diversity': 0.0,
                'motion_complexity': 0.0
            }
        }
    
    def _extract_trajectories(self, sparse_data: Dict) -> List[Dict]:
        """Extract and analyze motion trajectories from sparse flow."""
        if 'tracked_points' not in sparse_data:
            return []
        
        tracked_points = sparse_data['tracked_points']
        trajectories = []
        
        # Group points into trajectories
        point_tracks = {}
        for frame_idx, frame_points in enumerate(tracked_points):
            if not isinstance(frame_points, (list, np.ndarray)):
                continue
                
            for point_idx, point in enumerate(frame_points):
                track_id = f"{frame_idx}_{point_idx}"
                
                # Try to match with existing trajectory
                matched = False
                for tid, track in point_tracks.items():
                    if track['last_frame'] == frame_idx - 1:
                        last_point = track['points'][-1]
                        distance = np.linalg.norm(
                            np.array(point) - np.array(last_point)
                        )
                        if distance < 50:  # Matching threshold
                            track['points'].append(point)
                            track['last_frame'] = frame_idx
                            matched = True
                            break
                
                if not matched:
                    point_tracks[track_id] = {
                        'points': [point],
                        'start_frame': frame_idx,
                        'last_frame': frame_idx
                    }
        
        # Analyze trajectories
        for track_id, track in point_tracks.items():
            if len(track['points']) >= self.trajectory_min_length:
                trajectory = self._analyze_trajectory(track)
                trajectories.append(trajectory)
        
        return trajectories
    
    def _analyze_trajectory(self, track: Dict) -> Dict:
        """Analyze individual trajectory characteristics."""
        points = np.array(track['points'])
        
        # Basic metrics
        total_distance = np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))
        displacement = np.linalg.norm(points[-1] - points[0])
        
        # Trajectory shape analysis
        if len(points) >= 3:
            # Fit polynomial
            t = np.arange(len(points))
            try:
                poly_x = np.polyfit(t, points[:, 0], min(3, len(points)-1))
                poly_y = np.polyfit(t, points[:, 1], min(3, len(points)-1))
                curvature = self._compute_curvature(points)
            except:
                curvature = 0
            
            # Classify trajectory type
            traj_type = self._classify_trajectory_shape(
                points, total_distance, displacement, curvature
            )
        else:
            traj_type = 'short'
            curvature = 0
        
        return {
            'points': points.tolist(),
            'start_frame': track['start_frame'],
            'duration': len(points),
            'total_distance': float(total_distance),
            'displacement': float(displacement),
            'efficiency': float(displacement / (total_distance + 1e-6)),
            'curvature': float(curvature),
            'type': traj_type
        }
    
    def _compute_curvature(self, points: np.ndarray) -> float:
        """Compute average curvature of trajectory."""
        if len(points) < 3:
            return 0
        
        # Compute curvature at each point
        curvatures = []
        for i in range(1, len(points) - 1):
            v1 = points[i] - points[i-1]
            v2 = points[i+1] - points[i]
            
            # Angle between vectors
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            angle = np.arccos(np.clip(cos_angle, -1, 1))
            
            # Curvature approximation
            arc_length = (np.linalg.norm(v1) + np.linalg.norm(v2)) / 2
            curvature = angle / (arc_length + 1e-6)
            curvatures.append(curvature)
        
        return np.mean(curvatures) if curvatures else 0
    
    def _classify_trajectory_shape(self, points: np.ndarray, 
                                  total_distance: float,
                                  displacement: float,
                                  curvature: float) -> str:
        """Classify trajectory shape based on characteristics."""
        efficiency = displacement / (total_distance + 1e-6)
        
        # Linear trajectory
        if efficiency > 0.9:
            return 'linear'
        
        # Circular/arc trajectory
        if curvature > 0.1 and efficiency < 0.3:
            # Check for circular pattern
            center = np.mean(points, axis=0)
            radii = np.linalg.norm(points - center, axis=1)
            if np.std(radii) / np.mean(radii) < 0.2:
                return 'circular'
            else:
                return 'arc'
        
        # Zigzag pattern
        if self._is_zigzag(points):
            return 'zigzag'
        
        # Spiral pattern
        if self._is_spiral(points):
            return 'spiral'
        
        # Complex/irregular
        if efficiency < 0.5:
            return 'irregular'
        
        return 'curved'
    
    def _is_zigzag(self, points: np.ndarray) -> bool:
        """Check if trajectory follows zigzag pattern."""
        if len(points) < 4:
            return False
        
        # Check for alternating directions
        velocities = np.diff(points, axis=0)
        angles = np.arctan2(velocities[:, 1], velocities[:, 0])
        angle_changes = np.diff(angles)
        
        # Count sign changes
        sign_changes = np.sum(np.diff(np.sign(angle_changes)) != 0)
        
        return sign_changes >= len(angle_changes) * 0.6
    
    def _is_spiral(self, points: np.ndarray) -> bool:
        """Check if trajectory follows spiral pattern."""
        if len(points) < 8:
            return False
        
        center = np.mean(points, axis=0)
        
        # Convert to polar coordinates
        rel_points = points - center
        radii = np.linalg.norm(rel_points, axis=1)
        angles = np.arctan2(rel_points[:, 1], rel_points[:, 0])
        
        # Unwrap angles
        angles = np.unwrap(angles)
        
        # Check for monotonic angle increase/decrease with radius change
        angle_trend = np.polyfit(np.arange(len(angles)), angles, 1)[0]
        radius_trend = np.polyfit(np.arange(len(radii)), radii, 1)[0]
        
        return abs(angle_trend) > 0.1 and abs(radius_trend) > 1.0
    
    def _analyze_flow_patterns(self, flow_field: List[np.ndarray]) -> Dict:
        """Analyze patterns in dense optical flow."""
        patterns = {
            'divergence': [],
            'curl': [],
            'shear': [],
            'uniformity': [],
            'magnitude_distribution': []
        }
        
        for flow in flow_field:
            h, w = flow.shape[:2]
            
            # Compute flow derivatives
            du_dx = np.gradient(flow[..., 0], axis=1)
            du_dy = np.gradient(flow[..., 0], axis=0)
            dv_dx = np.gradient(flow[..., 1], axis=1)
            dv_dy = np.gradient(flow[..., 1], axis=0)
            
            # Divergence (expansion/contraction)
            div = du_dx + dv_dy
            patterns['divergence'].append(np.mean(div))
            
            # Curl (rotation)
            curl = dv_dx - du_dy
            patterns['curl'].append(np.mean(curl))
            
            # Shear
            shear = du_dy + dv_dx
            patterns['shear'].append(np.mean(np.abs(shear)))
            
            # Uniformity (how similar flow vectors are)
            flow_flat = flow.reshape(-1, 2)
            if len(flow_flat) > 1:
                distances = pdist(flow_flat, metric='cosine')
                uniformity = 1.0 - np.mean(distances)
            else:
                uniformity = 1.0
            patterns['uniformity'].append(uniformity)
            
            # Magnitude distribution
            magnitudes = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            patterns['magnitude_distribution'].append({
                'mean': float(np.mean(magnitudes)),
                'std': float(np.std(magnitudes)),
                'max': float(np.max(magnitudes))
            })
        
        return patterns
    
    def _classify_motion_windows(self, flow_field: List[np.ndarray],
                                trajectories: List[Dict]) -> List[Dict]:
        """Classify motion patterns in sliding windows."""
        classifications = []
        
        for i in range(0, len(flow_field), self.window_size // 2):
            window_end = min(i + self.window_size, len(flow_field))
            window_flows = flow_field[i:window_end]
            
            if len(window_flows) < 3:
                continue
            
            # Extract window features
            features = self._extract_window_features(window_flows)
            
            # Find trajectories in this window
            window_trajs = [t for t in trajectories 
                           if t['start_frame'] >= i and 
                           t['start_frame'] < window_end]
            
            # Classify pattern
            pattern, confidence = self._classify_pattern(features, window_trajs)
            
            classifications.append({
                'start_frame': i,
                'end_frame': window_end,
                'pattern': pattern,
                'confidence': float(confidence),
                'features': features
            })
        
        return classifications
    
    def _extract_window_features(self, flows: np.ndarray) -> Dict:
        """Extract motion features from flow window."""
        features = {}
        
        # Temporal consistency
        flow_changes = []
        for i in range(1, len(flows)):
            change = np.mean(np.abs(flows[i] - flows[i-1]))
            flow_changes.append(change)
        features['temporal_consistency'] = 1.0 / (1.0 + np.std(flow_changes))
        
        # Spatial coherence
        coherences = []
        for flow in flows:
            h, w = flow.shape[:2]
            # Sample patches
            patch_size = min(h, w) // 4
            patches = []
            for y in range(0, h - patch_size, patch_size):
                for x in range(0, w - patch_size, patch_size):
                    patch = flow[y:y+patch_size, x:x+patch_size]
                    patches.append(np.mean(patch, axis=(0, 1)))
            
            if len(patches) > 1:
                patch_var = np.var(patches, axis=0)
                coherence = 1.0 / (1.0 + np.mean(patch_var))
            else:
                coherence = 1.0
            coherences.append(coherence)
        features['spatial_coherence'] = float(np.mean(coherences))
        
        # Dominant direction
        all_flows = np.concatenate([f.reshape(-1, 2) for f in flows])
        mean_flow = np.mean(all_flows, axis=0)
        features['dominant_direction'] = float(np.arctan2(mean_flow[1], mean_flow[0]))
        features['dominant_magnitude'] = float(np.linalg.norm(mean_flow))
        
        # Periodicity (if using Fourier)
        if self.use_fourier_analysis:
            features['periodicity'] = self._compute_periodicity(flows)
        
        return features
    
    def _compute_periodicity(self, flows: np.ndarray) -> float:
        """Compute periodicity score using Fourier analysis."""
        # Average flow magnitude over time
        magnitudes = [np.mean(np.sqrt(f[..., 0]**2 + f[..., 1]**2)) 
                     for f in flows]
        
        if len(magnitudes) < 4:
            return 0.0
        
        # FFT
        fft = np.fft.fft(magnitudes)
        freqs = np.fft.fftfreq(len(magnitudes))
        
        # Find dominant frequency (excluding DC)
        power = np.abs(fft[1:len(fft)//2])**2
        if len(power) > 0:
            dominant_idx = np.argmax(power)
            dominant_power = power[dominant_idx]
            total_power = np.sum(power)
            
            # Periodicity score based on dominant frequency strength
            periodicity = dominant_power / (total_power + 1e-6)
        else:
            periodicity = 0.0
        
        return float(periodicity)
    
    def _classify_pattern(self, features: Dict, 
                         trajectories: List[Dict]) -> Tuple[str, float]:
        """Classify motion pattern based on features."""
        # Static
        if features['dominant_magnitude'] < 0.5:
            return 'static', 0.95
        
        # Linear/directional
        if features['spatial_coherence'] > 0.8 and features['temporal_consistency'] > 0.7:
            return 'linear', 0.85
        
        # Oscillatory
        if features.get('periodicity', 0) > 0.4:
            return 'oscillatory', 0.8
        
        # Rotational
        if len(trajectories) > 0:
            circular_count = sum(1 for t in trajectories if t['type'] == 'circular')
            if circular_count / len(trajectories) > 0.5:
                return 'rotational', 0.75
        
        # Expansion/contraction
        if abs(features.get('divergence_mean', 0)) > 2.0:
            if features.get('divergence_mean', 0) > 0:
                return 'expansion', 0.7
            else:
                return 'contraction', 0.7
        
        # Turbulent/chaotic
        if features['spatial_coherence'] < 0.3 and features['temporal_consistency'] < 0.3:
            return 'turbulent', 0.7
        
        # Shear
        if features.get('shear_mean', 0) > 1.5:
            return 'shear', 0.65
        
        # Complex
        return 'complex', 0.5
    
    def _detect_global_patterns(self, flow_patterns: Dict,
                               trajectories: List[Dict]) -> Dict:
        """Detect global motion patterns across entire video."""
        global_patterns = {
            'has_vortex': False,
            'has_source_sink': False,
            'has_laminar_flow': False,
            'has_periodic_motion': False,
            'dominant_flow_type': 'none'
        }
        
        if not flow_patterns['divergence']:
            return global_patterns
        
        # Vortex detection (strong curl)
        curl_values = flow_patterns['curl']
        if np.mean(np.abs(curl_values)) > 1.0:
            global_patterns['has_vortex'] = True
        
        # Source/sink detection (strong divergence)
        div_values = flow_patterns['divergence']
        if np.max(np.abs(div_values)) > 2.0:
            global_patterns['has_source_sink'] = True
        
        # Laminar flow (high uniformity, low shear)
        if (np.mean(flow_patterns['uniformity']) > 0.7 and 
            np.mean(flow_patterns['shear']) < 0.5):
            global_patterns['has_laminar_flow'] = True
        
        # Periodic motion
        if self.use_fourier_analysis:
            # Check for periodicity in magnitude
            mag_timeline = [m['mean'] for m in flow_patterns['magnitude_distribution']]
            if len(mag_timeline) > 10:
                periodicity = self._compute_periodicity_score(mag_timeline)
                global_patterns['has_periodic_motion'] = periodicity > 0.5
        
        # Determine dominant flow type
        flow_scores = {
            'uniform': np.mean(flow_patterns['uniformity']),
            'rotational': np.mean(np.abs(flow_patterns['curl'])),
            'divergent': np.mean(np.abs(flow_patterns['divergence'])),
            'shear': np.mean(flow_patterns['shear'])
        }
        global_patterns['dominant_flow_type'] = max(flow_scores.items(), 
                                                   key=lambda x: x[1])[0]
        
        return global_patterns
    
    def _compute_periodicity_score(self, timeline: List[float]) -> float:
        """Compute periodicity score for a timeline."""
        if len(timeline) < 4:
            return 0.0
        
        # Autocorrelation
        timeline = np.array(timeline)
        timeline = timeline - np.mean(timeline)
        
        autocorr = np.correlate(timeline, timeline, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]
        
        # Find peaks in autocorrelation
        peaks, _ = signal.find_peaks(autocorr[1:], height=0.3)
        
        if len(peaks) > 0:
            # Strong periodicity if regular peaks
            peak_distances = np.diff(peaks)
            if len(peak_distances) > 0:
                regularity = 1.0 - np.std(peak_distances) / (np.mean(peak_distances) + 1e-6)
                return min(1.0, regularity * autocorr[peaks[0] + 1])
        
        return 0.0
    
    def _analyze_motion_dynamics(self, flow_field: List[np.ndarray],
                                trajectories: List[Dict]) -> Dict:
        """Analyze motion dynamics and complexity."""
        dynamics = {
            'acceleration_patterns': [],
            'complexity_score': 0.0,
            'stability_score': 0.0,
            'energy_timeline': []
        }
        
        # Compute motion energy over time
        for flow in flow_field:
            energy = np.mean(flow[..., 0]**2 + flow[..., 1]**2)
            dynamics['energy_timeline'].append(float(energy))
        
        # Acceleration patterns
        if len(dynamics['energy_timeline']) > 2:
            accelerations = np.diff(dynamics['energy_timeline'], n=2)
            dynamics['acceleration_patterns'] = {
                'mean': float(np.mean(accelerations)),
                'std': float(np.std(accelerations)),
                'max': float(np.max(np.abs(accelerations)))
            }
        
        # Complexity score (entropy-based)
        if trajectories:
            trajectory_types = [t['type'] for t in trajectories]
            type_counts = {}
            for t_type in trajectory_types:
                type_counts[t_type] = type_counts.get(t_type, 0) + 1
            
            probs = np.array(list(type_counts.values())) / len(trajectory_types)
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            dynamics['complexity_score'] = float(entropy / np.log(len(type_counts) + 1e-10))
        
        # Stability score
        if len(dynamics['energy_timeline']) > 1:
            energy_var = np.var(dynamics['energy_timeline'])
            energy_mean = np.mean(dynamics['energy_timeline'])
            dynamics['stability_score'] = float(1.0 / (1.0 + energy_var / (energy_mean + 1e-6)))
        
        return dynamics
    
    def _detect_specific_patterns(self, trajectories: List[Dict],
                                 flow_patterns: Dict) -> Dict:
        """Detect specific motion patterns."""
        patterns = {
            'has_tracking_motion': False,
            'has_explosion_implosion': False,
            'has_wave_motion': False,
            'has_swirl_motion': False,
            'pattern_locations': []
        }
        
        # Tracking motion (consistent linear trajectories)
        if trajectories:
            linear_trajs = [t for t in trajectories if t['type'] == 'linear']
            if len(linear_trajs) / len(trajectories) > 0.5:
                patterns['has_tracking_motion'] = True
        
        # Explosion/implosion (strong divergence spike)
        if flow_patterns['divergence']:
            div_array = np.array(flow_patterns['divergence'])
            if np.max(np.abs(div_array)) > 3.0:
                patterns['has_explosion_implosion'] = True
                # Find frame
                frame = np.argmax(np.abs(div_array))
                patterns['pattern_locations'].append({
                    'type': 'explosion' if div_array[frame] > 0 else 'implosion',
                    'frame': int(frame),
                    'strength': float(abs(div_array[frame]))
                })
        
        # Wave motion (periodic with spatial propagation)
        if flow_patterns['uniformity'] and len(flow_patterns['uniformity']) > 10:
            uniformity_fft = np.fft.fft(flow_patterns['uniformity'])
            if np.max(np.abs(uniformity_fft[1:len(uniformity_fft)//2])) > 0.3:
                patterns['has_wave_motion'] = True
        
        # Swirl motion (high curl values)
        if flow_patterns['curl']:
            curl_array = np.array(flow_patterns['curl'])
            if np.mean(np.abs(curl_array)) > 1.5:
                patterns['has_swirl_motion'] = True
        
        return patterns
    
    def _summarize_trajectories(self, trajectories: List[Dict]) -> List[Dict]:
        """Summarize trajectories for output."""
        # Limit output size
        summary = []
        for traj in trajectories[:20]:  # Top 20 trajectories
            summary.append({
                'start_frame': traj['start_frame'],
                'duration': traj['duration'],
                'type': traj['type'],
                'efficiency': traj['efficiency'],
                'displacement': traj['displacement']
            })
        
        return summary
    
    def _compute_statistics(self, classifications: List[Dict],
                           global_patterns: Dict) -> Dict:
        """Compute motion pattern statistics."""
        if not classifications:
            return {
                'dominant_pattern': 'static',
                'pattern_diversity': 0.0,
                'motion_complexity': 0.0,
                'avg_confidence': 0.0
            }
        
        # Pattern counts
        pattern_counts = {}
        for c in classifications:
            pattern = c['pattern']
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        # Dominant pattern
        dominant = max(pattern_counts.items(), key=lambda x: x[1])[0]
        
        # Pattern diversity (entropy)
        total = sum(pattern_counts.values())
        probs = [count/total for count in pattern_counts.values()]
        diversity = -sum(p * np.log(p + 1e-10) for p in probs)
        
        # Motion complexity
        complexity_factors = [
            global_patterns.get('has_vortex', False),
            global_patterns.get('has_source_sink', False),
            global_patterns.get('has_periodic_motion', False),
            len(pattern_counts) > 3
        ]
        complexity = sum(complexity_factors) / len(complexity_factors)
        
        return {
            'dominant_pattern': dominant,
            'pattern_diversity': float(diversity),
            'motion_complexity': float(complexity),
            'avg_confidence': float(np.mean([c['confidence'] for c in classifications])),
            'pattern_distribution': pattern_counts
        }