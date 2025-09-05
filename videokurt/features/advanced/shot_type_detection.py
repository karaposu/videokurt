"""Shot type detection using motion analysis and visual composition."""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from scipy import signal, stats
from scipy.ndimage import gaussian_filter, median_filter
from scipy.spatial import distance
import cv2

from ..base import BaseFeature


class ShotTypeDetection(BaseFeature):
    """Detect camera shot types and cinematographic techniques."""
    
    FEATURE_NAME = 'shot_type_detection'
    REQUIRED_ANALYSES = ['optical_flow_dense', 'frame_diff', 'edge_canny', 'face_detection']
    
    def __init__(self, 
                 stability_threshold: float = 0.1,
                 shake_frequency_range: Tuple[float, float] = (1.0, 10.0),
                 composition_grid_size: int = 3,
                 temporal_window: int = 15,
                 shot_min_duration: int = 10,
                 use_composition_analysis: bool = True):
        """
        Args:
            stability_threshold: Threshold for camera stability
            shake_frequency_range: Frequency range for handheld shake (Hz)
            composition_grid_size: Grid size for composition analysis
            temporal_window: Window for temporal analysis
            shot_min_duration: Minimum frames for a shot segment
            use_composition_analysis: Use rule-of-thirds and composition
        """
        super().__init__()
        self.stability_threshold = stability_threshold
        self.shake_frequency_range = shake_frequency_range
        self.composition_grid_size = composition_grid_size
        self.temporal_window = temporal_window
        self.shot_min_duration = shot_min_duration
        self.use_composition_analysis = use_composition_analysis
    
    def compute(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect shot types using comprehensive motion and composition analysis.
        
        Returns:
            Dict with shot classifications, techniques, and cinematography
        """
        flow_field = analysis_data['optical_flow_dense'].data['flow_field']
        frame_diffs = analysis_data['frame_diff'].data['pixel_diff']
        edge_maps = analysis_data['edge_canny'].data['edge_map']
        face_data = analysis_data.get('face_detection', {}).get('data', {})
        
        if len(flow_field) == 0:
            return self._empty_result()
        
        # Extract motion characteristics
        motion_features = self._extract_motion_features(flow_field, frame_diffs)
        
        # Analyze composition if enabled
        if self.use_composition_analysis:
            composition_features = self._analyze_composition(edge_maps, face_data)
        else:
            composition_features = None
        
        # Classify shots with temporal consistency
        shot_classifications = self._classify_shots_temporal(
            motion_features, composition_features
        )
        
        # Segment into shot sequences
        shot_segments = self._segment_shots(shot_classifications)
        
        # Detect cinematographic techniques
        techniques = self._detect_cinematographic_techniques(
            motion_features, shot_segments
        )
        
        # Analyze shot transitions
        transitions = self._analyze_shot_transitions(shot_segments, flow_field)
        
        # Compute shot quality metrics
        quality_metrics = self._compute_quality_metrics(
            motion_features, shot_segments
        )
        
        return {
            'shot_segments': shot_segments,
            'techniques': techniques,
            'transitions': transitions,
            'quality_metrics': quality_metrics,
            'motion_characteristics': self._summarize_motion(motion_features),
            'statistics': self._compute_statistics(shot_segments, techniques)
        }
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result structure."""
        return {
            'shot_segments': [],
            'techniques': {},
            'transitions': [],
            'quality_metrics': {},
            'motion_characteristics': {},
            'statistics': {
                'dominant_shot_type': 'static',
                'avg_shot_duration': 0,
                'shot_variety': 0
            }
        }
    
    def _extract_motion_features(self, flow_field: List[np.ndarray],
                                frame_diffs: List[np.ndarray]) -> Dict:
        """Extract comprehensive motion features for shot classification."""
        features = {
            'magnitude': [],
            'direction': [],
            'uniformity': [],
            'stability': [],
            'shake_score': [],
            'zoom_score': [],
            'pan_tilt_score': [],
            'rotation_score': []
        }
        
        for i, flow in enumerate(flow_field):
            h, w = flow.shape[:2]
            
            # Basic motion metrics
            flow_x = flow[..., 0]
            flow_y = flow[..., 1]
            magnitude = np.sqrt(flow_x**2 + flow_y**2)
            
            features['magnitude'].append(np.mean(magnitude))
            
            # Direction (average flow angle)
            avg_angle = np.arctan2(np.mean(flow_y), np.mean(flow_x))
            features['direction'].append(avg_angle)
            
            # Uniformity (how consistent the flow is)
            if np.mean(magnitude) > 0:
                uniformity = 1.0 - np.std(magnitude) / np.mean(magnitude)
            else:
                uniformity = 1.0
            features['uniformity'].append(uniformity)
            
            # Stability score
            stability = self._compute_stability_score(flow, magnitude)
            features['stability'].append(stability)
            
            # Shake detection
            shake = self._detect_shake_pattern(flow, magnitude)
            features['shake_score'].append(shake)
            
            # Zoom detection
            zoom = self._detect_zoom_pattern(flow, w//2, h//2)
            features['zoom_score'].append(zoom)
            
            # Pan/tilt detection
            pan_tilt = self._detect_pan_tilt(flow_x, flow_y)
            features['pan_tilt_score'].append(pan_tilt)
            
            # Rotation detection
            rotation = self._detect_rotation(flow, w//2, h//2)
            features['rotation_score'].append(rotation)
        
        # Apply temporal smoothing
        for key in features:
            if len(features[key]) > 3:
                features[key] = gaussian_filter(features[key], sigma=1.0).tolist()
        
        return features
    
    def _compute_stability_score(self, flow: np.ndarray, magnitude: np.ndarray) -> float:
        """Compute camera stability score."""
        # Low magnitude and low variance = stable
        avg_mag = np.mean(magnitude)
        mag_std = np.std(magnitude)
        
        if avg_mag < self.stability_threshold:
            return 1.0
        
        # Check for consistent motion (tracking shot)
        flow_vectors = flow.reshape(-1, 2)
        if len(flow_vectors) > 1:
            # Compute pairwise angles between flow vectors
            angles = []
            for i in range(min(100, len(flow_vectors))):
                if np.linalg.norm(flow_vectors[i]) > 0:
                    angle = np.arctan2(flow_vectors[i][1], flow_vectors[i][0])
                    angles.append(angle)
            
            if angles:
                angle_consistency = 1.0 - np.std(angles) / (np.pi + 1e-6)
            else:
                angle_consistency = 0
        else:
            angle_consistency = 1.0
        
        stability = (1.0 / (1.0 + avg_mag)) * angle_consistency
        return float(stability)
    
    def _detect_shake_pattern(self, flow: np.ndarray, magnitude: np.ndarray) -> float:
        """Detect handheld shake characteristics."""
        # High frequency, low amplitude motion = shake
        avg_mag = np.mean(magnitude)
        
        if avg_mag < 0.5 or avg_mag > 5.0:
            return 0.0
        
        # Analyze flow randomness
        flow_x = flow[..., 0].flatten()
        flow_y = flow[..., 1].flatten()
        
        # Compute autocorrelation to detect jitter
        if len(flow_x) > 10:
            autocorr_x = np.correlate(flow_x[:100], flow_x[:100], mode='same')
            autocorr_y = np.correlate(flow_y[:100], flow_y[:100], mode='same')
            
            # High frequency components indicate shake
            freq_score = np.std(np.diff(autocorr_x)) + np.std(np.diff(autocorr_y))
            shake_score = min(1.0, freq_score / 10.0)
        else:
            shake_score = 0.0
        
        return float(shake_score)
    
    def _detect_zoom_pattern(self, flow: np.ndarray, cx: int, cy: int) -> float:
        """Detect zoom/dolly pattern in optical flow."""
        h, w = flow.shape[:2]
        
        # Sample points in a grid
        sample_points = []
        radial_components = []
        
        for y in range(0, h, h//8):
            for x in range(0, w, w//8):
                dx = x - cx
                dy = y - cy
                r = np.sqrt(dx**2 + dy**2)
                
                if r > 10:  # Skip center
                    # Normalized radial vector
                    rx = dx / r
                    ry = dy / r
                    
                    # Radial component of flow
                    radial_flow = flow[y, x, 0] * rx + flow[y, x, 1] * ry
                    radial_components.append(radial_flow)
        
        if radial_components:
            # Consistent radial flow = zoom
            mean_radial = np.mean(radial_components)
            std_radial = np.std(radial_components)
            
            if abs(mean_radial) > 0.5:
                consistency = 1.0 - std_radial / (abs(mean_radial) + 1e-6)
                zoom_score = consistency * min(1.0, abs(mean_radial) / 2.0)
            else:
                zoom_score = 0.0
        else:
            zoom_score = 0.0
        
        return float(zoom_score)
    
    def _detect_pan_tilt(self, flow_x: np.ndarray, flow_y: np.ndarray) -> float:
        """Detect pan/tilt camera movement."""
        avg_x = np.mean(flow_x)
        avg_y = np.mean(flow_y)
        
        # Strong horizontal or vertical movement
        if abs(avg_x) > 0.5 or abs(avg_y) > 0.5:
            # Check consistency
            std_x = np.std(flow_x)
            std_y = np.std(flow_y)
            
            if abs(avg_x) > abs(avg_y):
                # Horizontal pan
                consistency = 1.0 - std_x / (abs(avg_x) + 1e-6)
            else:
                # Vertical tilt
                consistency = 1.0 - std_y / (abs(avg_y) + 1e-6)
            
            pan_tilt_score = consistency * min(1.0, max(abs(avg_x), abs(avg_y)) / 2.0)
        else:
            pan_tilt_score = 0.0
        
        return float(pan_tilt_score)
    
    def _detect_rotation(self, flow: np.ndarray, cx: int, cy: int) -> float:
        """Detect rotational camera movement."""
        h, w = flow.shape[:2]
        
        tangential_components = []
        
        for y in range(0, h, h//8):
            for x in range(0, w, w//8):
                dx = x - cx
                dy = y - cy
                r = np.sqrt(dx**2 + dy**2)
                
                if r > 10:
                    # Tangential vector (perpendicular to radial)
                    tx = -dy / r
                    ty = dx / r
                    
                    # Tangential component of flow
                    tang_flow = flow[y, x, 0] * tx + flow[y, x, 1] * ty
                    tangential_components.append(tang_flow)
        
        if tangential_components:
            mean_tang = np.mean(tangential_components)
            std_tang = np.std(tangential_components)
            
            if abs(mean_tang) > 0.3:
                consistency = 1.0 - std_tang / (abs(mean_tang) + 1e-6)
                rotation_score = consistency * min(1.0, abs(mean_tang) / 1.5)
            else:
                rotation_score = 0.0
        else:
            rotation_score = 0.0
        
        return float(rotation_score)
    
    def _analyze_composition(self, edge_maps: List[np.ndarray],
                           face_data: Dict) -> Dict:
        """Analyze visual composition for shot classification."""
        composition = {
            'rule_of_thirds': [],
            'symmetry': [],
            'depth_layers': [],
            'face_positions': [],
            'leading_lines': []
        }
        
        faces = face_data.get('faces', [])
        
        for i, edges in enumerate(edge_maps):
            h, w = edges.shape
            
            # Rule of thirds analysis
            thirds_score = self._analyze_rule_of_thirds(edges)
            composition['rule_of_thirds'].append(thirds_score)
            
            # Symmetry analysis
            symmetry = self._analyze_symmetry(edges)
            composition['symmetry'].append(symmetry)
            
            # Depth layers (foreground/background separation)
            depth = self._analyze_depth_layers(edges)
            composition['depth_layers'].append(depth)
            
            # Face positions if available
            if i < len(faces) and faces[i]:
                face_comp = self._analyze_face_composition(faces[i], w, h)
                composition['face_positions'].append(face_comp)
            else:
                composition['face_positions'].append({'centered': False, 'rule_of_thirds': False})
            
            # Leading lines
            lines = self._detect_leading_lines(edges)
            composition['leading_lines'].append(lines)
        
        return composition
    
    def _analyze_rule_of_thirds(self, edges: np.ndarray) -> float:
        """Analyze if important elements follow rule of thirds."""
        h, w = edges.shape
        
        # Define rule of thirds lines
        h_third = h // 3
        w_third = w // 3
        
        # Check edge density at intersection points
        intersections = [
            (h_third, w_third),
            (h_third, 2*w_third),
            (2*h_third, w_third),
            (2*h_third, 2*w_third)
        ]
        
        scores = []
        for y, x in intersections:
            # Sample region around intersection
            region = edges[max(0, y-20):min(h, y+20),
                          max(0, x-20):min(w, x+20)]
            density = np.mean(region > 0)
            scores.append(density)
        
        return float(np.max(scores) if scores else 0)
    
    def _analyze_symmetry(self, edges: np.ndarray) -> float:
        """Analyze visual symmetry in the frame."""
        h, w = edges.shape
        
        # Vertical symmetry
        left = edges[:, :w//2]
        right = np.fliplr(edges[:, w//2:w//2*2])
        
        if left.shape == right.shape:
            v_symmetry = 1.0 - np.mean(np.abs(left - right)) / 255.0
        else:
            v_symmetry = 0
        
        # Horizontal symmetry
        top = edges[:h//2, :]
        bottom = np.flipud(edges[h//2:h//2*2, :])
        
        if top.shape == bottom.shape:
            h_symmetry = 1.0 - np.mean(np.abs(top - bottom)) / 255.0
        else:
            h_symmetry = 0
        
        return float(max(v_symmetry, h_symmetry))
    
    def _analyze_depth_layers(self, edges: np.ndarray) -> int:
        """Estimate number of depth layers from edges."""
        h, w = edges.shape
        
        # Horizontal bands analysis
        bands = []
        band_height = h // 5
        
        for i in range(5):
            band = edges[i*band_height:(i+1)*band_height]
            density = np.mean(band > 0)
            bands.append(density)
        
        # Count significant changes in density (depth layers)
        layers = 1
        for i in range(1, len(bands)):
            if abs(bands[i] - bands[i-1]) > 0.1:
                layers += 1
        
        return min(3, layers)
    
    def _analyze_face_composition(self, faces: List[Dict], w: int, h: int) -> Dict:
        """Analyze face positioning in frame."""
        if not faces:
            return {'centered': False, 'rule_of_thirds': False}
        
        # Get largest face
        largest_face = max(faces, key=lambda f: f.get('area', 0))
        
        if 'bbox' in largest_face:
            x, y, fw, fh = largest_face['bbox']
            face_center_x = x + fw // 2
            face_center_y = y + fh // 2
            
            # Check if centered
            centered = (abs(face_center_x - w//2) < w * 0.1 and 
                       abs(face_center_y - h//2) < h * 0.1)
            
            # Check rule of thirds
            thirds_x = abs(face_center_x - w//3) < w * 0.1 or abs(face_center_x - 2*w//3) < w * 0.1
            thirds_y = abs(face_center_y - h//3) < h * 0.1 or abs(face_center_y - 2*h//3) < h * 0.1
            rule_of_thirds = thirds_x and thirds_y
            
            return {'centered': centered, 'rule_of_thirds': rule_of_thirds}
        
        return {'centered': False, 'rule_of_thirds': False}
    
    def _detect_leading_lines(self, edges: np.ndarray) -> float:
        """Detect leading lines in composition."""
        # Use Hough transform to detect lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50,
                               minLineLength=50, maxLineGap=10)
        
        if lines is not None:
            # Check for converging lines (leading lines)
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1)
                angles.append(angle)
            
            if len(angles) > 2:
                # Variance in angles suggests converging lines
                angle_var = np.var(angles)
                if angle_var > 0.5:
                    return float(min(1.0, len(lines) / 20.0))
        
        return 0.0
    
    def _classify_shots_temporal(self, motion_features: Dict,
                                composition_features: Optional[Dict]) -> List[Dict]:
        """Classify shots with temporal consistency."""
        classifications = []
        
        num_frames = len(motion_features['magnitude'])
        
        for i in range(num_frames):
            # Extract features for current frame
            features = {
                'magnitude': motion_features['magnitude'][i],
                'uniformity': motion_features['uniformity'][i],
                'stability': motion_features['stability'][i],
                'shake': motion_features['shake_score'][i],
                'zoom': motion_features['zoom_score'][i],
                'pan_tilt': motion_features['pan_tilt_score'][i],
                'rotation': motion_features['rotation_score'][i]
            }
            
            # Add composition features if available
            if composition_features and i < len(composition_features['rule_of_thirds']):
                features['composition'] = {
                    'rule_of_thirds': composition_features['rule_of_thirds'][i],
                    'symmetry': composition_features['symmetry'][i],
                    'depth': composition_features['depth_layers'][i]
                }
            
            # Classify shot type
            shot_type, confidence, subtype = self._classify_single_shot(features)
            
            classifications.append({
                'frame': i,
                'type': shot_type,
                'subtype': subtype,
                'confidence': confidence,
                'features': features
            })
        
        return classifications
    
    def _classify_single_shot(self, features: Dict) -> Tuple[str, float, str]:
        """Classify a single frame's shot type."""
        # Static shot
        if features['stability'] > 0.9 and features['magnitude'] < self.stability_threshold:
            return 'static', 0.95, 'locked'
        
        # Handheld shot
        if features['shake'] > 0.5:
            if features['magnitude'] < 1.0:
                return 'handheld', 0.85, 'steady'
            else:
                return 'handheld', 0.85, 'walking'
        
        # Tracking shot
        if features['pan_tilt'] > 0.6 and features['uniformity'] > 0.7:
            if features['magnitude'] > 2.0:
                return 'tracking', 0.9, 'fast'
            else:
                return 'tracking', 0.9, 'slow'
        
        # Zoom/dolly shot
        if features['zoom'] > 0.5:
            if features['zoom'] > 0:
                return 'zoom', 0.85, 'in'
            else:
                return 'zoom', 0.85, 'out'
        
        # Crane shot (vertical movement)
        if features['pan_tilt'] > 0.5:
            # Check if vertical component dominates
            return 'crane', 0.75, 'vertical'
        
        # Rotating shot
        if features['rotation'] > 0.5:
            return 'rotating', 0.8, 'roll'
        
        # Establishing shot (wide, stable, good composition)
        if features.get('composition', {}).get('depth', 0) > 2:
            return 'establishing', 0.7, 'wide'
        
        # Dynamic/action shot
        if features['magnitude'] > 3.0 and features['uniformity'] < 0.5:
            return 'action', 0.75, 'dynamic'
        
        # Default
        return 'medium', 0.5, 'standard'
    
    def _segment_shots(self, classifications: List[Dict]) -> List[Dict]:
        """Segment classifications into coherent shot sequences."""
        if not classifications:
            return []
        
        segments = []
        current_segment = {
            'type': classifications[0]['type'],
            'subtype': classifications[0]['subtype'],
            'start_frame': 0,
            'end_frame': 0,
            'confidence': classifications[0]['confidence']
        }
        
        for i in range(1, len(classifications)):
            c = classifications[i]
            
            # Check if same shot continues
            if (c['type'] == current_segment['type'] and 
                c['subtype'] == current_segment['subtype']):
                current_segment['end_frame'] = i
                current_segment['confidence'] = max(current_segment['confidence'],
                                                   c['confidence'])
            else:
                # Save current segment if long enough
                duration = current_segment['end_frame'] - current_segment['start_frame'] + 1
                if duration >= self.shot_min_duration:
                    segments.append(current_segment)
                
                # Start new segment
                current_segment = {
                    'type': c['type'],
                    'subtype': c['subtype'],
                    'start_frame': i,
                    'end_frame': i,
                    'confidence': c['confidence']
                }
        
        # Add final segment
        duration = current_segment['end_frame'] - current_segment['start_frame'] + 1
        if duration >= self.shot_min_duration:
            segments.append(current_segment)
        
        return segments
    
    def _detect_cinematographic_techniques(self, motion_features: Dict,
                                          segments: List[Dict]) -> Dict:
        """Detect specific cinematographic techniques."""
        techniques = {
            'has_dolly_zoom': False,
            'has_whip_pan': False,
            'has_dutch_angle': False,
            'has_rack_focus': False,
            'has_long_take': False,
            'technique_instances': []
        }
        
        # Dolly zoom (zoom + opposite camera movement)
        for i in range(1, len(motion_features['zoom_score'])):
            zoom = motion_features['zoom_score'][i]
            movement = motion_features['magnitude'][i]
            
            if abs(zoom) > 0.5 and movement > 1.0:
                # Check if zoom and movement are opposite
                techniques['has_dolly_zoom'] = True
                techniques['technique_instances'].append({
                    'type': 'dolly_zoom',
                    'frame': i,
                    'strength': float(abs(zoom))
                })
        
        # Whip pan (very fast pan)
        for i, pan in enumerate(motion_features['pan_tilt_score']):
            if pan > 0.8 and motion_features['magnitude'][i] > 5.0:
                techniques['has_whip_pan'] = True
                techniques['technique_instances'].append({
                    'type': 'whip_pan',
                    'frame': i,
                    'speed': float(motion_features['magnitude'][i])
                })
        
        # Dutch angle (rotation)
        for i, rot in enumerate(motion_features['rotation_score']):
            if rot > 0.3:
                techniques['has_dutch_angle'] = True
                techniques['technique_instances'].append({
                    'type': 'dutch_angle',
                    'frame': i,
                    'angle': float(rot * 45)  # Approximate angle
                })
        
        # Long take (extended single shot)
        for seg in segments:
            duration = seg['end_frame'] - seg['start_frame'] + 1
            if duration > 150:  # ~5 seconds at 30fps
                techniques['has_long_take'] = True
                techniques['technique_instances'].append({
                    'type': 'long_take',
                    'start': seg['start_frame'],
                    'duration': duration
                })
        
        return techniques
    
    def _analyze_shot_transitions(self, segments: List[Dict],
                                 flow_field: List[np.ndarray]) -> List[Dict]:
        """Analyze transitions between shots."""
        transitions = []
        
        for i in range(1, len(segments)):
            prev_seg = segments[i-1]
            curr_seg = segments[i]
            
            transition_frame = curr_seg['start_frame']
            
            # Analyze flow at transition
            if transition_frame < len(flow_field):
                flow = flow_field[transition_frame]
                magnitude = np.mean(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2))
                
                # Classify transition
                if magnitude > 10.0:
                    trans_type = 'cut'
                elif magnitude > 5.0:
                    trans_type = 'fast_cut'
                elif prev_seg['type'] != curr_seg['type']:
                    trans_type = 'smooth'
                else:
                    trans_type = 'continuous'
                
                transitions.append({
                    'frame': transition_frame,
                    'from_shot': prev_seg['type'],
                    'to_shot': curr_seg['type'],
                    'type': trans_type,
                    'smoothness': float(1.0 / (1.0 + magnitude))
                })
        
        return transitions
    
    def _compute_quality_metrics(self, motion_features: Dict,
                                segments: List[Dict]) -> Dict:
        """Compute shot quality metrics."""
        metrics = {
            'stability_score': 0.0,
            'composition_score': 0.0,
            'smoothness_score': 0.0,
            'variety_score': 0.0
        }
        
        if motion_features['stability']:
            metrics['stability_score'] = float(np.mean(motion_features['stability']))
        
        # Smoothness (low jitter)
        if len(motion_features['magnitude']) > 1:
            magnitude_changes = np.diff(motion_features['magnitude'])
            metrics['smoothness_score'] = float(1.0 / (1.0 + np.std(magnitude_changes)))
        
        # Variety (different shot types)
        if segments:
            unique_shots = len(set(s['type'] for s in segments))
            metrics['variety_score'] = float(unique_shots / len(segments))
        
        return metrics
    
    def _summarize_motion(self, motion_features: Dict) -> Dict:
        """Summarize motion characteristics."""
        return {
            'avg_magnitude': float(np.mean(motion_features['magnitude'])),
            'avg_stability': float(np.mean(motion_features['stability'])),
            'has_shake': any(s > 0.5 for s in motion_features['shake_score']),
            'has_zoom': any(z > 0.5 for z in motion_features['zoom_score']),
            'has_pan_tilt': any(p > 0.5 for p in motion_features['pan_tilt_score']),
            'has_rotation': any(r > 0.3 for r in motion_features['rotation_score'])
        }
    
    def _compute_statistics(self, segments: List[Dict], techniques: Dict) -> Dict:
        """Compute shot statistics."""
        if not segments:
            return {
                'dominant_shot_type': 'static',
                'avg_shot_duration': 0,
                'shot_variety': 0,
                'num_techniques': 0
            }
        
        # Shot type distribution
        shot_counts = {}
        total_duration = 0
        
        for seg in segments:
            shot_type = seg['type']
            duration = seg['end_frame'] - seg['start_frame'] + 1
            shot_counts[shot_type] = shot_counts.get(shot_type, 0) + 1
            total_duration += duration
        
        dominant = max(shot_counts.items(), key=lambda x: x[1])[0]
        
        return {
            'dominant_shot_type': dominant,
            'avg_shot_duration': total_duration / len(segments) if segments else 0,
            'shot_variety': len(shot_counts) / len(segments) if segments else 0,
            'num_techniques': len(techniques.get('technique_instances', [])),
            'shot_distribution': shot_counts
        }