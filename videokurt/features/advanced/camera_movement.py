"""Camera movement classification (pan, zoom, tilt, rotate)."""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from scipy import signal
from scipy.ndimage import gaussian_filter

from ..base import AdvancedFeature


class CameraMovement(AdvancedFeature):
    """Classify camera movement patterns from optical flow with robust algorithms."""
    
    FEATURE_NAME = 'camera_movement'
    REQUIRED_ANALYSES = ['optical_flow_dense']
    
    def __init__(self, 
                 motion_threshold: float = 0.5,
                 consistency_threshold: float = 0.6,
                 smoothing_window: int = 5,
                 min_movement_frames: int = 3,
                 grid_size: int = 16):
        """
        Args:
            motion_threshold: Minimum motion magnitude to consider camera movement
            consistency_threshold: Flow consistency required for classification
            smoothing_window: Temporal smoothing window size
            min_movement_frames: Minimum consecutive frames for valid movement
            grid_size: Grid size for spatial sampling
        """
        super().__init__()
        self.motion_threshold = motion_threshold
        self.consistency_threshold = consistency_threshold
        self.smoothing_window = smoothing_window
        self.min_movement_frames = min_movement_frames
        self.grid_size = grid_size
    
    def _compute_advanced(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Classify camera movements using robust optical flow analysis.
        
        Returns:
            Dict with camera movement events, types, and confidence scores
        """
        flow_field = analysis_data['optical_flow_dense'].data['flow_field']
        
        if len(flow_field) == 0:
            return {
                'movements': [],
                'movement_timeline': [],
                'movement_counts': {},
                'dominant_movement': 'static',
                'camera_stability': 1.0
            }
        
        # Analyze each frame
        frame_movements = []
        confidence_scores = []
        
        for i, flow in enumerate(flow_field):
            movement, confidence = self._analyze_single_frame(flow)
            frame_movements.append(movement)
            confidence_scores.append(confidence)
        
        # Apply temporal smoothing
        smoothed_movements = self._temporal_smoothing(frame_movements, confidence_scores)
        
        # Extract movement segments
        movement_segments = self._extract_segments(smoothed_movements, confidence_scores)
        
        # Filter short movements
        filtered_segments = [s for s in movement_segments 
                           if s['duration'] >= self.min_movement_frames]
        
        # Compute statistics
        movement_counts = {}
        total_frames = len(frame_movements)
        movement_frames = 0
        
        for seg in filtered_segments:
            movement_counts[seg['type']] = movement_counts.get(seg['type'], 0) + 1
            movement_frames += seg['duration']
        
        camera_stability = 1.0 - (movement_frames / total_frames) if total_frames > 0 else 1.0
        
        # Identify dominant movement
        if movement_counts:
            dominant = max(movement_counts.items(), key=lambda x: x[1])[0]
        else:
            dominant = 'static'
        
        # Create movement timeline with confidence
        movement_timeline = []
        for i, (mov, conf) in enumerate(zip(smoothed_movements, confidence_scores)):
            movement_timeline.append({
                'frame': i,
                'type': mov,
                'confidence': conf
            })
        
        return {
            'movements': filtered_segments,
            'movement_timeline': movement_timeline,
            'movement_counts': movement_counts,
            'dominant_movement': dominant,
            'camera_stability': float(camera_stability),
            'avg_confidence': float(np.mean(confidence_scores)) if confidence_scores else 0
        }
    
    def _analyze_single_frame(self, flow: np.ndarray) -> Tuple[str, float]:
        """Analyze optical flow for a single frame with confidence scoring."""
        h, w = flow.shape[:2]
        
        # Sample flow field on a grid for efficiency
        grid_h = min(self.grid_size, h)
        grid_w = min(self.grid_size, w)
        
        y_samples = np.linspace(0, h-1, grid_h, dtype=int)
        x_samples = np.linspace(0, w-1, grid_w, dtype=int)
        
        sampled_flow = flow[np.ix_(y_samples, x_samples)]
        
        # Compute flow statistics
        flow_x = sampled_flow[..., 0]
        flow_y = sampled_flow[..., 1]
        magnitudes = np.sqrt(flow_x**2 + flow_y**2)
        
        avg_magnitude = np.mean(magnitudes)
        
        # Check if there's significant motion
        if avg_magnitude < self.motion_threshold:
            return 'static', 1.0
        
        # Compute motion models and their fit scores
        models = {}
        
        # 1. Translation model (pan/tilt)
        translation_score, translation_type = self._fit_translation_model(
            flow_x, flow_y, magnitudes
        )
        models[translation_type] = translation_score
        
        # 2. Zoom model
        zoom_score, zoom_type = self._fit_zoom_model(
            sampled_flow, x_samples, y_samples, w, h
        )
        models[zoom_type] = zoom_score
        
        # 3. Rotation model
        rotation_score, rotation_type = self._fit_rotation_model(
            sampled_flow, x_samples, y_samples, w, h
        )
        models[rotation_type] = rotation_score
        
        # 4. Combined/complex motion
        if max(models.values()) < self.consistency_threshold:
            # Check for combined movements
            combined_score = self._check_combined_movement(models)
            if combined_score > self.consistency_threshold:
                return 'complex', combined_score
        
        # Select best fitting model
        best_model = max(models.items(), key=lambda x: x[1])
        
        if best_model[1] >= self.consistency_threshold:
            return best_model[0], best_model[1]
        else:
            return 'complex', best_model[1]
    
    def _fit_translation_model(self, flow_x: np.ndarray, flow_y: np.ndarray, 
                               magnitudes: np.ndarray) -> Tuple[float, str]:
        """Fit and score translation (pan/tilt) model."""
        avg_flow_x = np.mean(flow_x)
        avg_flow_y = np.mean(flow_y)
        
        # Compute consistency (low variance = high consistency)
        if np.mean(magnitudes) > 0:
            consistency_x = 1.0 - np.std(flow_x) / (np.abs(avg_flow_x) + 1e-6)
            consistency_y = 1.0 - np.std(flow_y) / (np.abs(avg_flow_y) + 1e-6)
            
            # Determine dominant direction
            if abs(avg_flow_x) > abs(avg_flow_y):
                # Horizontal movement
                score = consistency_x * (abs(avg_flow_x) / (abs(avg_flow_x) + abs(avg_flow_y)))
                movement_type = 'pan_right' if avg_flow_x > 0 else 'pan_left'
            else:
                # Vertical movement
                score = consistency_y * (abs(avg_flow_y) / (abs(avg_flow_x) + abs(avg_flow_y)))
                movement_type = 'tilt_down' if avg_flow_y > 0 else 'tilt_up'
            
            # Penalize if motion is not uniform
            uniformity = 1.0 - np.std(magnitudes) / (np.mean(magnitudes) + 1e-6)
            score *= uniformity
            
            return max(0, min(1, score)), movement_type
        
        return 0.0, 'static'
    
    def _fit_zoom_model(self, flow: np.ndarray, x_samples: np.ndarray, 
                       y_samples: np.ndarray, w: int, h: int) -> Tuple[float, str]:
        """Fit and score zoom model using radial flow analysis."""
        # Define multiple potential zoom centers
        centers = [
            (w//2, h//2),  # Frame center
            (w//3, h//2),  # Rule of thirds points
            (2*w//3, h//2),
            (w//2, h//3),
            (w//2, 2*h//3)
        ]
        
        best_score = 0
        best_zoom_type = 'zoom_in'
        
        for center_x, center_y in centers:
            # Compute radial flow for this center
            radial_scores = []
            radial_sum = 0
            
            for i, y in enumerate(y_samples):
                for j, x in enumerate(x_samples):
                    dx = x - center_x
                    dy = y - center_y
                    r = np.sqrt(dx**2 + dy**2) + 1e-6
                    
                    # Normalized radial vector
                    rx = dx / r
                    ry = dy / r
                    
                    # Radial component of flow
                    radial_flow = flow[i, j, 0] * rx + flow[i, j, 1] * ry
                    
                    # Expected radial flow for perfect zoom
                    expected_magnitude = np.sqrt(flow[i, j, 0]**2 + flow[i, j, 1]**2)
                    
                    if expected_magnitude > 0:
                        # Score based on how well flow aligns with radial direction
                        alignment = abs(radial_flow) / expected_magnitude
                        radial_scores.append(alignment)
                        radial_sum += radial_flow
            
            if radial_scores:
                # Average alignment score
                score = np.mean(radial_scores)
                
                # Penalize if radial flow is not consistent
                radial_consistency = 1.0 - np.std(radial_scores)
                score *= max(0, radial_consistency)
                
                if score > best_score:
                    best_score = score
                    best_zoom_type = 'zoom_in' if radial_sum > 0 else 'zoom_out'
        
        return best_score, best_zoom_type
    
    def _fit_rotation_model(self, flow: np.ndarray, x_samples: np.ndarray,
                           y_samples: np.ndarray, w: int, h: int) -> Tuple[float, str]:
        """Fit and score rotation model using tangential flow analysis."""
        center_x, center_y = w // 2, h // 2
        
        tangential_scores = []
        tangential_sum = 0
        
        for i, y in enumerate(y_samples):
            for j, x in enumerate(x_samples):
                dx = x - center_x
                dy = y - center_y
                r = np.sqrt(dx**2 + dy**2) + 1e-6
                
                # Normalized tangential vector (perpendicular to radial)
                tx = -dy / r
                ty = dx / r
                
                # Tangential component of flow
                tangential_flow = flow[i, j, 0] * tx + flow[i, j, 1] * ty
                
                # Expected magnitude
                expected_magnitude = np.sqrt(flow[i, j, 0]**2 + flow[i, j, 1]**2)
                
                if expected_magnitude > 0:
                    # Score based on how well flow aligns with tangential direction
                    alignment = abs(tangential_flow) / expected_magnitude
                    tangential_scores.append(alignment)
                    tangential_sum += tangential_flow
        
        if tangential_scores:
            score = np.mean(tangential_scores)
            
            # Penalize if rotation is not consistent across radius
            consistency = 1.0 - np.std(tangential_scores)
            score *= max(0, consistency)
            
            rotation_type = 'rotate_cw' if tangential_sum > 0 else 'rotate_ccw'
            return score, rotation_type
        
        return 0.0, 'static'
    
    def _check_combined_movement(self, models: Dict[str, float]) -> float:
        """Check for combined movements (e.g., zoom + pan)."""
        # Check common combinations
        combinations = [
            (['pan_left', 'pan_right'], ['zoom_in', 'zoom_out']),  # Dolly zoom
            (['tilt_up', 'tilt_down'], ['zoom_in', 'zoom_out']),   # Crane zoom
            (['pan_left', 'pan_right'], ['rotate_cw', 'rotate_ccw'])  # Arc shot
        ]
        
        max_combined_score = 0
        
        for combo in combinations:
            scores = []
            for movement_group in combo:
                group_scores = [models.get(m, 0) for m in movement_group]
                if group_scores:
                    scores.append(max(group_scores))
            
            if len(scores) == 2 and all(s > 0.3 for s in scores):
                combined_score = np.mean(scores)
                max_combined_score = max(max_combined_score, combined_score)
        
        return max_combined_score
    
    def _temporal_smoothing(self, movements: List[str], 
                           confidences: List[float]) -> List[str]:
        """Apply temporal smoothing to reduce noise in classifications."""
        if len(movements) <= self.smoothing_window:
            return movements
        
        smoothed = movements.copy()
        
        for i in range(len(movements)):
            window_start = max(0, i - self.smoothing_window // 2)
            window_end = min(len(movements), i + self.smoothing_window // 2 + 1)
            
            window = movements[window_start:window_end]
            window_conf = confidences[window_start:window_end]
            
            # Weighted voting by confidence
            movement_scores = {}
            for mov, conf in zip(window, window_conf):
                movement_scores[mov] = movement_scores.get(mov, 0) + conf
            
            # Select most confident movement in window
            if movement_scores:
                smoothed[i] = max(movement_scores.items(), key=lambda x: x[1])[0]
        
        return smoothed
    
    def _extract_segments(self, movements: List[str], 
                         confidences: List[float]) -> List[Dict]:
        """Extract continuous movement segments."""
        if not movements:
            return []
        
        segments = []
        current_segment = {
            'type': movements[0],
            'start_frame': 0,
            'end_frame': 0,
            'confidence': confidences[0],
            'duration': 1
        }
        
        for i in range(1, len(movements)):
            if movements[i] == current_segment['type']:
                # Extend current segment
                current_segment['end_frame'] = i
                current_segment['duration'] += 1
                current_segment['confidence'] = max(current_segment['confidence'], 
                                                   confidences[i])
            else:
                # Save current segment and start new one
                if current_segment['type'] != 'static':
                    segments.append(current_segment.copy())
                
                current_segment = {
                    'type': movements[i],
                    'start_frame': i,
                    'end_frame': i,
                    'confidence': confidences[i],
                    'duration': 1
                }
        
        # Add final segment
        if current_segment['type'] != 'static':
            segments.append(current_segment)
        
        return segments