"""Visual anomaly detection using multi-modal statistical analysis and pattern recognition."""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from scipy import signal, stats
from scipy.ndimage import gaussian_filter, median_filter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.covariance import EllipticEnvelope
import warnings

from ..base import AdvancedFeature


class VisualAnomalyDetection(AdvancedFeature):
    """Detect visual anomalies using statistical outlier detection and pattern analysis."""
    
    FEATURE_NAME = 'visual_anomaly_detection'
    REQUIRED_ANALYSES = ['frame_diff', 'optical_flow_dense', 'edge_canny', 
                         'color_histogram', 'frequency_fft']
    
    def __init__(self, 
                 zscore_threshold: float = 3.0,
                 contamination: float = 0.1,
                 temporal_window: int = 30,
                 spatial_grid_size: int = 8,
                 use_multivariate: bool = True,
                 anomaly_persistence: int = 3,
                 feature_dimensions: int = 10):
        """
        Args:
            zscore_threshold: Z-score threshold for univariate anomaly detection
            contamination: Expected proportion of anomalies
            temporal_window: Window for temporal anomaly detection
            spatial_grid_size: Grid size for spatial anomaly analysis
            use_multivariate: Use multivariate anomaly detection
            anomaly_persistence: Min frames for persistent anomaly
            feature_dimensions: Number of PCA dimensions for features
        """
        super().__init__()
        self.zscore_threshold = zscore_threshold
        self.contamination = contamination
        self.temporal_window = temporal_window
        self.spatial_grid_size = spatial_grid_size
        self.use_multivariate = use_multivariate
        self.anomaly_persistence = anomaly_persistence
        self.feature_dimensions = feature_dimensions
    
    def _compute_advanced(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect visual anomalies using comprehensive statistical analysis.
        
        Returns:
            Dict with anomaly events, types, and patterns
        """
        frame_diffs = analysis_data['frame_diff'].data['pixel_diff']
        flow_field = analysis_data['optical_flow_dense'].data['flow_field']
        edge_maps = analysis_data['edge_canny'].data['edge_map']
        color_hists = analysis_data['color_histogram'].data['histogram']
        freq_data = analysis_data['frequency_fft'].data
        
        if len(frame_diffs) == 0:
            return self._empty_result()
        
        # Extract comprehensive features
        feature_matrix = self._extract_feature_matrix(
            frame_diffs, flow_field, edge_maps, color_hists, freq_data
        )
        
        # Detect temporal anomalies
        temporal_anomalies = self._detect_temporal_anomalies(feature_matrix)
        
        # Detect spatial anomalies
        spatial_anomalies = self._detect_spatial_anomalies(
            frame_diffs, flow_field, edge_maps
        )
        
        # Detect multivariate anomalies if enabled
        if self.use_multivariate:
            multivariate_anomalies = self._detect_multivariate_anomalies(feature_matrix)
        else:
            multivariate_anomalies = []
        
        # Merge and classify anomalies
        all_anomalies = self._merge_anomalies(
            temporal_anomalies, spatial_anomalies, multivariate_anomalies
        )
        
        # Detect anomaly patterns
        patterns = self._detect_anomaly_patterns(all_anomalies)
        
        # Analyze anomaly characteristics
        characteristics = self._analyze_anomaly_characteristics(
            all_anomalies, feature_matrix
        )
        
        return {
            'anomalies': all_anomalies,
            'patterns': patterns,
            'characteristics': characteristics,
            'statistics': self._compute_statistics(all_anomalies, len(frame_diffs))
        }
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result structure."""
        return {
            'anomalies': [],
            'patterns': {},
            'characteristics': {},
            'statistics': {
                'num_anomalies': 0,
                'anomaly_rate': 0.0,
                'dominant_type': 'none'
            }
        }
    
    def _extract_feature_matrix(self, frame_diffs: List[np.ndarray],
                               flow_field: List[np.ndarray],
                               edge_maps: List[np.ndarray],
                               color_hists: List[np.ndarray],
                               freq_data: Dict) -> np.ndarray:
        """Extract comprehensive feature matrix for anomaly detection."""
        features = []
        
        num_frames = min(len(frame_diffs), len(flow_field), len(edge_maps))
        
        for i in range(num_frames):
            frame_features = []
            
            # Frame difference features
            diff = frame_diffs[i]
            frame_features.extend([
                np.mean(diff),
                np.std(diff),
                np.percentile(diff, 95),
                stats.skew(diff.flatten()),
                stats.kurtosis(diff.flatten())
            ])
            
            # Optical flow features
            if i < len(flow_field):
                flow = flow_field[i]
                flow_mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                frame_features.extend([
                    np.mean(flow_mag),
                    np.std(flow_mag),
                    np.max(flow_mag),
                    np.percentile(flow_mag, 95),
                    self._compute_flow_complexity(flow)
                ])
            else:
                frame_features.extend([0] * 5)
            
            # Edge features
            if i < len(edge_maps):
                edges = edge_maps[i]
                frame_features.extend([
                    np.mean(edges > 0),
                    self._compute_edge_distribution_entropy(edges),
                    self._compute_edge_clustering(edges)
                ])
            else:
                frame_features.extend([0] * 3)
            
            # Color histogram features
            if i < len(color_hists):
                hist = color_hists[i].flatten()
                hist_norm = hist / (np.sum(hist) + 1e-10)
                frame_features.extend([
                    stats.entropy(hist_norm + 1e-10),
                    self._compute_color_diversity(hist_norm),
                    np.argmax(hist) / len(hist)  # Dominant color position
                ])
            else:
                frame_features.extend([0] * 3)
            
            # Frequency domain features
            if 'magnitude_spectrum' in freq_data and i < len(freq_data['magnitude_spectrum']):
                spectrum = freq_data['magnitude_spectrum'][i]
                frame_features.extend([
                    self._compute_frequency_centroid(spectrum),
                    self._compute_high_frequency_energy(spectrum)
                ])
            else:
                frame_features.extend([0] * 2)
            
            features.append(frame_features)
        
        return np.array(features)
    
    def _compute_flow_complexity(self, flow: np.ndarray) -> float:
        """Compute optical flow complexity metric."""
        # Compute flow gradients
        flow_grad_x = np.gradient(flow[..., 0], axis=1)
        flow_grad_y = np.gradient(flow[..., 1], axis=0)
        
        # Complexity is variance in gradients
        complexity = np.std(flow_grad_x) + np.std(flow_grad_y)
        return float(complexity)
    
    def _compute_edge_distribution_entropy(self, edges: np.ndarray) -> float:
        """Compute entropy of edge distribution."""
        h, w = edges.shape
        grid_h, grid_w = 4, 4
        
        densities = []
        for i in range(grid_h):
            for j in range(grid_w):
                region = edges[i*h//grid_h:(i+1)*h//grid_h,
                             j*w//grid_w:(j+1)*w//grid_w]
                densities.append(np.mean(region > 0))
        
        if densities:
            densities = np.array(densities) + 1e-10
            densities = densities / np.sum(densities)
            return float(stats.entropy(densities))
        return 0.0
    
    def _compute_edge_clustering(self, edges: np.ndarray) -> float:
        """Compute edge clustering coefficient."""
        # Downsample for efficiency
        h, w = edges.shape
        edges_small = edges[::4, ::4]
        
        if edges_small.size == 0:
            return 0.0
        
        # Compute local clustering
        kernel = np.ones((3, 3))
        kernel[1, 1] = 0
        
        edge_points = edges_small > 0
        if np.sum(edge_points) == 0:
            return 0.0
        
        neighbors = signal.convolve2d(edge_points.astype(float), kernel, mode='same')
        clustering = neighbors[edge_points] / 8.0
        
        return float(np.mean(clustering))
    
    def _compute_color_diversity(self, hist: np.ndarray) -> float:
        """Compute color diversity metric."""
        # Find peaks in histogram
        peaks = signal.find_peaks(hist, height=0.01)[0]
        
        if len(peaks) <= 1:
            return 0.0
        
        # Diversity is based on number and spread of peaks
        diversity = len(peaks) / len(hist)
        spread = (peaks[-1] - peaks[0]) / len(hist)
        
        return float(diversity * spread)
    
    def _compute_frequency_centroid(self, spectrum: np.ndarray) -> float:
        """Compute spectral centroid."""
        if spectrum.size == 0:
            return 0.0
        
        freqs = np.arange(len(spectrum))
        centroid = np.sum(freqs * spectrum) / (np.sum(spectrum) + 1e-10)
        return float(centroid / len(spectrum))
    
    def _compute_high_frequency_energy(self, spectrum: np.ndarray) -> float:
        """Compute high frequency energy ratio."""
        if spectrum.size == 0:
            return 0.0
        
        mid_point = len(spectrum) // 2
        high_freq_energy = np.sum(spectrum[mid_point:])
        total_energy = np.sum(spectrum)
        
        return float(high_freq_energy / (total_energy + 1e-10))
    
    def _detect_temporal_anomalies(self, features: np.ndarray) -> List[Dict]:
        """Detect temporal anomalies using statistical methods."""
        anomalies = []
        
        if len(features) < 3:
            return anomalies
        
        # Z-score based detection for each feature
        for feat_idx in range(features.shape[1]):
            feat_values = features[:, feat_idx]
            
            # Compute rolling statistics
            for i in range(len(feat_values)):
                window_start = max(0, i - self.temporal_window)
                window = feat_values[window_start:i+1]
                
                if len(window) > 3:
                    mean = np.mean(window[:-1])  # Exclude current
                    std = np.std(window[:-1]) + 1e-6
                    
                    z_score = abs((feat_values[i] - mean) / std)
                    
                    if z_score > self.zscore_threshold:
                        anomalies.append({
                            'frame': i,
                            'type': 'temporal',
                            'feature_idx': feat_idx,
                            'z_score': float(z_score),
                            'value': float(feat_values[i]),
                            'expected': float(mean)
                        })
        
        # Merge nearby anomalies
        merged = self._merge_temporal_anomalies(anomalies)
        
        return merged
    
    def _merge_temporal_anomalies(self, anomalies: List[Dict]) -> List[Dict]:
        """Merge nearby temporal anomalies."""
        if not anomalies:
            return []
        
        # Sort by frame
        sorted_anomalies = sorted(anomalies, key=lambda x: x['frame'])
        
        merged = []
        current_group = [sorted_anomalies[0]]
        
        for anom in sorted_anomalies[1:]:
            if anom['frame'] - current_group[-1]['frame'] <= 2:
                current_group.append(anom)
            else:
                # Process current group
                merged.append(self._summarize_anomaly_group(current_group))
                current_group = [anom]
        
        if current_group:
            merged.append(self._summarize_anomaly_group(current_group))
        
        return merged
    
    def _summarize_anomaly_group(self, group: List[Dict]) -> Dict:
        """Summarize a group of anomalies."""
        frames = [a['frame'] for a in group]
        z_scores = [a['z_score'] for a in group]
        
        return {
            'start_frame': min(frames),
            'end_frame': max(frames),
            'peak_frame': frames[np.argmax(z_scores)],
            'type': 'temporal',
            'severity': float(np.max(z_scores)),
            'persistence': len(frames),
            'affected_features': list(set(a.get('feature_idx', -1) for a in group))
        }
    
    def _detect_spatial_anomalies(self, frame_diffs: List[np.ndarray],
                                 flow_field: List[np.ndarray],
                                 edge_maps: List[np.ndarray]) -> List[Dict]:
        """Detect spatial anomalies in frames."""
        anomalies = []
        
        for i in range(min(len(frame_diffs), len(flow_field))):
            # Detect spatial outliers in frame difference
            diff_anomaly = self._detect_spatial_outliers(frame_diffs[i])
            
            # Detect motion anomalies
            if i < len(flow_field):
                motion_anomaly = self._detect_motion_outliers(flow_field[i])
            else:
                motion_anomaly = None
            
            # Detect edge anomalies
            if i < len(edge_maps):
                edge_anomaly = self._detect_edge_anomalies(edge_maps[i])
            else:
                edge_anomaly = None
            
            # Combine spatial anomalies
            if diff_anomaly or motion_anomaly or edge_anomaly:
                anomaly = {
                    'frame': i,
                    'type': 'spatial',
                    'regions': []
                }
                
                if diff_anomaly:
                    anomaly['regions'].extend(diff_anomaly)
                if motion_anomaly:
                    anomaly['regions'].extend(motion_anomaly)
                if edge_anomaly:
                    anomaly['regions'].extend(edge_anomaly)
                
                anomaly['severity'] = max(r['severity'] for r in anomaly['regions'])
                anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_spatial_outliers(self, image: np.ndarray) -> List[Dict]:
        """Detect spatial outliers in an image."""
        h, w = image.shape[:2] if len(image.shape) >= 2 else (0, 0)
        
        if h == 0 or w == 0:
            return []
        
        outliers = []
        
        # Divide into grid
        grid_h = min(self.spatial_grid_size, h)
        grid_w = min(self.spatial_grid_size, w)
        
        cell_h = h // grid_h
        cell_w = w // grid_w
        
        # Compute statistics for each cell
        cell_values = []
        cell_positions = []
        
        for i in range(grid_h):
            for j in range(grid_w):
                cell = image[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                if cell.size > 0:
                    cell_values.append(np.mean(cell))
                    cell_positions.append((i, j))
        
        if len(cell_values) > 3:
            cell_values = np.array(cell_values)
            mean = np.mean(cell_values)
            std = np.std(cell_values) + 1e-6
            
            for idx, val in enumerate(cell_values):
                z_score = abs((val - mean) / std)
                
                if z_score > self.zscore_threshold:
                    i, j = cell_positions[idx]
                    outliers.append({
                        'bbox': (j*cell_w, i*cell_h, (j+1)*cell_w, (i+1)*cell_h),
                        'severity': float(z_score),
                        'anomaly_type': 'intensity_outlier'
                    })
        
        return outliers
    
    def _detect_motion_outliers(self, flow: np.ndarray) -> List[Dict]:
        """Detect motion outliers in optical flow."""
        if flow.size == 0:
            return []
        
        h, w = flow.shape[:2]
        flow_mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        
        # Find high motion regions
        threshold = np.percentile(flow_mag, 95)
        high_motion = flow_mag > threshold
        
        # Apply morphological operations
        from scipy.ndimage import binary_dilation, label
        high_motion = binary_dilation(high_motion, iterations=2)
        
        # Find connected components
        labeled, num_features = label(high_motion)
        
        outliers = []
        for i in range(1, num_features + 1):
            mask = labeled == i
            
            if np.sum(mask) > h * w * 0.001:  # Min size threshold
                coords = np.where(mask)
                y_min, y_max = coords[0].min(), coords[0].max()
                x_min, x_max = coords[1].min(), coords[1].max()
                
                region_flow = flow_mag[mask]
                severity = np.mean(region_flow) / (np.mean(flow_mag) + 1e-6)
                
                outliers.append({
                    'bbox': (int(x_min), int(y_min), int(x_max), int(y_max)),
                    'severity': float(severity),
                    'anomaly_type': 'motion_outlier'
                })
        
        return outliers
    
    def _detect_edge_anomalies(self, edges: np.ndarray) -> List[Dict]:
        """Detect anomalies in edge patterns."""
        if edges.size == 0:
            return []
        
        h, w = edges.shape
        
        # Compute local edge density
        kernel_size = 20
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size**2)
        edge_density = signal.convolve2d(edges.astype(float), kernel, mode='same')
        
        # Find anomalous density regions
        mean_density = np.mean(edge_density)
        std_density = np.std(edge_density) + 1e-6
        
        anomaly_mask = np.abs(edge_density - mean_density) > self.zscore_threshold * std_density
        
        # Find connected components
        from scipy.ndimage import label
        labeled, num_features = label(anomaly_mask)
        
        outliers = []
        for i in range(1, num_features + 1):
            mask = labeled == i
            
            if np.sum(mask) > h * w * 0.002:  # Min size
                coords = np.where(mask)
                y_min, y_max = coords[0].min(), coords[0].max()
                x_min, x_max = coords[1].min(), coords[1].max()
                
                region_density = edge_density[mask]
                severity = np.max(np.abs(region_density - mean_density)) / std_density
                
                outliers.append({
                    'bbox': (int(x_min), int(y_min), int(x_max), int(y_max)),
                    'severity': float(severity),
                    'anomaly_type': 'edge_anomaly'
                })
        
        return outliers
    
    def _detect_multivariate_anomalies(self, features: np.ndarray) -> List[Dict]:
        """Detect multivariate anomalies using robust covariance."""
        if len(features) < 10:
            return []
        
        anomalies = []
        
        try:
            # Standardize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Reduce dimensions if needed
            if features_scaled.shape[1] > self.feature_dimensions:
                pca = PCA(n_components=self.feature_dimensions)
                features_scaled = pca.fit_transform(features_scaled)
            
            # Robust covariance estimation
            clf = EllipticEnvelope(contamination=self.contamination)
            clf.fit(features_scaled)
            
            # Predict anomalies
            predictions = clf.predict(features_scaled)
            scores = clf.score_samples(features_scaled)
            
            for i, (pred, score) in enumerate(zip(predictions, scores)):
                if pred == -1:  # Anomaly
                    anomalies.append({
                        'frame': i,
                        'type': 'multivariate',
                        'severity': float(abs(score)),
                        'anomaly_score': float(-score)
                    })
        except Exception as e:
            warnings.warn(f"Multivariate detection failed: {e}")
        
        return anomalies
    
    def _merge_anomalies(self, temporal: List[Dict], spatial: List[Dict],
                        multivariate: List[Dict]) -> List[Dict]:
        """Merge different types of anomalies."""
        all_anomalies = {}
        
        # Add temporal anomalies
        for anom in temporal:
            frame = anom.get('peak_frame', anom.get('frame', -1))
            if frame not in all_anomalies:
                all_anomalies[frame] = {
                    'frame': frame,
                    'types': [],
                    'severity': 0,
                    'details': {}
                }
            all_anomalies[frame]['types'].append('temporal')
            all_anomalies[frame]['severity'] = max(all_anomalies[frame]['severity'],
                                                  anom.get('severity', 0))
            all_anomalies[frame]['details']['temporal'] = anom
        
        # Add spatial anomalies
        for anom in spatial:
            frame = anom['frame']
            if frame not in all_anomalies:
                all_anomalies[frame] = {
                    'frame': frame,
                    'types': [],
                    'severity': 0,
                    'details': {}
                }
            all_anomalies[frame]['types'].append('spatial')
            all_anomalies[frame]['severity'] = max(all_anomalies[frame]['severity'],
                                                  anom.get('severity', 0))
            all_anomalies[frame]['details']['spatial'] = anom
        
        # Add multivariate anomalies
        for anom in multivariate:
            frame = anom['frame']
            if frame not in all_anomalies:
                all_anomalies[frame] = {
                    'frame': frame,
                    'types': [],
                    'severity': 0,
                    'details': {}
                }
            all_anomalies[frame]['types'].append('multivariate')
            all_anomalies[frame]['severity'] = max(all_anomalies[frame]['severity'],
                                                  anom.get('severity', 0))
            all_anomalies[frame]['details']['multivariate'] = anom
        
        # Convert to list and classify
        merged = []
        for frame, data in all_anomalies.items():
            data['anomaly_class'] = self._classify_anomaly(data)
            merged.append(data)
        
        return sorted(merged, key=lambda x: x['frame'])
    
    def _classify_anomaly(self, anomaly: Dict) -> str:
        """Classify the type of anomaly."""
        types = anomaly['types']
        details = anomaly.get('details', {})
        
        # Multi-modal anomaly
        if len(types) >= 2:
            if 'temporal' in types and 'spatial' in types:
                return 'complex_anomaly'
            elif 'multivariate' in types:
                return 'statistical_outlier'
        
        # Single-modal classification
        if 'spatial' in types and 'spatial' in details:
            spatial_details = details['spatial']
            if 'regions' in spatial_details:
                region_types = [r['anomaly_type'] for r in spatial_details['regions']]
                if 'motion_outlier' in region_types:
                    return 'motion_anomaly'
                elif 'edge_anomaly' in region_types:
                    return 'structural_anomaly'
                else:
                    return 'intensity_anomaly'
        
        if 'temporal' in types:
            return 'temporal_anomaly'
        
        return 'unknown_anomaly'
    
    def _detect_anomaly_patterns(self, anomalies: List[Dict]) -> Dict:
        """Detect patterns in anomalies."""
        patterns = {
            'has_periodic_anomalies': False,
            'has_clustered_anomalies': False,
            'has_persistent_anomaly': False,
            'anomaly_trend': 'stable',
            'cluster_locations': []
        }
        
        if len(anomalies) < 2:
            return patterns
        
        frames = [a['frame'] for a in anomalies]
        
        # Check for periodic anomalies
        if len(frames) > 3:
            intervals = np.diff(frames)
            if len(intervals) > 2:
                # Check if intervals are regular
                interval_std = np.std(intervals)
                interval_mean = np.mean(intervals)
                if interval_mean > 0 and interval_std / interval_mean < 0.3:
                    patterns['has_periodic_anomalies'] = True
        
        # Check for clustered anomalies
        for i in range(len(frames) - 2):
            if frames[i+2] - frames[i] < 10:  # 3 anomalies within 10 frames
                patterns['has_clustered_anomalies'] = True
                patterns['cluster_locations'].append({
                    'start': frames[i],
                    'end': frames[i+2],
                    'num_anomalies': 3
                })
        
        # Check for persistent anomaly
        for i in range(len(frames) - 1):
            if frames[i+1] - frames[i] == 1:
                # Check if continues
                consecutive = 2
                for j in range(i+2, len(frames)):
                    if frames[j] - frames[j-1] == 1:
                        consecutive += 1
                    else:
                        break
                
                if consecutive >= self.anomaly_persistence:
                    patterns['has_persistent_anomaly'] = True
                    break
        
        # Detect trend
        if len(frames) > 5:
            first_half = frames[:len(frames)//2]
            second_half = frames[len(frames)//2:]
            
            first_density = len(first_half) / (first_half[-1] - first_half[0] + 1)
            second_density = len(second_half) / (second_half[-1] - second_half[0] + 1)
            
            if second_density > first_density * 1.5:
                patterns['anomaly_trend'] = 'increasing'
            elif first_density > second_density * 1.5:
                patterns['anomaly_trend'] = 'decreasing'
        
        return patterns
    
    def _analyze_anomaly_characteristics(self, anomalies: List[Dict],
                                        features: np.ndarray) -> Dict:
        """Analyze characteristics of detected anomalies."""
        if not anomalies:
            return {
                'avg_severity': 0,
                'max_severity': 0,
                'most_affected_feature': 'none'
            }
        
        severities = [a['severity'] for a in anomalies]
        
        # Find most affected features
        feature_counts = {}
        for anom in anomalies:
            if 'temporal' in anom.get('details', {}):
                temporal = anom['details']['temporal']
                if 'affected_features' in temporal:
                    for feat_idx in temporal['affected_features']:
                        feature_counts[feat_idx] = feature_counts.get(feat_idx, 0) + 1
        
        feature_names = [
            'intensity_mean', 'intensity_std', 'intensity_p95', 'intensity_skew', 'intensity_kurt',
            'motion_mean', 'motion_std', 'motion_max', 'motion_p95', 'motion_complexity',
            'edge_density', 'edge_entropy', 'edge_clustering',
            'color_entropy', 'color_diversity', 'dominant_color',
            'freq_centroid', 'high_freq_energy'
        ]
        
        most_affected = 'none'
        if feature_counts:
            most_affected_idx = max(feature_counts.items(), key=lambda x: x[1])[0]
            if 0 <= most_affected_idx < len(feature_names):
                most_affected = feature_names[most_affected_idx]
        
        return {
            'avg_severity': float(np.mean(severities)),
            'max_severity': float(np.max(severities)),
            'severity_std': float(np.std(severities)),
            'most_affected_feature': most_affected,
            'feature_impact_distribution': feature_counts
        }
    
    def _compute_statistics(self, anomalies: List[Dict], total_frames: int) -> Dict:
        """Compute anomaly statistics."""
        if not anomalies:
            return {
                'num_anomalies': 0,
                'anomaly_rate': 0.0,
                'dominant_type': 'none',
                'avg_severity': 0.0
            }
        
        # Type distribution
        type_counts = {}
        for anom in anomalies:
            anom_class = anom.get('anomaly_class', 'unknown')
            type_counts[anom_class] = type_counts.get(anom_class, 0) + 1
        
        dominant = max(type_counts.items(), key=lambda x: x[1])[0]
        
        return {
            'num_anomalies': len(anomalies),
            'anomaly_rate': len(anomalies) / total_frames if total_frames > 0 else 0,
            'dominant_type': dominant,
            'avg_severity': float(np.mean([a['severity'] for a in anomalies])),
            'type_distribution': type_counts
        }