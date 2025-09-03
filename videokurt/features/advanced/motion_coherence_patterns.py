"""Motion coherence pattern detection using advanced flow field analysis."""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from scipy import signal, stats
from scipy.ndimage import label, gaussian_filter
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.decomposition import PCA
import warnings

from ..base import AdvancedFeature


class MotionCoherencePatterns(AdvancedFeature):
    """Analyze motion coherence patterns using flow field segmentation and clustering."""
    
    FEATURE_NAME = 'motion_coherence_patterns'
    REQUIRED_ANALYSES = ['optical_flow_dense', 'optical_flow_sparse', 'edge_canny']
    
    def __init__(self, 
                 coherence_threshold: float = 0.7,
                 block_size: int = 16,
                 min_region_size: float = 0.01,
                 clustering_threshold: float = 0.3,
                 temporal_window: int = 10,
                 use_hierarchical_analysis: bool = True,
                 motion_significance_threshold: float = 0.5):
        """
        Args:
            coherence_threshold: Threshold for coherent motion
            block_size: Block size for local coherence analysis
            min_region_size: Minimum region size (fraction of frame)
            clustering_threshold: Threshold for motion clustering
            temporal_window: Window for temporal coherence
            use_hierarchical_analysis: Use multi-scale analysis
            motion_significance_threshold: Threshold for significant motion
        """
        super().__init__()
        self.coherence_threshold = coherence_threshold
        self.block_size = block_size
        self.min_region_size = min_region_size
        self.clustering_threshold = clustering_threshold
        self.temporal_window = temporal_window
        self.use_hierarchical_analysis = use_hierarchical_analysis
        self.motion_significance_threshold = motion_significance_threshold
    
    def _compute_advanced(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze motion coherence patterns using comprehensive methods.
        
        Returns:
            Dict with coherence patterns, regions, and dynamics
        """
        dense_flow = analysis_data['optical_flow_dense'].data['flow_field']
        sparse_data = analysis_data['optical_flow_sparse'].data
        edge_maps = analysis_data['edge_canny'].data['edge_map']
        
        if len(dense_flow) == 0:
            return self._empty_result()
        
        # Compute local and global coherence
        coherence_analysis = self._analyze_coherence(dense_flow)
        
        # Segment motion into coherent regions
        motion_regions = self._segment_motion_regions(dense_flow, edge_maps)
        
        # Analyze temporal coherence
        temporal_coherence = self._analyze_temporal_coherence(dense_flow)
        
        # Detect coherence patterns
        coherence_patterns = self._detect_coherence_patterns(
            coherence_analysis, motion_regions
        )
        
        # Analyze motion organization
        organization = self._analyze_motion_organization(dense_flow, motion_regions)
        
        # Detect specific coherence phenomena
        phenomena = self._detect_coherence_phenomena(
            dense_flow, coherence_analysis, temporal_coherence
        )
        
        # Compute coherence dynamics
        dynamics = self._compute_coherence_dynamics(
            coherence_analysis, temporal_coherence
        )
        
        return {
            'coherence_analysis': coherence_analysis,
            'motion_regions': motion_regions,
            'temporal_coherence': temporal_coherence,
            'coherence_patterns': coherence_patterns,
            'organization': organization,
            'phenomena': phenomena,
            'dynamics': dynamics,
            'statistics': self._compute_statistics(
                coherence_analysis, coherence_patterns
            )
        }
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result structure."""
        return {
            'coherence_analysis': {},
            'motion_regions': [],
            'temporal_coherence': {},
            'coherence_patterns': [],
            'organization': {},
            'phenomena': {},
            'dynamics': {},
            'statistics': {
                'mean_coherence': 0.0,
                'dominant_pattern': 'none',
                'num_coherent_regions': 0
            }
        }
    
    def _analyze_coherence(self, flow_field: List[np.ndarray]) -> Dict:
        """Analyze motion coherence at multiple scales."""
        analysis = {
            'local_coherence': [],
            'global_coherence': [],
            'spatial_coherence_maps': [],
            'directional_coherence': [],
            'magnitude_coherence': []
        }
        
        for flow in flow_field:
            if flow.size == 0:
                continue
            
            # Local coherence
            local_coh = self._compute_local_coherence(flow)
            analysis['local_coherence'].append(local_coh['mean_coherence'])
            analysis['spatial_coherence_maps'].append(local_coh['coherence_map'])
            
            # Global coherence
            global_coh = self._compute_global_coherence(flow)
            analysis['global_coherence'].append(global_coh)
            
            # Directional coherence
            dir_coh = self._compute_directional_coherence(flow)
            analysis['directional_coherence'].append(dir_coh)
            
            # Magnitude coherence
            mag_coh = self._compute_magnitude_coherence(flow)
            analysis['magnitude_coherence'].append(mag_coh)
        
        return analysis
    
    def _compute_local_coherence(self, flow: np.ndarray) -> Dict:
        """Compute local motion coherence using block analysis."""
        h, w = flow.shape[:2]
        
        # Adjust block size if needed
        block_h = min(self.block_size, h // 4)
        block_w = min(self.block_size, w // 4)
        
        coherence_map = np.zeros((h // block_h, w // block_w))
        
        for i in range(0, h - block_h + 1, block_h):
            for j in range(0, w - block_w + 1, block_w):
                block = flow[i:i+block_h, j:j+block_w]
                
                # Compute block coherence
                coherence = self._compute_block_coherence(block)
                coherence_map[i // block_h, j // block_w] = coherence
        
        return {
            'coherence_map': coherence_map,
            'mean_coherence': float(np.mean(coherence_map)),
            'std_coherence': float(np.std(coherence_map))
        }
    
    def _compute_block_coherence(self, block: np.ndarray) -> float:
        """Compute coherence for a single block."""
        if block.size == 0:
            return 0.0
        
        # Compute mean flow
        mean_flow = np.mean(block, axis=(0, 1))
        magnitude = np.linalg.norm(mean_flow)
        
        if magnitude < self.motion_significance_threshold:
            return 0.0
        
        # Method 1: Variance-based coherence
        flow_vectors = block.reshape(-1, 2)
        
        # Normalize vectors
        norms = np.linalg.norm(flow_vectors, axis=1)
        valid = norms > 0
        
        if np.sum(valid) < 2:
            return 0.0
        
        normalized = flow_vectors[valid] / norms[valid, np.newaxis]
        
        # Compute pairwise similarity
        similarities = np.dot(normalized, normalized.T)
        
        # Coherence is mean similarity
        coherence_var = np.mean(similarities[np.triu_indices_from(similarities, k=1)])
        
        # Method 2: Direction consistency
        angles = np.arctan2(flow_vectors[valid, 1], flow_vectors[valid, 0])
        angle_std = np.std(angles)
        coherence_dir = np.exp(-angle_std)
        
        # Method 3: Magnitude consistency
        mag_cv = np.std(norms[valid]) / (np.mean(norms[valid]) + 1e-6)
        coherence_mag = 1.0 / (1.0 + mag_cv)
        
        # Combined coherence
        coherence = (coherence_var + coherence_dir + coherence_mag) / 3.0
        
        return float(coherence)
    
    def _compute_global_coherence(self, flow: np.ndarray) -> float:
        """Compute global motion coherence."""
        if flow.size == 0:
            return 0.0
        
        # Flatten flow vectors
        flow_vectors = flow.reshape(-1, 2)
        
        # Filter out small motions
        magnitudes = np.linalg.norm(flow_vectors, axis=1)
        significant = magnitudes > self.motion_significance_threshold
        
        if np.sum(significant) < 2:
            return 0.0
        
        significant_flows = flow_vectors[significant]
        
        # Compute principal components
        pca = PCA(n_components=2)
        pca.fit(significant_flows)
        
        # Coherence based on explained variance ratio
        coherence = pca.explained_variance_ratio_[0]
        
        return float(coherence)
    
    def _compute_directional_coherence(self, flow: np.ndarray) -> float:
        """Compute directional coherence of motion."""
        if flow.size == 0:
            return 0.0
        
        # Compute angles
        angles = np.arctan2(flow[..., 1], flow[..., 0])
        
        # Filter by magnitude
        magnitudes = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        significant = magnitudes > self.motion_significance_threshold
        
        if np.sum(significant) < 2:
            return 0.0
        
        significant_angles = angles[significant]
        
        # Compute circular mean and variance
        mean_angle = np.arctan2(
            np.mean(np.sin(significant_angles)),
            np.mean(np.cos(significant_angles))
        )
        
        # Angular deviation
        deviations = np.abs(np.arctan2(
            np.sin(significant_angles - mean_angle),
            np.cos(significant_angles - mean_angle)
        ))
        
        # Coherence based on concentration
        coherence = np.exp(-np.mean(deviations))
        
        return float(coherence)
    
    def _compute_magnitude_coherence(self, flow: np.ndarray) -> float:
        """Compute magnitude coherence of motion."""
        if flow.size == 0:
            return 0.0
        
        magnitudes = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        
        # Filter significant motion
        significant = magnitudes > self.motion_significance_threshold
        
        if np.sum(significant) < 2:
            return 0.0
        
        significant_mags = magnitudes[significant]
        
        # Coherence based on coefficient of variation
        cv = np.std(significant_mags) / (np.mean(significant_mags) + 1e-6)
        coherence = 1.0 / (1.0 + cv)
        
        return float(coherence)
    
    def _segment_motion_regions(self, flow_field: List[np.ndarray],
                               edge_maps: List[np.ndarray]) -> List[Dict]:
        """Segment flow field into coherent motion regions."""
        regions = []
        
        for frame_idx, flow in enumerate(flow_field[:min(len(flow_field), 100)]):  # Limit processing
            if flow.size == 0:
                continue
            
            h, w = flow.shape[:2]
            
            # Create feature vectors for clustering
            features = self._extract_motion_features(flow)
            
            if features is None or len(features) < 10:
                continue
            
            # Cluster motion vectors
            clusters = self._cluster_motion(features, h, w)
            
            # Extract region properties
            frame_regions = []
            for cluster_id in np.unique(clusters):
                if cluster_id < 0:  # Skip noise
                    continue
                
                mask = clusters == cluster_id
                region_size = np.sum(mask) / (h * w)
                
                if region_size < self.min_region_size:
                    continue
                
                # Compute region properties
                region_flow = flow[mask]
                mean_flow = np.mean(region_flow, axis=0)
                
                # Find bounding box
                coords = np.where(mask)
                bbox = (
                    int(np.min(coords[1])),
                    int(np.min(coords[0])),
                    int(np.max(coords[1])),
                    int(np.max(coords[0]))
                )
                
                # Compute coherence within region
                region_coherence = self._compute_region_coherence(region_flow)
                
                frame_regions.append({
                    'cluster_id': int(cluster_id),
                    'bbox': bbox,
                    'size': float(region_size),
                    'mean_flow': mean_flow.tolist(),
                    'coherence': region_coherence,
                    'centroid': (int(np.mean(coords[1])), int(np.mean(coords[0])))
                })
            
            if frame_regions:
                regions.append({
                    'frame': frame_idx,
                    'num_regions': len(frame_regions),
                    'regions': frame_regions
                })
        
        return regions
    
    def _extract_motion_features(self, flow: np.ndarray) -> Optional[np.ndarray]:
        """Extract features for motion clustering."""
        h, w = flow.shape[:2]
        
        # Downsample for efficiency
        step = max(1, min(h, w) // 64)
        flow_sampled = flow[::step, ::step]
        
        if flow_sampled.size == 0:
            return None
        
        # Create feature vectors [x, y, u, v, magnitude, angle]
        h_s, w_s = flow_sampled.shape[:2]
        
        y_coords, x_coords = np.mgrid[0:h_s, 0:w_s]
        
        features = np.column_stack([
            x_coords.ravel() / w_s,  # Normalized position
            y_coords.ravel() / h_s,
            flow_sampled[..., 0].ravel(),  # Flow components
            flow_sampled[..., 1].ravel(),
            np.sqrt(flow_sampled[..., 0]**2 + flow_sampled[..., 1]**2).ravel(),  # Magnitude
            np.arctan2(flow_sampled[..., 1], flow_sampled[..., 0]).ravel()  # Angle
        ])
        
        # Filter out low motion
        magnitudes = features[:, 4]
        significant = magnitudes > self.motion_significance_threshold
        
        if np.sum(significant) < 10:
            return None
        
        return features[significant]
    
    def _cluster_motion(self, features: np.ndarray, h: int, w: int) -> np.ndarray:
        """Cluster motion vectors using hierarchical clustering."""
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import DBSCAN
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features[:, 2:])  # Exclude positions for now
        
        # Add weighted positions
        positions_scaled = features[:, :2] * 2.0  # Weight spatial proximity
        features_combined = np.hstack([positions_scaled, features_scaled])
        
        # Cluster using DBSCAN
        clustering = DBSCAN(eps=self.clustering_threshold, min_samples=5)
        labels = clustering.fit_predict(features_combined)
        
        # Map back to full image
        h_s = h // max(1, min(h, w) // 64)
        w_s = w // max(1, min(h, w) // 64)
        
        full_labels = np.full((h_s, w_s), -1)
        
        # Reconstruct labels
        idx = 0
        for i in range(h_s):
            for j in range(w_s):
                y_norm = i / h_s
                x_norm = j / w_s
                
                # Find if this position was in features
                for k, feat in enumerate(features):
                    if abs(feat[0] - x_norm) < 0.02 and abs(feat[1] - y_norm) < 0.02:
                        full_labels[i, j] = labels[k]
                        break
        
        # Upsample to original size
        from scipy.ndimage import zoom
        scale_h = h / h_s
        scale_w = w / w_s
        full_labels_upsampled = zoom(full_labels, (scale_h, scale_w), order=0)
        
        return full_labels_upsampled
    
    def _compute_region_coherence(self, region_flow: np.ndarray) -> float:
        """Compute coherence within a region."""
        if region_flow.size == 0:
            return 0.0
        
        # Reshape to vectors
        flow_vectors = region_flow.reshape(-1, 2)
        
        # Filter significant motion
        magnitudes = np.linalg.norm(flow_vectors, axis=1)
        significant = magnitudes > self.motion_significance_threshold
        
        if np.sum(significant) < 2:
            return 0.0
        
        significant_flows = flow_vectors[significant]
        
        # Normalize
        norms = np.linalg.norm(significant_flows, axis=1)
        normalized = significant_flows / norms[:, np.newaxis]
        
        # Compute mean direction
        mean_dir = np.mean(normalized, axis=0)
        mean_dir = mean_dir / np.linalg.norm(mean_dir)
        
        # Coherence is mean similarity to mean direction
        similarities = np.dot(normalized, mean_dir)
        coherence = np.mean(similarities)
        
        return float(coherence)
    
    def _analyze_temporal_coherence(self, flow_field: List[np.ndarray]) -> Dict:
        """Analyze temporal coherence of motion patterns."""
        temporal = {
            'frame_to_frame_coherence': [],
            'window_coherence': [],
            'coherence_stability': 0.0,
            'coherence_trend': 'stable'
        }
        
        if len(flow_field) < 2:
            return temporal
        
        # Frame-to-frame coherence
        for i in range(1, len(flow_field)):
            if flow_field[i].size == 0 or flow_field[i-1].size == 0:
                temporal['frame_to_frame_coherence'].append(0.0)
                continue
            
            coherence = self._compute_flow_similarity(flow_field[i-1], flow_field[i])
            temporal['frame_to_frame_coherence'].append(coherence)
        
        # Window-based coherence
        for i in range(0, len(flow_field) - self.temporal_window + 1, self.temporal_window // 2):
            window = flow_field[i:i+self.temporal_window]
            window_coh = self._compute_window_coherence(window)
            temporal['window_coherence'].append({
                'start_frame': i,
                'end_frame': i + self.temporal_window,
                'coherence': window_coh
            })
        
        # Coherence stability
        if temporal['frame_to_frame_coherence']:
            coherence_array = np.array(temporal['frame_to_frame_coherence'])
            temporal['coherence_stability'] = float(
                1.0 - np.std(coherence_array) / (np.mean(coherence_array) + 1e-6)
            )
            
            # Trend detection
            if len(coherence_array) > 10:
                trend = np.polyfit(np.arange(len(coherence_array)), coherence_array, 1)[0]
                if trend > 0.01:
                    temporal['coherence_trend'] = 'increasing'
                elif trend < -0.01:
                    temporal['coherence_trend'] = 'decreasing'
        
        return temporal
    
    def _compute_flow_similarity(self, flow1: np.ndarray, flow2: np.ndarray) -> float:
        """Compute similarity between two flow fields."""
        if flow1.shape != flow2.shape:
            return 0.0
        
        # Compute magnitude fields
        mag1 = np.sqrt(flow1[..., 0]**2 + flow1[..., 1]**2)
        mag2 = np.sqrt(flow2[..., 0]**2 + flow2[..., 1]**2)
        
        # Filter significant motion
        significant = (mag1 > self.motion_significance_threshold) & \
                     (mag2 > self.motion_significance_threshold)
        
        if np.sum(significant) < 10:
            return 0.0
        
        # Compare flow vectors
        flow1_sig = flow1[significant]
        flow2_sig = flow2[significant]
        
        # Normalize
        norm1 = np.linalg.norm(flow1_sig, axis=1)
        norm2 = np.linalg.norm(flow2_sig, axis=1)
        
        flow1_norm = flow1_sig / norm1[:, np.newaxis]
        flow2_norm = flow2_sig / norm2[:, np.newaxis]
        
        # Compute similarity
        similarities = np.sum(flow1_norm * flow2_norm, axis=1)
        
        return float(np.mean(similarities))
    
    def _compute_window_coherence(self, window: List[np.ndarray]) -> float:
        """Compute coherence within a temporal window."""
        if len(window) < 2:
            return 0.0
        
        # Compute pairwise similarities
        similarities = []
        
        for i in range(len(window)):
            for j in range(i+1, len(window)):
                if window[i].size > 0 and window[j].size > 0:
                    sim = self._compute_flow_similarity(window[i], window[j])
                    similarities.append(sim)
        
        if similarities:
            return float(np.mean(similarities))
        
        return 0.0
    
    def _detect_coherence_patterns(self, coherence_analysis: Dict,
                                  motion_regions: List[Dict]) -> List[Dict]:
        """Detect specific coherence patterns."""
        patterns = []
        
        if not coherence_analysis.get('local_coherence'):
            return patterns
        
        # Analyze coherence timeline
        local_coh = np.array(coherence_analysis['local_coherence'])
        global_coh = np.array(coherence_analysis['global_coherence'])
        
        for i in range(len(local_coh)):
            pattern_type = 'none'
            characteristics = {}
            
            # Classify based on coherence values
            if global_coh[i] > 0.8:
                pattern_type = 'uniform_flow'
                characteristics['uniformity'] = global_coh[i]
            elif local_coh[i] > 0.7 and global_coh[i] < 0.5:
                pattern_type = 'locally_coherent'
                characteristics['local_coherence'] = local_coh[i]
                characteristics['global_coherence'] = global_coh[i]
            elif local_coh[i] < 0.3:
                pattern_type = 'turbulent'
                characteristics['turbulence'] = 1.0 - local_coh[i]
            elif len([r for r in motion_regions if r['frame'] == i]) > 0:
                frame_regions = [r for r in motion_regions if r['frame'] == i]
                if frame_regions and frame_regions[0]['num_regions'] > 2:
                    pattern_type = 'multi_region'
                    characteristics['num_regions'] = frame_regions[0]['num_regions']
            else:
                if local_coh[i] > 0.5:
                    pattern_type = 'semi_coherent'
                else:
                    pattern_type = 'scattered'
            
            if pattern_type != 'none':
                patterns.append({
                    'frame': i,
                    'type': pattern_type,
                    'confidence': float(max(local_coh[i], global_coh[i])),
                    'characteristics': characteristics
                })
        
        return patterns
    
    def _analyze_motion_organization(self, flow_field: List[np.ndarray],
                                   motion_regions: List[Dict]) -> Dict:
        """Analyze the organization of motion patterns."""
        organization = {
            'organization_level': 'none',
            'hierarchical_structure': [],
            'dominant_scale': 'none',
            'symmetry': 0.0,
            'regularity': 0.0
        }
        
        if not flow_field or not motion_regions:
            return organization
        
        # Analyze organization level
        avg_num_regions = np.mean([r['num_regions'] for r in motion_regions])
        avg_coherence = np.mean([
            r['regions'][0]['coherence'] 
            for r in motion_regions 
            if r['regions']
        ])
        
        if avg_num_regions < 2 and avg_coherence > 0.8:
            organization['organization_level'] = 'highly_organized'
        elif avg_num_regions < 4 and avg_coherence > 0.6:
            organization['organization_level'] = 'organized'
        elif avg_num_regions > 6 or avg_coherence < 0.4:
            organization['organization_level'] = 'disorganized'
        else:
            organization['organization_level'] = 'semi_organized'
        
        # Analyze hierarchical structure
        if self.use_hierarchical_analysis:
            for flow in flow_field[:10]:  # Sample frames
                hierarchy = self._analyze_motion_hierarchy(flow)
                if hierarchy:
                    organization['hierarchical_structure'].append(hierarchy)
        
        # Determine dominant scale
        if organization['hierarchical_structure']:
            scales = [h['dominant_scale'] for h in organization['hierarchical_structure']]
            from collections import Counter
            scale_counts = Counter(scales)
            organization['dominant_scale'] = scale_counts.most_common(1)[0][0]
        
        # Analyze symmetry
        for flow in flow_field[:10]:
            symmetry = self._compute_motion_symmetry(flow)
            organization['symmetry'] = max(organization['symmetry'], symmetry)
        
        # Analyze regularity
        if motion_regions:
            region_sizes = []
            for r in motion_regions:
                region_sizes.extend([reg['size'] for reg in r['regions']])
            
            if region_sizes:
                cv = np.std(region_sizes) / (np.mean(region_sizes) + 1e-6)
                organization['regularity'] = float(1.0 / (1.0 + cv))
        
        return organization
    
    def _analyze_motion_hierarchy(self, flow: np.ndarray) -> Optional[Dict]:
        """Analyze hierarchical structure of motion."""
        if flow.size == 0:
            return None
        
        hierarchy = {
            'levels': [],
            'dominant_scale': 'none'
        }
        
        # Analyze at multiple scales
        scales = [8, 16, 32, 64]
        coherences = []
        
        for scale in scales:
            if scale > min(flow.shape[:2]) // 2:
                continue
            
            # Downsample flow
            from scipy.ndimage import zoom
            factor = scale / min(flow.shape[:2])
            flow_scaled = zoom(flow, (factor, factor, 1))
            
            # Compute coherence at this scale
            coherence = self._compute_global_coherence(flow_scaled)
            coherences.append(coherence)
            
            hierarchy['levels'].append({
                'scale': scale,
                'coherence': coherence
            })
        
        # Determine dominant scale
        if coherences:
            best_idx = np.argmax(coherences)
            if best_idx == 0:
                hierarchy['dominant_scale'] = 'fine'
            elif best_idx == len(coherences) - 1:
                hierarchy['dominant_scale'] = 'coarse'
            else:
                hierarchy['dominant_scale'] = 'medium'
        
        return hierarchy
    
    def _compute_motion_symmetry(self, flow: np.ndarray) -> float:
        """Compute symmetry in motion field."""
        if flow.size == 0:
            return 0.0
        
        h, w = flow.shape[:2]
        
        # Vertical symmetry
        left = flow[:, :w//2]
        right = np.flip(flow[:, w//2:w//2*2], axis=1)
        
        if left.shape == right.shape:
            # Flip x-component for right side
            right_flipped = right.copy()
            right_flipped[..., 0] = -right_flipped[..., 0]
            
            v_symmetry = 1.0 - np.mean(np.abs(left - right_flipped)) / \
                        (np.mean(np.abs(left)) + np.mean(np.abs(right_flipped)) + 1e-6)
        else:
            v_symmetry = 0.0
        
        # Horizontal symmetry
        top = flow[:h//2, :]
        bottom = np.flip(flow[h//2:h//2*2, :], axis=0)
        
        if top.shape == bottom.shape:
            # Flip y-component for bottom
            bottom_flipped = bottom.copy()
            bottom_flipped[..., 1] = -bottom_flipped[..., 1]
            
            h_symmetry = 1.0 - np.mean(np.abs(top - bottom_flipped)) / \
                        (np.mean(np.abs(top)) + np.mean(np.abs(bottom_flipped)) + 1e-6)
        else:
            h_symmetry = 0.0
        
        return float(max(v_symmetry, h_symmetry))
    
    def _detect_coherence_phenomena(self, flow_field: List[np.ndarray],
                                   coherence_analysis: Dict,
                                   temporal_coherence: Dict) -> Dict:
        """Detect specific coherence phenomena."""
        phenomena = {
            'has_vortices': False,
            'has_divergence_convergence': False,
            'has_shear_layers': False,
            'has_coherent_structures': False,
            'phenomena_instances': []
        }
        
        for i, flow in enumerate(flow_field[:50]):  # Limit processing
            if flow.size == 0:
                continue
            
            # Detect vortices
            vortex = self._detect_vortex(flow)
            if vortex['has_vortex']:
                phenomena['has_vortices'] = True
                phenomena['phenomena_instances'].append({
                    'type': 'vortex',
                    'frame': i,
                    **vortex
                })
            
            # Detect divergence/convergence
            div_conv = self._detect_divergence_convergence(flow)
            if div_conv['has_pattern']:
                phenomena['has_divergence_convergence'] = True
                phenomena['phenomena_instances'].append({
                    'type': div_conv['pattern_type'],
                    'frame': i,
                    **div_conv
                })
            
            # Detect shear layers
            shear = self._detect_shear_layers(flow)
            if shear['has_shear']:
                phenomena['has_shear_layers'] = True
                phenomena['phenomena_instances'].append({
                    'type': 'shear_layer',
                    'frame': i,
                    **shear
                })
        
        # Check for coherent structures
        if coherence_analysis.get('global_coherence'):
            high_coherence_frames = [
                i for i, c in enumerate(coherence_analysis['global_coherence'])
                if c > 0.7
            ]
            
            if len(high_coherence_frames) > len(coherence_analysis['global_coherence']) * 0.3:
                phenomena['has_coherent_structures'] = True
        
        return phenomena
    
    def _detect_vortex(self, flow: np.ndarray) -> Dict:
        """Detect vortex patterns in flow field."""
        h, w = flow.shape[:2]
        
        # Compute curl (vorticity)
        du_dy = np.gradient(flow[..., 0], axis=0)
        dv_dx = np.gradient(flow[..., 1], axis=1)
        curl = dv_dx - du_dy
        
        # Find high vorticity regions
        vorticity_threshold = np.percentile(np.abs(curl), 95)
        
        if vorticity_threshold < 0.5:
            return {'has_vortex': False}
        
        high_vorticity = np.abs(curl) > vorticity_threshold
        
        # Check for circular pattern
        from scipy.ndimage import label
        labeled, num_features = label(high_vorticity)
        
        for i in range(1, num_features + 1):
            mask = labeled == i
            
            if np.sum(mask) > h * w * 0.001:
                # Check if flow forms circular pattern
                coords = np.where(mask)
                center_y = np.mean(coords[0])
                center_x = np.mean(coords[1])
                
                # Sample flow around center
                angles = []
                for y, x in zip(coords[0][::10], coords[1][::10]):
                    dy = y - center_y
                    dx = x - center_x
                    
                    if dx != 0 or dy != 0:
                        radial_angle = np.arctan2(dy, dx)
                        flow_angle = np.arctan2(flow[y, x, 1], flow[y, x, 0])
                        
                        # Check if flow is perpendicular to radial
                        diff = abs(flow_angle - radial_angle - np.pi/2)
                        diff = min(diff, 2*np.pi - diff)
                        angles.append(diff)
                
                if angles and np.mean(angles) < np.pi/4:
                    return {
                        'has_vortex': True,
                        'center': (float(center_x), float(center_y)),
                        'strength': float(np.mean(np.abs(curl[mask])))
                    }
        
        return {'has_vortex': False}
    
    def _detect_divergence_convergence(self, flow: np.ndarray) -> Dict:
        """Detect divergence or convergence patterns."""
        h, w = flow.shape[:2]
        
        # Compute divergence
        du_dx = np.gradient(flow[..., 0], axis=1)
        dv_dy = np.gradient(flow[..., 1], axis=0)
        divergence = du_dx + dv_dy
        
        # Check for significant divergence/convergence
        div_threshold = np.percentile(np.abs(divergence), 95)
        
        if div_threshold < 0.5:
            return {'has_pattern': False}
        
        # Find source/sink regions
        sources = divergence > div_threshold
        sinks = divergence < -div_threshold
        
        if np.sum(sources) > h * w * 0.01:
            return {
                'has_pattern': True,
                'pattern_type': 'divergence',
                'strength': float(np.max(divergence))
            }
        elif np.sum(sinks) > h * w * 0.01:
            return {
                'has_pattern': True,
                'pattern_type': 'convergence',
                'strength': float(abs(np.min(divergence)))
            }
        
        return {'has_pattern': False}
    
    def _detect_shear_layers(self, flow: np.ndarray) -> Dict:
        """Detect shear layers in flow field."""
        h, w = flow.shape[:2]
        
        # Compute shear
        du_dy = np.gradient(flow[..., 0], axis=0)
        dv_dx = np.gradient(flow[..., 1], axis=1)
        shear = du_dy + dv_dx
        
        # Find high shear regions
        shear_threshold = np.percentile(np.abs(shear), 95)
        
        if shear_threshold < 0.5:
            return {'has_shear': False}
        
        high_shear = np.abs(shear) > shear_threshold
        
        # Check for linear structure
        from scipy.ndimage import label
        labeled, num_features = label(high_shear)
        
        for i in range(1, num_features + 1):
            mask = labeled == i
            
            if np.sum(mask) > h * w * 0.005:
                coords = np.where(mask)
                
                # Check aspect ratio
                y_range = np.max(coords[0]) - np.min(coords[0])
                x_range = np.max(coords[1]) - np.min(coords[1])
                
                aspect_ratio = max(y_range, x_range) / (min(y_range, x_range) + 1)
                
                if aspect_ratio > 3:  # Elongated structure
                    return {
                        'has_shear': True,
                        'orientation': 'horizontal' if x_range > y_range else 'vertical',
                        'strength': float(np.mean(np.abs(shear[mask])))
                    }
        
        return {'has_shear': False}
    
    def _compute_coherence_dynamics(self, coherence_analysis: Dict,
                                   temporal_coherence: Dict) -> Dict:
        """Compute dynamics of coherence patterns."""
        dynamics = {
            'coherence_evolution': 'stable',
            'transition_points': [],
            'periodicity': 0.0,
            'trend': 0.0
        }
        
        if not coherence_analysis.get('local_coherence'):
            return dynamics
        
        coherence_timeline = np.array(coherence_analysis['local_coherence'])
        
        # Detect transition points
        if len(coherence_timeline) > 3:
            # Compute gradient
            gradient = np.gradient(coherence_timeline)
            
            # Find significant changes
            threshold = np.std(gradient) * 2
            transitions = np.where(np.abs(gradient) > threshold)[0]
            
            dynamics['transition_points'] = transitions.tolist()
        
        # Detect periodicity
        if len(coherence_timeline) > 10:
            # Autocorrelation
            autocorr = np.correlate(
                coherence_timeline - np.mean(coherence_timeline),
                coherence_timeline - np.mean(coherence_timeline),
                mode='full'
            )
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr / autocorr[0]
            
            # Find peaks
            peaks, _ = signal.find_peaks(autocorr[1:], height=0.3)
            
            if len(peaks) > 0:
                dynamics['periodicity'] = float(peaks[0] + 1)
        
        # Compute trend
        if len(coherence_timeline) > 5:
            trend = np.polyfit(np.arange(len(coherence_timeline)), coherence_timeline, 1)[0]
            dynamics['trend'] = float(trend)
            
            if trend > 0.01:
                dynamics['coherence_evolution'] = 'increasing'
            elif trend < -0.01:
                dynamics['coherence_evolution'] = 'decreasing'
            elif dynamics['periodicity'] > 0:
                dynamics['coherence_evolution'] = 'oscillating'
        
        return dynamics
    
    def _compute_statistics(self, coherence_analysis: Dict,
                          coherence_patterns: List[Dict]) -> Dict:
        """Compute coherence statistics."""
        stats = {
            'mean_coherence': 0.0,
            'dominant_pattern': 'none',
            'num_coherent_regions': 0,
            'coherence_distribution': {}
        }
        
        # Mean coherence
        if coherence_analysis.get('global_coherence'):
            stats['mean_coherence'] = float(np.mean(coherence_analysis['global_coherence']))
        
        # Dominant pattern
        if coherence_patterns:
            pattern_counts = {}
            for p in coherence_patterns:
                pattern_counts[p['type']] = pattern_counts.get(p['type'], 0) + 1
            
            stats['dominant_pattern'] = max(pattern_counts.items(), key=lambda x: x[1])[0]
            stats['coherence_distribution'] = pattern_counts
        
        # Number of coherent regions
        if coherence_analysis.get('spatial_coherence_maps'):
            for coh_map in coherence_analysis['spatial_coherence_maps']:
                if isinstance(coh_map, np.ndarray):
                    coherent_blocks = np.sum(coh_map > self.coherence_threshold)
                    stats['num_coherent_regions'] = max(
                        stats['num_coherent_regions'],
                        int(coherent_blocks)
                    )
        
        return stats