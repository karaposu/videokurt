"""Structural change pattern detection using advanced edge and layout analysis."""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from scipy import signal, stats
from scipy.ndimage import gaussian_filter, label, binary_dilation
from scipy.spatial.distance import cosine, euclidean
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.decomposition import PCA
import cv2
import warnings

from ..base import BaseFeature


class StructuralChangePatterns(BaseFeature):
    """Detect and analyze patterns in structural changes using comprehensive edge analysis."""
    
    FEATURE_NAME = 'structural_change_patterns'
    REQUIRED_ANALYSES = ['edge_canny', 'frame_diff', 'contour_detection', 'color_histogram']
    
    def __init__(self, 
                 change_threshold: float = 0.2,
                 temporal_window: int = 15,
                 spatial_grid_size: int = 8,
                 structure_similarity_threshold: float = 0.7,
                 min_edge_density: float = 0.05,
                 use_multi_scale: bool = True,
                 persistence_threshold: int = 3):
        """
        Args:
            change_threshold: Threshold for significant structural change
            temporal_window: Window for temporal analysis
            spatial_grid_size: Grid size for spatial structure analysis
            structure_similarity_threshold: Threshold for structure similarity
            min_edge_density: Minimum edge density to consider structure
            use_multi_scale: Use multi-scale structural analysis
            persistence_threshold: Min frames for persistent change
        """
        super().__init__()
        self.change_threshold = change_threshold
        self.temporal_window = temporal_window
        self.spatial_grid_size = spatial_grid_size
        self.structure_similarity_threshold = structure_similarity_threshold
        self.min_edge_density = min_edge_density
        self.use_multi_scale = use_multi_scale
        self.persistence_threshold = persistence_threshold
    
    def compute(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect structural change patterns using comprehensive analysis.
        
        Returns:
            Dict with structural changes, patterns, and evolution
        """
        edge_maps = analysis_data['edge_canny'].data['edge_map']
        frame_diffs = analysis_data['frame_diff'].data['pixel_diff']
        contours = analysis_data['contour_detection'].data.get('contours', [])
        color_hists = analysis_data['color_histogram'].data['histograms']
        
        if len(edge_maps) == 0:
            return self._empty_result()
        
        # Extract structural signatures
        structural_signatures = self._extract_structural_signatures(
            edge_maps, contours
        )
        
        # Detect structural changes
        structural_changes = self._detect_structural_changes(
            structural_signatures, frame_diffs
        )
        
        # Analyze change patterns
        change_patterns = self._analyze_change_patterns(
            structural_changes, structural_signatures
        )
        
        # Detect structural evolution
        evolution = self._detect_structural_evolution(
            structural_signatures, structural_changes
        )
        
        # Identify structural motifs
        motifs = self._identify_structural_motifs(
            structural_signatures, edge_maps
        )
        
        # Analyze layout changes
        layout_changes = self._analyze_layout_changes(
            edge_maps, structural_signatures
        )
        
        # Detect specific phenomena
        phenomena = self._detect_structural_phenomena(
            structural_changes, evolution
        )
        
        return {
            'structural_changes': structural_changes,
            'change_patterns': change_patterns,
            'evolution': evolution,
            'motifs': motifs,
            'layout_changes': layout_changes,
            'phenomena': phenomena,
            'statistics': self._compute_statistics(
                structural_changes, change_patterns
            )
        }
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result structure."""
        return {
            'structural_changes': [],
            'change_patterns': {},
            'evolution': {},
            'motifs': [],
            'layout_changes': [],
            'phenomena': {},
            'statistics': {
                'num_changes': 0,
                'dominant_change_type': 'none',
                'structural_stability': 1.0
            }
        }
    
    def _extract_structural_signatures(self, edge_maps: List[np.ndarray],
                                      contours: List) -> List[Dict]:
        """Extract comprehensive structural signatures from edge maps."""
        signatures = []
        
        for i, edges in enumerate(edge_maps):
            if edges.size == 0:
                signatures.append({})
                continue
            
            signature = {
                'spatial_distribution': self._compute_spatial_distribution(edges),
                'edge_statistics': self._compute_edge_statistics(edges),
                'structural_elements': self._extract_structural_elements(edges),
                'geometric_features': self._compute_geometric_features(edges)
            }
            
            # Add contour-based features if available
            if i < len(contours) and contours[i]:
                signature['contour_features'] = self._extract_contour_features(contours[i])
            
            # Multi-scale analysis if enabled
            if self.use_multi_scale:
                signature['multi_scale'] = self._multi_scale_analysis(edges)
            
            signatures.append(signature)
        
        return signatures
    
    def _compute_spatial_distribution(self, edges: np.ndarray) -> Dict:
        """Compute spatial distribution of edges."""
        h, w = edges.shape
        
        # Grid-based distribution
        grid_h = min(self.spatial_grid_size, h // 4)
        grid_w = min(self.spatial_grid_size, w // 4)
        
        distribution = np.zeros((grid_h, grid_w))
        cell_h = h // grid_h
        cell_w = w // grid_w
        
        for i in range(grid_h):
            for j in range(grid_w):
                cell = edges[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                distribution[i, j] = np.mean(cell > 0)
        
        # Compute distribution metrics
        return {
            'grid': distribution.tolist(),
            'mean_density': float(np.mean(distribution)),
            'std_density': float(np.std(distribution)),
            'entropy': float(stats.entropy(distribution.flatten() + 1e-10)),
            'center_weight': float(distribution[grid_h//2, grid_w//2]),
            'edge_weight': float(np.mean([
                distribution[0, :].mean(),
                distribution[-1, :].mean(),
                distribution[:, 0].mean(),
                distribution[:, -1].mean()
            ]))
        }
    
    def _compute_edge_statistics(self, edges: np.ndarray) -> Dict:
        """Compute statistical properties of edges."""
        # Edge density
        edge_pixels = edges > 0
        density = np.mean(edge_pixels)
        
        # Connected components
        labeled, num_components = label(edge_pixels)
        
        # Component sizes
        component_sizes = []
        for i in range(1, num_components + 1):
            size = np.sum(labeled == i)
            component_sizes.append(size)
        
        # Edge orientations
        orientations = self._compute_edge_orientations(edges)
        
        return {
            'density': float(density),
            'num_components': int(num_components),
            'largest_component': int(max(component_sizes)) if component_sizes else 0,
            'mean_component_size': float(np.mean(component_sizes)) if component_sizes else 0,
            'orientation_histogram': orientations['histogram'],
            'dominant_orientation': orientations['dominant'],
            'orientation_entropy': orientations['entropy']
        }
    
    def _compute_edge_orientations(self, edges: np.ndarray) -> Dict:
        """Compute edge orientation distribution."""
        # Compute gradients
        gx = cv2.Sobel(edges.astype(float), cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(edges.astype(float), cv2.CV_64F, 0, 1, ksize=3)
        
        # Compute angles where edges exist
        edge_mask = edges > 0
        angles = np.arctan2(gy[edge_mask], gx[edge_mask])
        
        if len(angles) == 0:
            return {
                'histogram': [0] * 8,
                'dominant': 0,
                'entropy': 0
            }
        
        # Create orientation histogram (8 bins)
        hist, _ = np.histogram(angles, bins=8, range=(-np.pi, np.pi))
        hist = hist / (np.sum(hist) + 1e-10)
        
        # Find dominant orientation
        dominant_bin = np.argmax(hist)
        dominant_angle = -np.pi + (dominant_bin + 0.5) * 2 * np.pi / 8
        
        # Compute entropy
        entropy = stats.entropy(hist + 1e-10)
        
        return {
            'histogram': hist.tolist(),
            'dominant': float(dominant_angle),
            'entropy': float(entropy)
        }
    
    def _extract_structural_elements(self, edges: np.ndarray) -> Dict:
        """Extract structural elements like lines, corners, junctions."""
        h, w = edges.shape
        
        elements = {
            'lines': self._detect_lines(edges),
            'corners': self._detect_corners(edges),
            'junctions': self._detect_junctions(edges),
            'rectangles': self._detect_rectangles(edges)
        }
        
        return elements
    
    def _detect_lines(self, edges: np.ndarray) -> Dict:
        """Detect line structures."""
        # Use Hough transform
        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180, threshold=50,
            minLineLength=30, maxLineGap=10
        )
        
        if lines is None:
            return {'count': 0, 'total_length': 0, 'orientations': []}
        
        line_info = {
            'count': len(lines),
            'total_length': 0,
            'orientations': []
        }
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            angle = np.arctan2(y2-y1, x2-x1)
            
            line_info['total_length'] += length
            line_info['orientations'].append(angle)
        
        line_info['total_length'] = float(line_info['total_length'])
        
        return line_info
    
    def _detect_corners(self, edges: np.ndarray) -> Dict:
        """Detect corner structures."""
        # Harris corner detection
        corners = cv2.cornerHarris(edges.astype(np.float32), 2, 3, 0.04)
        
        # Threshold
        corner_threshold = 0.01 * corners.max()
        corner_points = corners > corner_threshold
        
        # Find corner locations
        corner_coords = np.where(corner_points)
        
        return {
            'count': len(corner_coords[0]),
            'density': float(np.sum(corner_points) / edges.size),
            'strength': float(np.mean(corners[corner_points])) if np.any(corner_points) else 0
        }
    
    def _detect_junctions(self, edges: np.ndarray) -> Dict:
        """Detect junction points (T-junctions, X-junctions)."""
        # Simplified junction detection using morphology
        kernel = np.ones((3, 3), np.uint8)
        
        # Count neighbors for each edge pixel
        edge_binary = (edges > 0).astype(np.uint8)
        neighbor_count = cv2.filter2D(edge_binary, -1, kernel) - edge_binary
        
        # Junctions have 3+ neighbors
        t_junctions = (neighbor_count == 3) & edge_binary
        x_junctions = (neighbor_count >= 4) & edge_binary
        
        return {
            't_count': int(np.sum(t_junctions)),
            'x_count': int(np.sum(x_junctions)),
            'total': int(np.sum(t_junctions) + np.sum(x_junctions))
        }
    
    def _detect_rectangles(self, edges: np.ndarray) -> Dict:
        """Detect rectangular structures."""
        # Find contours
        contours, _ = cv2.findContours(
            edges.astype(np.uint8), 
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        rectangles = {
            'count': 0,
            'total_area': 0,
            'aspect_ratios': []
        }
        
        for contour in contours:
            # Approximate contour
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Check if rectangular (4 vertices)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                
                rectangles['count'] += 1
                rectangles['total_area'] += w * h
                rectangles['aspect_ratios'].append(w / h if h > 0 else 0)
        
        rectangles['total_area'] = float(rectangles['total_area'])
        
        return rectangles
    
    def _compute_geometric_features(self, edges: np.ndarray) -> Dict:
        """Compute geometric features of the edge structure."""
        h, w = edges.shape
        
        # Compute moments
        moments = cv2.moments(edges)
        
        features = {}
        
        if moments['m00'] > 0:
            # Centroid
            cx = moments['m10'] / moments['m00']
            cy = moments['m01'] / moments['m00']
            features['centroid'] = (float(cx / w), float(cy / h))  # Normalized
            
            # Central moments
            features['central_moments'] = {
                'mu20': float(moments['mu20']),
                'mu11': float(moments['mu11']),
                'mu02': float(moments['mu02'])
            }
            
            # Hu moments (invariant features)
            hu_moments = cv2.HuMoments(moments).flatten()
            features['hu_moments'] = hu_moments[:4].tolist()  # First 4 moments
        else:
            features['centroid'] = (0.5, 0.5)
            features['central_moments'] = {'mu20': 0, 'mu11': 0, 'mu02': 0}
            features['hu_moments'] = [0] * 4
        
        return features
    
    def _extract_contour_features(self, contour_data: Dict) -> Dict:
        """Extract features from contour data."""
        if not contour_data or 'contours' not in contour_data:
            return {}
        
        contours = contour_data['contours']
        
        features = {
            'num_contours': len(contours),
            'total_perimeter': 0,
            'mean_area': 0,
            'complexity': []
        }
        
        areas = []
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            area = cv2.contourArea(contour)
            
            features['total_perimeter'] += perimeter
            areas.append(area)
            
            # Complexity as perimeter^2 / area
            if area > 0:
                complexity = perimeter * perimeter / area
                features['complexity'].append(complexity)
        
        if areas:
            features['mean_area'] = float(np.mean(areas))
            features['total_perimeter'] = float(features['total_perimeter'])
            features['mean_complexity'] = float(np.mean(features['complexity']))
        
        return features
    
    def _multi_scale_analysis(self, edges: np.ndarray) -> Dict:
        """Perform multi-scale structural analysis."""
        scales = [1, 2, 4, 8]
        multi_scale = {
            'scale_densities': [],
            'scale_components': [],
            'dominant_scale': 1
        }
        
        max_density = 0
        dominant_scale = 1
        
        for scale in scales:
            if scale > 1:
                # Downsample
                scaled = cv2.resize(
                    edges, 
                    (edges.shape[1]//scale, edges.shape[0]//scale),
                    interpolation=cv2.INTER_AREA
                )
            else:
                scaled = edges
            
            # Compute density at this scale
            density = np.mean(scaled > 0)
            multi_scale['scale_densities'].append(float(density))
            
            # Count components
            _, num_components = label(scaled > 0)
            multi_scale['scale_components'].append(int(num_components))
            
            # Track dominant scale
            if density > max_density:
                max_density = density
                dominant_scale = scale
        
        multi_scale['dominant_scale'] = dominant_scale
        
        return multi_scale
    
    def _detect_structural_changes(self, signatures: List[Dict],
                                  frame_diffs: List[np.ndarray]) -> List[Dict]:
        """Detect structural changes between frames."""
        changes = []
        
        for i in range(1, len(signatures)):
            if not signatures[i] or not signatures[i-1]:
                continue
            
            # Compute change metrics
            change_metrics = self._compute_change_metrics(
                signatures[i-1], signatures[i]
            )
            
            # Check if significant change
            if change_metrics['total_change'] > self.change_threshold:
                # Classify change type
                change_type = self._classify_change_type(
                    change_metrics, 
                    frame_diffs[i-1] if i-1 < len(frame_diffs) else None
                )
                
                changes.append({
                    'frame': i,
                    'type': change_type,
                    'magnitude': change_metrics['total_change'],
                    'metrics': change_metrics,
                    'affected_regions': self._identify_affected_regions(
                        signatures[i-1], signatures[i]
                    )
                })
        
        # Filter for persistent changes
        changes = self._filter_persistent_changes(changes)
        
        return changes
    
    def _compute_change_metrics(self, sig1: Dict, sig2: Dict) -> Dict:
        """Compute detailed change metrics between signatures."""
        metrics = {
            'spatial_change': 0,
            'density_change': 0,
            'orientation_change': 0,
            'component_change': 0,
            'geometric_change': 0,
            'total_change': 0
        }
        
        # Spatial distribution change
        if 'spatial_distribution' in sig1 and 'spatial_distribution' in sig2:
            dist1 = np.array(sig1['spatial_distribution']['grid']).flatten()
            dist2 = np.array(sig2['spatial_distribution']['grid']).flatten()
            
            if dist1.shape == dist2.shape:
                metrics['spatial_change'] = float(np.mean(np.abs(dist1 - dist2)))
        
        # Density change
        if 'edge_statistics' in sig1 and 'edge_statistics' in sig2:
            metrics['density_change'] = abs(
                sig2['edge_statistics']['density'] - 
                sig1['edge_statistics']['density']
            )
            
            # Component change
            metrics['component_change'] = abs(
                sig2['edge_statistics']['num_components'] - 
                sig1['edge_statistics']['num_components']
            ) / 100  # Normalize
        
        # Orientation change
        if 'edge_statistics' in sig1 and 'edge_statistics' in sig2:
            hist1 = np.array(sig1['edge_statistics']['orientation_histogram'])
            hist2 = np.array(sig2['edge_statistics']['orientation_histogram'])
            
            if hist1.shape == hist2.shape:
                metrics['orientation_change'] = float(
                    np.sum(np.abs(hist1 - hist2)) / 2
                )
        
        # Geometric change (using Hu moments)
        if 'geometric_features' in sig1 and 'geometric_features' in sig2:
            hu1 = np.array(sig1['geometric_features']['hu_moments'])
            hu2 = np.array(sig2['geometric_features']['hu_moments'])
            
            if hu1.shape == hu2.shape:
                # Use log scale for Hu moments
                with np.errstate(divide='ignore', invalid='ignore'):
                    log_hu1 = -np.sign(hu1) * np.log10(np.abs(hu1) + 1e-10)
                    log_hu2 = -np.sign(hu2) * np.log10(np.abs(hu2) + 1e-10)
                
                metrics['geometric_change'] = float(
                    np.mean(np.abs(log_hu1 - log_hu2))
                ) / 10  # Normalize
        
        # Total change (weighted sum)
        weights = {
            'spatial_change': 0.3,
            'density_change': 0.2,
            'orientation_change': 0.2,
            'component_change': 0.15,
            'geometric_change': 0.15
        }
        
        metrics['total_change'] = sum(
            metrics[key] * weights.get(key, 0) 
            for key in metrics if key != 'total_change'
        )
        
        return metrics
    
    def _classify_change_type(self, metrics: Dict, 
                             frame_diff: Optional[np.ndarray]) -> str:
        """Classify the type of structural change."""
        # Major structural change
        if metrics['total_change'] > 0.5:
            if metrics['component_change'] > 0.3:
                return 'structural_reorganization'
            else:
                return 'major_structural_change'
        
        # Layout change
        if metrics['spatial_change'] > metrics['density_change'] * 2:
            return 'layout_shift'
        
        # Orientation change
        if metrics['orientation_change'] > 0.3:
            return 'orientation_change'
        
        # Density change
        if metrics['density_change'] > 0.2:
            if metrics['density_change'] > 0:
                return 'structure_addition'
            else:
                return 'structure_removal'
        
        # Geometric transformation
        if metrics['geometric_change'] > 0.2:
            return 'geometric_transformation'
        
        # Component change
        if metrics['component_change'] > 0.1:
            return 'component_change'
        
        return 'minor_structural_change'
    
    def _identify_affected_regions(self, sig1: Dict, sig2: Dict) -> List[Dict]:
        """Identify regions most affected by structural change."""
        affected = []
        
        if 'spatial_distribution' not in sig1 or 'spatial_distribution' not in sig2:
            return affected
        
        grid1 = np.array(sig1['spatial_distribution']['grid'])
        grid2 = np.array(sig2['spatial_distribution']['grid'])
        
        if grid1.shape != grid2.shape:
            return affected
        
        # Compute regional changes
        changes = np.abs(grid2 - grid1)
        h, w = changes.shape
        
        # Find significant changes
        threshold = np.mean(changes) + np.std(changes)
        
        for i in range(h):
            for j in range(w):
                if changes[i, j] > threshold:
                    affected.append({
                        'grid_position': (i, j),
                        'change_magnitude': float(changes[i, j]),
                        'region_name': self._get_region_name(i, j, h, w)
                    })
        
        return affected
    
    def _get_region_name(self, i: int, j: int, h: int, w: int) -> str:
        """Get descriptive name for grid region."""
        v_pos = 'top' if i < h//3 else 'bottom' if i > 2*h//3 else 'middle'
        h_pos = 'left' if j < w//3 else 'right' if j > 2*w//3 else 'center'
        
        if v_pos == 'middle' and h_pos == 'center':
            return 'center'
        else:
            return f'{v_pos}_{h_pos}'
    
    def _filter_persistent_changes(self, changes: List[Dict]) -> List[Dict]:
        """Filter for persistent structural changes."""
        if len(changes) < 2:
            return changes
        
        filtered = []
        
        i = 0
        while i < len(changes):
            # Check if change persists
            persistence = 1
            j = i + 1
            
            while j < len(changes) and changes[j]['frame'] - changes[i]['frame'] <= self.persistence_threshold:
                if changes[j]['type'] == changes[i]['type']:
                    persistence += 1
                j += 1
            
            if persistence >= self.persistence_threshold or changes[i]['magnitude'] > 0.5:
                filtered.append(changes[i])
            
            i += 1
        
        return filtered
    
    def _analyze_change_patterns(self, changes: List[Dict],
                                signatures: List[Dict]) -> Dict:
        """Analyze patterns in structural changes."""
        patterns = {
            'temporal_pattern': 'none',
            'spatial_pattern': 'none',
            'change_frequency': 0,
            'periodicity': 0,
            'clustering': []
        }
        
        if not changes:
            return patterns
        
        # Temporal pattern analysis
        if len(changes) > 2:
            intervals = np.diff([c['frame'] for c in changes])
            
            if len(intervals) > 0:
                patterns['change_frequency'] = 1.0 / np.mean(intervals)
                
                # Check for periodicity
                if len(intervals) > 3:
                    autocorr = np.correlate(intervals, intervals, mode='full')
                    autocorr = autocorr[len(autocorr)//2:]
                    
                    peaks, _ = signal.find_peaks(autocorr[1:], height=0.5*autocorr[0])
                    if len(peaks) > 0:
                        patterns['periodicity'] = float(peaks[0] + 1)
                        patterns['temporal_pattern'] = 'periodic'
                    elif np.std(intervals) < np.mean(intervals) * 0.3:
                        patterns['temporal_pattern'] = 'regular'
                    else:
                        patterns['temporal_pattern'] = 'irregular'
        
        # Spatial pattern analysis
        affected_regions = []
        for change in changes:
            affected_regions.extend(change.get('affected_regions', []))
        
        if affected_regions:
            region_counts = {}
            for region in affected_regions:
                name = region['region_name']
                region_counts[name] = region_counts.get(name, 0) + 1
            
            # Determine spatial pattern
            if 'center' in region_counts and region_counts['center'] > len(affected_regions) * 0.5:
                patterns['spatial_pattern'] = 'central_focus'
            elif any('top' in r for r in region_counts) and any('bottom' in r for r in region_counts):
                patterns['spatial_pattern'] = 'vertical_distribution'
            elif any('left' in r for r in region_counts) and any('right' in r for r in region_counts):
                patterns['spatial_pattern'] = 'horizontal_distribution'
            else:
                patterns['spatial_pattern'] = 'localized'
        
        # Clustering analysis
        if len(changes) > 5:
            frames = [c['frame'] for c in changes]
            clusters = self._detect_change_clusters(frames)
            patterns['clustering'] = clusters
        
        return patterns
    
    def _detect_change_clusters(self, frames: List[int]) -> List[Dict]:
        """Detect clusters of structural changes."""
        clusters = []
        
        if len(frames) < 2:
            return clusters
        
        # Simple clustering based on frame proximity
        cluster_threshold = self.temporal_window
        
        current_cluster = [frames[0]]
        
        for frame in frames[1:]:
            if frame - current_cluster[-1] <= cluster_threshold:
                current_cluster.append(frame)
            else:
                if len(current_cluster) >= 3:
                    clusters.append({
                        'start': current_cluster[0],
                        'end': current_cluster[-1],
                        'size': len(current_cluster),
                        'density': len(current_cluster) / (current_cluster[-1] - current_cluster[0] + 1)
                    })
                current_cluster = [frame]
        
        # Add final cluster
        if len(current_cluster) >= 3:
            clusters.append({
                'start': current_cluster[0],
                'end': current_cluster[-1],
                'size': len(current_cluster),
                'density': len(current_cluster) / (current_cluster[-1] - current_cluster[0] + 1)
            })
        
        return clusters
    
    def _detect_structural_evolution(self, signatures: List[Dict],
                                    changes: List[Dict]) -> Dict:
        """Detect evolution of structural patterns over time."""
        evolution = {
            'trend': 'stable',
            'complexity_timeline': [],
            'stability_periods': [],
            'transformation_sequences': []
        }
        
        if not signatures:
            return evolution
        
        # Compute complexity timeline
        for sig in signatures:
            if sig and 'edge_statistics' in sig:
                complexity = (
                    sig['edge_statistics']['density'] * 
                    np.log(sig['edge_statistics']['num_components'] + 1)
                )
                evolution['complexity_timeline'].append(complexity)
            else:
                evolution['complexity_timeline'].append(0)
        
        # Detect trend
        if len(evolution['complexity_timeline']) > 10:
            x = np.arange(len(evolution['complexity_timeline']))
            y = np.array(evolution['complexity_timeline'])
            
            if np.any(y > 0):
                trend = np.polyfit(x[y > 0], y[y > 0], 1)[0]
                
                if trend > 0.01:
                    evolution['trend'] = 'increasing_complexity'
                elif trend < -0.01:
                    evolution['trend'] = 'decreasing_complexity'
        
        # Identify stability periods
        stability_threshold = self.change_threshold * 0.5
        stable_start = 0
        
        for i in range(1, len(signatures)):
            if i < len(changes) and any(c['frame'] == i for c in changes):
                # End of stability period
                if i - stable_start >= self.temporal_window:
                    evolution['stability_periods'].append({
                        'start': stable_start,
                        'end': i - 1,
                        'duration': i - stable_start
                    })
                stable_start = i + 1
        
        # Add final stability period
        if len(signatures) - stable_start >= self.temporal_window:
            evolution['stability_periods'].append({
                'start': stable_start,
                'end': len(signatures) - 1,
                'duration': len(signatures) - stable_start
            })
        
        # Detect transformation sequences
        if len(changes) > 2:
            sequences = self._detect_transformation_sequences(changes)
            evolution['transformation_sequences'] = sequences
        
        return evolution
    
    def _detect_transformation_sequences(self, changes: List[Dict]) -> List[Dict]:
        """Detect sequences of related structural transformations."""
        sequences = []
        
        if len(changes) < 2:
            return sequences
        
        current_sequence = [changes[0]]
        
        for change in changes[1:]:
            # Check if part of sequence
            if (change['frame'] - current_sequence[-1]['frame'] <= self.temporal_window and
                self._are_related_changes(current_sequence[-1], change)):
                current_sequence.append(change)
            else:
                if len(current_sequence) >= 2:
                    sequences.append({
                        'start_frame': current_sequence[0]['frame'],
                        'end_frame': current_sequence[-1]['frame'],
                        'num_changes': len(current_sequence),
                        'transformation_type': self._classify_transformation(current_sequence)
                    })
                current_sequence = [change]
        
        # Add final sequence
        if len(current_sequence) >= 2:
            sequences.append({
                'start_frame': current_sequence[0]['frame'],
                'end_frame': current_sequence[-1]['frame'],
                'num_changes': len(current_sequence),
                'transformation_type': self._classify_transformation(current_sequence)
            })
        
        return sequences
    
    def _are_related_changes(self, change1: Dict, change2: Dict) -> bool:
        """Check if two changes are related."""
        # Same type
        if change1['type'] == change2['type']:
            return True
        
        # Similar affected regions
        regions1 = set(r['region_name'] for r in change1.get('affected_regions', []))
        regions2 = set(r['region_name'] for r in change2.get('affected_regions', []))
        
        if regions1 and regions2:
            overlap = len(regions1.intersection(regions2)) / len(regions1.union(regions2))
            return overlap > 0.5
        
        return False
    
    def _classify_transformation(self, sequence: List[Dict]) -> str:
        """Classify a transformation sequence."""
        types = [c['type'] for c in sequence]
        type_counts = {}
        for t in types:
            type_counts[t] = type_counts.get(t, 0) + 1
        
        dominant_type = max(type_counts.items(), key=lambda x: x[1])[0]
        
        # Check for progressive transformation
        magnitudes = [c['magnitude'] for c in sequence]
        if len(magnitudes) > 2:
            if all(magnitudes[i] <= magnitudes[i+1] for i in range(len(magnitudes)-1)):
                return f'progressive_{dominant_type}'
            elif all(magnitudes[i] >= magnitudes[i+1] for i in range(len(magnitudes)-1)):
                return f'regressive_{dominant_type}'
        
        return dominant_type
    
    def _identify_structural_motifs(self, signatures: List[Dict],
                                   edge_maps: List[np.ndarray]) -> List[Dict]:
        """Identify recurring structural motifs."""
        motifs = []
        
        if len(signatures) < 10:
            return motifs
        
        # Extract feature vectors for clustering
        feature_vectors = []
        valid_indices = []
        
        for i, sig in enumerate(signatures):
            if sig and 'geometric_features' in sig:
                features = []
                
                # Add various features
                if 'edge_statistics' in sig:
                    features.extend([
                        sig['edge_statistics']['density'],
                        sig['edge_statistics']['num_components'] / 100
                    ])
                
                if 'geometric_features' in sig:
                    features.extend(sig['geometric_features']['hu_moments'][:2])
                
                if features:
                    feature_vectors.append(features)
                    valid_indices.append(i)
        
        if len(feature_vectors) < 5:
            return motifs
        
        # Cluster to find motifs
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import DBSCAN
        
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(feature_vectors)
        
        clustering = DBSCAN(eps=0.5, min_samples=3)
        labels = clustering.fit_predict(features_scaled)
        
        # Extract motifs
        unique_labels = set(labels)
        for label in unique_labels:
            if label < 0:  # Skip noise
                continue
            
            indices = [valid_indices[i] for i, l in enumerate(labels) if l == label]
            
            if len(indices) >= 3:
                motifs.append({
                    'motif_id': int(label),
                    'occurrences': indices,
                    'frequency': len(indices),
                    'characteristics': self._characterize_motif(
                        [signatures[i] for i in indices]
                    )
                })
        
        return motifs
    
    def _characterize_motif(self, motif_signatures: List[Dict]) -> Dict:
        """Characterize a structural motif."""
        characteristics = {
            'mean_density': 0,
            'mean_components': 0,
            'dominant_orientation': 0
        }
        
        densities = []
        components = []
        orientations = []
        
        for sig in motif_signatures:
            if 'edge_statistics' in sig:
                densities.append(sig['edge_statistics']['density'])
                components.append(sig['edge_statistics']['num_components'])
                orientations.append(sig['edge_statistics']['dominant_orientation'])
        
        if densities:
            characteristics['mean_density'] = float(np.mean(densities))
            characteristics['mean_components'] = float(np.mean(components))
            characteristics['dominant_orientation'] = float(
                np.arctan2(np.mean(np.sin(orientations)), 
                          np.mean(np.cos(orientations)))
            )
        
        return characteristics
    
    def _analyze_layout_changes(self, edge_maps: List[np.ndarray],
                               signatures: List[Dict]) -> List[Dict]:
        """Analyze changes in layout structure."""
        layout_changes = []
        
        for i in range(1, min(len(edge_maps), len(signatures))):
            if not signatures[i] or not signatures[i-1]:
                continue
            
            # Check for layout change indicators
            layout_change = self._detect_layout_change(
                signatures[i-1], signatures[i], 
                edge_maps[i-1], edge_maps[i]
            )
            
            if layout_change['has_change']:
                layout_change['frame'] = i
                layout_changes.append(layout_change)
        
        return layout_changes
    
    def _detect_layout_change(self, sig1: Dict, sig2: Dict,
                             edges1: np.ndarray, edges2: np.ndarray) -> Dict:
        """Detect layout change between frames."""
        layout_change = {
            'has_change': False,
            'type': 'none',
            'metrics': {}
        }
        
        # Check structural element changes
        if 'structural_elements' in sig1 and 'structural_elements' in sig2:
            elem1 = sig1['structural_elements']
            elem2 = sig2['structural_elements']
            
            # Significant line change
            if 'lines' in elem1 and 'lines' in elem2:
                line_change = abs(elem2['lines']['count'] - elem1['lines']['count'])
                if line_change > 5:
                    layout_change['has_change'] = True
                    layout_change['type'] = 'line_structure_change'
                    layout_change['metrics']['line_change'] = line_change
            
            # Rectangle change (UI elements)
            if 'rectangles' in elem1 and 'rectangles' in elem2:
                rect_change = abs(elem2['rectangles']['count'] - elem1['rectangles']['count'])
                if rect_change > 2:
                    layout_change['has_change'] = True
                    layout_change['type'] = 'ui_element_change'
                    layout_change['metrics']['rectangle_change'] = rect_change
        
        # Check spatial redistribution
        if 'spatial_distribution' in sig1 and 'spatial_distribution' in sig2:
            center_change = abs(
                sig2['spatial_distribution']['center_weight'] - 
                sig1['spatial_distribution']['center_weight']
            )
            edge_change = abs(
                sig2['spatial_distribution']['edge_weight'] - 
                sig1['spatial_distribution']['edge_weight']
            )
            
            if center_change > 0.3 or edge_change > 0.3:
                layout_change['has_change'] = True
                layout_change['type'] = 'spatial_redistribution'
                layout_change['metrics']['center_change'] = center_change
                layout_change['metrics']['edge_change'] = edge_change
        
        return layout_change
    
    def _detect_structural_phenomena(self, changes: List[Dict],
                                    evolution: Dict) -> Dict:
        """Detect specific structural phenomena."""
        phenomena = {
            'has_gradual_buildup': False,
            'has_sudden_restructuring': False,
            'has_oscillating_structure': False,
            'has_structural_decay': False,
            'phenomena_instances': []
        }
        
        if not changes:
            return phenomena
        
        # Gradual buildup
        if evolution['trend'] == 'increasing_complexity':
            phenomena['has_gradual_buildup'] = True
            phenomena['phenomena_instances'].append({
                'type': 'gradual_buildup',
                'confidence': 0.8
            })
        
        # Sudden restructuring
        for change in changes:
            if change['magnitude'] > 0.6:
                phenomena['has_sudden_restructuring'] = True
                phenomena['phenomena_instances'].append({
                    'type': 'sudden_restructuring',
                    'frame': change['frame'],
                    'magnitude': change['magnitude']
                })
        
        # Oscillating structure
        if len(evolution['complexity_timeline']) > 10:
            timeline = np.array(evolution['complexity_timeline'])
            
            # Check for oscillation using FFT
            if np.any(timeline > 0):
                fft = np.fft.fft(timeline - np.mean(timeline))
                freqs = np.fft.fftfreq(len(timeline))
                
                # Find dominant frequency
                power = np.abs(fft[1:len(fft)//2])
                if len(power) > 0 and np.max(power) > np.mean(power) * 3:
                    phenomena['has_oscillating_structure'] = True
                    dominant_freq_idx = np.argmax(power)
                    phenomena['phenomena_instances'].append({
                        'type': 'oscillating_structure',
                        'period': 1.0 / freqs[dominant_freq_idx + 1] if freqs[dominant_freq_idx + 1] != 0 else 0
                    })
        
        # Structural decay
        if evolution['trend'] == 'decreasing_complexity':
            phenomena['has_structural_decay'] = True
            phenomena['phenomena_instances'].append({
                'type': 'structural_decay',
                'confidence': 0.8
            })
        
        return phenomena
    
    def _compute_statistics(self, changes: List[Dict],
                           patterns: Dict) -> Dict:
        """Compute structural change statistics."""
        stats = {
            'num_changes': len(changes),
            'dominant_change_type': 'none',
            'structural_stability': 1.0,
            'change_distribution': {}
        }
        
        if changes:
            # Change type distribution
            type_counts = {}
            for change in changes:
                change_type = change['type']
                type_counts[change_type] = type_counts.get(change_type, 0) + 1
            
            stats['dominant_change_type'] = max(type_counts.items(), 
                                               key=lambda x: x[1])[0]
            stats['change_distribution'] = type_counts
            
            # Structural stability (inverse of change frequency)
            if patterns.get('change_frequency', 0) > 0:
                stats['structural_stability'] = float(
                    1.0 / (1.0 + patterns['change_frequency'])
                )
        
        return stats