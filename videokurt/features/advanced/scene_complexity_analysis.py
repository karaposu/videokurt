"""
Scene Complexity Analysis

Analyzes the visual complexity of video scenes using multiple metrics including:
- Spatial complexity (edges, textures, regions)
- Color complexity (palette diversity, gradients)
- Information theory metrics (entropy, compression ratio)
- Compositional complexity (layout, balance, visual weight)
- Temporal complexity (motion patterns, changes over time)

This feature is crucial for understanding content density, visual richness,
and cognitive load of different video segments.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import cv2
from scipy import stats, signal, ndimage
from scipy.spatial import distance
from scipy.fftpack import dct2, idct2
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


@dataclass
class SpatialComplexity:
    """Spatial complexity metrics"""
    edge_density: float = 0.0
    edge_distribution_entropy: float = 0.0
    texture_richness: float = 0.0
    texture_variety: float = 0.0
    region_count: int = 0
    region_size_variance: float = 0.0
    detail_level: float = 0.0
    frequency_energy: Dict[str, float] = field(default_factory=dict)
    corner_density: float = 0.0
    line_complexity: float = 0.0


@dataclass
class ColorComplexity:
    """Color complexity metrics"""
    unique_colors: int = 0
    color_entropy: float = 0.0
    dominant_colors: int = 0
    color_variance: float = 0.0
    gradient_strength: float = 0.0
    color_harmony_score: float = 0.0
    saturation_distribution: float = 0.0
    hue_diversity: float = 0.0
    luminance_range: float = 0.0
    color_temperature_variance: float = 0.0


@dataclass
class InformationMetrics:
    """Information theory based complexity metrics"""
    shannon_entropy: float = 0.0
    kolmogorov_estimate: float = 0.0
    compression_ratio: float = 0.0
    fractal_dimension: float = 0.0
    mutual_information: float = 0.0
    redundancy: float = 0.0
    joint_entropy: float = 0.0
    conditional_entropy: float = 0.0


@dataclass
class CompositionalComplexity:
    """Compositional and layout complexity"""
    rule_of_thirds_score: float = 0.0
    symmetry_score: float = 0.0
    balance_score: float = 0.0
    focal_points: int = 0
    visual_weight_distribution: float = 0.0
    depth_layers: int = 0
    perspective_complexity: float = 0.0
    compositional_tension: float = 0.0
    negative_space_ratio: float = 0.0
    visual_hierarchy_score: float = 0.0


@dataclass
class TemporalComplexity:
    """Temporal complexity over time windows"""
    motion_complexity: float = 0.0
    change_frequency: float = 0.0
    temporal_coherence: float = 0.0
    rhythm_score: float = 0.0
    pace_variance: float = 0.0
    temporal_patterns: List[str] = field(default_factory=list)
    stability_score: float = 0.0
    predictability: float = 0.0


@dataclass
class SceneComplexityResult:
    """Complete scene complexity analysis result"""
    spatial: SpatialComplexity
    color: ColorComplexity
    information: InformationMetrics
    compositional: CompositionalComplexity
    temporal: TemporalComplexity
    overall_complexity: float = 0.0
    complexity_category: str = ""
    complexity_profile: Dict[str, float] = field(default_factory=dict)
    cognitive_load_estimate: float = 0.0
    visual_richness: float = 0.0
    perceptual_difficulty: float = 0.0
    attention_demand: float = 0.0


class SceneComplexityAnalyzer:
    """Analyzes scene complexity across multiple dimensions"""
    
    def __init__(self, 
                 temporal_window: int = 30,
                 spatial_scales: List[float] = None,
                 complexity_thresholds: Optional[Dict[str, float]] = None):
        """
        Initialize scene complexity analyzer
        
        Args:
            temporal_window: Number of frames for temporal analysis
            spatial_scales: Scales for multi-scale spatial analysis
            complexity_thresholds: Thresholds for categorizing complexity levels
        """
        self.temporal_window = temporal_window
        self.spatial_scales = spatial_scales or [1.0, 0.5, 0.25]
        self.complexity_thresholds = complexity_thresholds or {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.8,
            'very_high': 0.95
        }
        
        self.frame_buffer = []
        self.complexity_history = []
        
    def analyze(self, frame: np.ndarray, 
                optical_flow: Optional[np.ndarray] = None) -> SceneComplexityResult:
        """
        Analyze scene complexity for a frame
        
        Args:
            frame: Input frame (H, W, 3)
            optical_flow: Optional optical flow field
            
        Returns:
            Complete scene complexity analysis
        """
        # Update buffers
        self._update_buffers(frame, optical_flow)
        
        # Analyze different complexity dimensions
        spatial = self._analyze_spatial_complexity(frame)
        color = self._analyze_color_complexity(frame)
        information = self._analyze_information_metrics(frame)
        compositional = self._analyze_compositional_complexity(frame)
        temporal = self._analyze_temporal_complexity()
        
        # Calculate overall complexity
        overall_complexity = self._calculate_overall_complexity(
            spatial, color, information, compositional, temporal
        )
        
        # Determine complexity category
        complexity_category = self._categorize_complexity(overall_complexity)
        
        # Create complexity profile
        complexity_profile = self._create_complexity_profile(
            spatial, color, information, compositional, temporal
        )
        
        # Estimate cognitive metrics
        cognitive_load = self._estimate_cognitive_load(complexity_profile)
        visual_richness = self._estimate_visual_richness(complexity_profile)
        perceptual_difficulty = self._estimate_perceptual_difficulty(complexity_profile)
        attention_demand = self._estimate_attention_demand(complexity_profile)
        
        return SceneComplexityResult(
            spatial=spatial,
            color=color,
            information=information,
            compositional=compositional,
            temporal=temporal,
            overall_complexity=overall_complexity,
            complexity_category=complexity_category,
            complexity_profile=complexity_profile,
            cognitive_load_estimate=cognitive_load,
            visual_richness=visual_richness,
            perceptual_difficulty=perceptual_difficulty,
            attention_demand=attention_demand
        )
    
    def _update_buffers(self, frame: np.ndarray, optical_flow: Optional[np.ndarray]):
        """Update frame and flow buffers"""
        self.frame_buffer.append({
            'frame': frame.copy(),
            'flow': optical_flow.copy() if optical_flow is not None else None,
            'timestamp': len(self.frame_buffer)
        })
        
        # Keep only temporal window
        if len(self.frame_buffer) > self.temporal_window:
            self.frame_buffer.pop(0)
    
    def _analyze_spatial_complexity(self, frame: np.ndarray) -> SpatialComplexity:
        """Analyze spatial complexity of the frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Edge analysis
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Edge distribution entropy
        edge_hist, _ = np.histogram(edges.flatten(), bins=256)
        edge_hist = edge_hist[edge_hist > 0]
        if len(edge_hist) > 0:
            edge_probs = edge_hist / edge_hist.sum()
            edge_distribution_entropy = -np.sum(edge_probs * np.log2(edge_probs + 1e-10))
        else:
            edge_distribution_entropy = 0.0
        
        # Texture analysis using Local Binary Patterns
        texture_richness, texture_variety = self._analyze_texture(gray)
        
        # Region segmentation
        region_count, region_size_variance = self._analyze_regions(frame)
        
        # Detail level using Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        detail_level = laplacian.var()
        
        # Frequency analysis
        frequency_energy = self._analyze_frequency_content(gray)
        
        # Corner detection
        corners = cv2.goodFeaturesToTrack(gray, 500, 0.01, 10)
        corner_density = len(corners) / (gray.shape[0] * gray.shape[1]) if corners is not None else 0
        
        # Line complexity using Hough transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=30, maxLineGap=10)
        line_complexity = len(lines) / 100.0 if lines is not None else 0
        
        return SpatialComplexity(
            edge_density=edge_density,
            edge_distribution_entropy=edge_distribution_entropy,
            texture_richness=texture_richness,
            texture_variety=texture_variety,
            region_count=region_count,
            region_size_variance=region_size_variance,
            detail_level=detail_level,
            frequency_energy=frequency_energy,
            corner_density=corner_density,
            line_complexity=min(line_complexity, 1.0)
        )
    
    def _analyze_texture(self, gray: np.ndarray) -> Tuple[float, float]:
        """Analyze texture using GLCM-like features"""
        # Simplified texture analysis using gradients
        dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Texture richness from gradient magnitude
        magnitude = np.sqrt(dx**2 + dy**2)
        texture_richness = np.std(magnitude) / (np.mean(magnitude) + 1e-10)
        
        # Texture variety from gradient orientations
        orientation = np.arctan2(dy, dx)
        hist, _ = np.histogram(orientation, bins=36, range=(-np.pi, np.pi))
        hist_probs = hist / hist.sum()
        texture_variety = -np.sum(hist_probs * np.log2(hist_probs + 1e-10))
        
        return texture_richness, texture_variety / np.log2(36)  # Normalize
    
    def _analyze_regions(self, frame: np.ndarray) -> Tuple[int, float]:
        """Analyze region segmentation"""
        # Simple region segmentation using watershed or mean-shift
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Use adaptive thresholding and connected components
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
        
        num_labels, labels = cv2.connectedComponents(thresh)
        
        # Calculate region size variance
        region_sizes = []
        for i in range(1, num_labels):
            size = np.sum(labels == i)
            if size > 100:  # Filter small regions
                region_sizes.append(size)
        
        if len(region_sizes) > 0:
            region_size_variance = np.var(region_sizes) / (np.mean(region_sizes)**2 + 1e-10)
        else:
            region_size_variance = 0.0
        
        return len(region_sizes), region_size_variance
    
    def _analyze_frequency_content(self, gray: np.ndarray) -> Dict[str, float]:
        """Analyze frequency content using DCT"""
        # Apply DCT
        dct_result = dct2(gray.astype(float))
        
        # Divide into frequency bands
        h, w = dct_result.shape
        
        # Low frequency (top-left corner)
        low_freq = dct_result[:h//4, :w//4]
        low_energy = np.sum(low_freq**2)
        
        # Mid frequency
        mid_freq = dct_result[h//4:h//2, w//4:w//2]
        mid_energy = np.sum(mid_freq**2)
        
        # High frequency
        high_freq = dct_result[h//2:, w//2:]
        high_energy = np.sum(high_freq**2)
        
        total_energy = low_energy + mid_energy + high_energy + 1e-10
        
        return {
            'low': low_energy / total_energy,
            'mid': mid_energy / total_energy,
            'high': high_energy / total_energy
        }
    
    def _analyze_color_complexity(self, frame: np.ndarray) -> ColorComplexity:
        """Analyze color complexity"""
        # Convert to different color spaces
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        
        # Count unique colors (quantized)
        quantized = (frame // 32) * 32
        unique_colors = len(np.unique(quantized.reshape(-1, 3), axis=0))
        
        # Color entropy
        color_hist = cv2.calcHist([frame], [0, 1, 2], None, [16, 16, 16], [0, 256, 0, 256, 0, 256])
        color_hist = color_hist.flatten()
        color_hist = color_hist[color_hist > 0]
        color_probs = color_hist / color_hist.sum()
        color_entropy = -np.sum(color_probs * np.log2(color_probs + 1e-10))
        
        # Dominant colors using k-means
        pixels = frame.reshape(-1, 3)
        sample_pixels = pixels[np.random.choice(pixels.shape[0], min(1000, pixels.shape[0]), replace=False)]
        
        try:
            kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
            kmeans.fit(sample_pixels)
            dominant_colors = len(np.unique(kmeans.labels_))
        except:
            dominant_colors = 1
        
        # Color variance
        color_variance = np.mean(np.var(frame.reshape(-1, 3), axis=0))
        
        # Gradient strength
        gradient_strength = self._calculate_gradient_strength(frame)
        
        # Color harmony score
        color_harmony_score = self._calculate_color_harmony(hsv)
        
        # Saturation distribution
        saturation = hsv[:, :, 1]
        saturation_distribution = np.std(saturation) / (np.mean(saturation) + 1e-10)
        
        # Hue diversity
        hue = hsv[:, :, 0]
        hue_hist, _ = np.histogram(hue, bins=36, range=(0, 180))
        hue_probs = hue_hist / hue_hist.sum()
        hue_diversity = -np.sum(hue_probs * np.log2(hue_probs + 1e-10))
        
        # Luminance range
        luminance = lab[:, :, 0]
        luminance_range = (np.max(luminance) - np.min(luminance)) / 255.0
        
        # Color temperature variance (simplified)
        color_temperature_variance = self._calculate_color_temperature_variance(frame)
        
        return ColorComplexity(
            unique_colors=unique_colors,
            color_entropy=color_entropy / np.log2(4096),  # Normalize
            dominant_colors=dominant_colors,
            color_variance=color_variance / 255.0,
            gradient_strength=gradient_strength,
            color_harmony_score=color_harmony_score,
            saturation_distribution=saturation_distribution,
            hue_diversity=hue_diversity / np.log2(36),  # Normalize
            luminance_range=luminance_range,
            color_temperature_variance=color_temperature_variance
        )
    
    def _calculate_gradient_strength(self, frame: np.ndarray) -> float:
        """Calculate color gradient strength"""
        # Calculate gradients in each channel
        gradients = []
        for i in range(3):
            dx = cv2.Sobel(frame[:, :, i], cv2.CV_64F, 1, 0, ksize=3)
            dy = cv2.Sobel(frame[:, :, i], cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(dx**2 + dy**2)
            gradients.append(magnitude)
        
        # Average gradient strength
        avg_gradient = np.mean(gradients)
        return min(avg_gradient / 100.0, 1.0)  # Normalize
    
    def _calculate_color_harmony(self, hsv: np.ndarray) -> float:
        """Calculate color harmony score based on color wheel relationships"""
        hue = hsv[:, :, 0].flatten()
        
        # Sample hues
        sample_size = min(1000, len(hue))
        sample_hues = hue[np.random.choice(len(hue), sample_size, replace=False)]
        
        # Check for common color harmonies
        harmony_scores = []
        
        # Complementary (180 degrees apart)
        for h in sample_hues[:100]:
            complement = (h + 90) % 180
            distances = np.abs(sample_hues - complement)
            distances = np.minimum(distances, 180 - distances)
            harmony_scores.append(np.sum(distances < 15) / len(sample_hues))
        
        # Triadic (120 degrees apart)
        for h in sample_hues[:100]:
            triad1 = (h + 60) % 180
            triad2 = (h + 120) % 180
            distances1 = np.abs(sample_hues - triad1)
            distances1 = np.minimum(distances1, 180 - distances1)
            distances2 = np.abs(sample_hues - triad2)
            distances2 = np.minimum(distances2, 180 - distances2)
            harmony_scores.append((np.sum(distances1 < 15) + np.sum(distances2 < 15)) / (2 * len(sample_hues)))
        
        return np.mean(harmony_scores) if harmony_scores else 0.0
    
    def _calculate_color_temperature_variance(self, frame: np.ndarray) -> float:
        """Calculate variance in color temperature"""
        # Simplified color temperature estimation
        r, g, b = frame[:, :, 2], frame[:, :, 1], frame[:, :, 0]
        
        # Estimate temperature from R/B ratio
        with np.errstate(divide='ignore', invalid='ignore'):
            temp_map = r / (b + 1e-10)
            temp_map = np.nan_to_num(temp_map, nan=1.0, posinf=10.0, neginf=0.1)
        
        # Calculate variance
        variance = np.var(temp_map)
        return min(variance / 10.0, 1.0)  # Normalize
    
    def _analyze_information_metrics(self, frame: np.ndarray) -> InformationMetrics:
        """Calculate information theory based metrics"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Shannon entropy
        hist, _ = np.histogram(gray, bins=256, range=(0, 256))
        hist = hist[hist > 0]
        probs = hist / hist.sum()
        shannon_entropy = -np.sum(probs * np.log2(probs + 1e-10))
        
        # Kolmogorov complexity estimate (using compression ratio)
        kolmogorov_estimate = self._estimate_kolmogorov_complexity(gray)
        
        # Compression ratio (simplified)
        compression_ratio = self._calculate_compression_ratio(gray)
        
        # Fractal dimension
        fractal_dimension = self._calculate_fractal_dimension(gray)
        
        # Mutual information between color channels
        mutual_information = self._calculate_mutual_information(frame)
        
        # Redundancy
        max_entropy = np.log2(256)
        redundancy = 1.0 - (shannon_entropy / max_entropy)
        
        # Joint entropy
        joint_entropy = self._calculate_joint_entropy(frame)
        
        # Conditional entropy
        conditional_entropy = joint_entropy - shannon_entropy
        
        return InformationMetrics(
            shannon_entropy=shannon_entropy / max_entropy,  # Normalize
            kolmogorov_estimate=kolmogorov_estimate,
            compression_ratio=compression_ratio,
            fractal_dimension=fractal_dimension / 3.0,  # Normalize (max ~3 for 2D)
            mutual_information=mutual_information,
            redundancy=redundancy,
            joint_entropy=joint_entropy / (3 * max_entropy),  # Normalize
            conditional_entropy=conditional_entropy / max_entropy  # Normalize
        )
    
    def _estimate_kolmogorov_complexity(self, gray: np.ndarray) -> float:
        """Estimate Kolmogorov complexity using LZ complexity"""
        # Convert to binary string
        binary = (gray > np.median(gray)).astype(np.uint8)
        binary_str = ''.join(binary.flatten().astype(str))
        
        # Simplified LZ complexity
        n = len(binary_str)
        i = 0
        complexity = 0
        
        while i < n:
            j = i + 1
            while j <= n and binary_str[i:j] in binary_str[:i]:
                j += 1
            complexity += 1
            i = j - 1 if j <= n else n
        
        # Normalize
        return complexity / (n / np.log2(n + 1) + 1)
    
    def _calculate_compression_ratio(self, gray: np.ndarray) -> float:
        """Calculate compression ratio using run-length encoding"""
        flat = gray.flatten()
        
        # Simple RLE
        rle_length = 0
        current_val = flat[0]
        
        for val in flat[1:]:
            if val != current_val:
                rle_length += 2  # Value and count
                current_val = val
        
        rle_length += 2  # Last run
        
        compression_ratio = len(flat) / rle_length
        return min(compression_ratio / 10.0, 1.0)  # Normalize
    
    def _calculate_fractal_dimension(self, gray: np.ndarray) -> float:
        """Calculate fractal dimension using box-counting method"""
        # Binarize
        binary = gray > np.median(gray)
        
        # Box counting
        sizes = [2, 4, 8, 16, 32, 64]
        counts = []
        
        for size in sizes:
            h, w = binary.shape
            h_boxes = h // size
            w_boxes = w // size
            
            count = 0
            for i in range(h_boxes):
                for j in range(w_boxes):
                    box = binary[i*size:(i+1)*size, j*size:(j+1)*size]
                    if np.any(box):
                        count += 1
            
            counts.append(count)
        
        # Fit log-log relationship
        if len(counts) > 1 and np.any(counts):
            log_sizes = np.log(sizes[:len(counts)])
            log_counts = np.log(np.array(counts) + 1)
            
            # Linear regression
            coeffs = np.polyfit(log_sizes, log_counts, 1)
            fractal_dimension = -coeffs[0]
        else:
            fractal_dimension = 2.0
        
        return fractal_dimension
    
    def _calculate_mutual_information(self, frame: np.ndarray) -> float:
        """Calculate mutual information between color channels"""
        # Simplified mutual information between R and G channels
        r = frame[:, :, 2].flatten()
        g = frame[:, :, 1].flatten()
        
        # Quantize
        r_q = r // 16
        g_q = g // 16
        
        # Joint histogram
        hist_2d, _, _ = np.histogram2d(r_q, g_q, bins=16)
        
        # Marginal distributions
        p_r = hist_2d.sum(axis=1)
        p_g = hist_2d.sum(axis=0)
        
        # Normalize
        hist_2d = hist_2d / hist_2d.sum()
        p_r = p_r / p_r.sum()
        p_g = p_g / p_g.sum()
        
        # Calculate MI
        mi = 0
        for i in range(16):
            for j in range(16):
                if hist_2d[i, j] > 0 and p_r[i] > 0 and p_g[j] > 0:
                    mi += hist_2d[i, j] * np.log2(hist_2d[i, j] / (p_r[i] * p_g[j]))
        
        return mi / np.log2(16)  # Normalize
    
    def _calculate_joint_entropy(self, frame: np.ndarray) -> float:
        """Calculate joint entropy of color channels"""
        # Quantize colors
        quantized = frame // 16
        
        # Create joint histogram
        pixels = quantized.reshape(-1, 3)
        unique, counts = np.unique(pixels, axis=0, return_counts=True)
        
        # Calculate entropy
        probs = counts / counts.sum()
        joint_entropy = -np.sum(probs * np.log2(probs + 1e-10))
        
        return joint_entropy
    
    def _analyze_compositional_complexity(self, frame: np.ndarray) -> CompositionalComplexity:
        """Analyze compositional complexity"""
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Rule of thirds analysis
        rule_of_thirds_score = self._calculate_rule_of_thirds(gray)
        
        # Symmetry analysis
        symmetry_score = self._calculate_symmetry(gray)
        
        # Balance analysis
        balance_score = self._calculate_balance(gray)
        
        # Focal points detection
        focal_points = self._detect_focal_points(gray)
        
        # Visual weight distribution
        visual_weight_distribution = self._calculate_visual_weight_distribution(gray)
        
        # Depth layers estimation
        depth_layers = self._estimate_depth_layers(frame)
        
        # Perspective complexity
        perspective_complexity = self._calculate_perspective_complexity(gray)
        
        # Compositional tension
        compositional_tension = self._calculate_compositional_tension(gray)
        
        # Negative space ratio
        negative_space_ratio = self._calculate_negative_space_ratio(gray)
        
        # Visual hierarchy
        visual_hierarchy_score = self._calculate_visual_hierarchy(gray)
        
        return CompositionalComplexity(
            rule_of_thirds_score=rule_of_thirds_score,
            symmetry_score=symmetry_score,
            balance_score=balance_score,
            focal_points=len(focal_points),
            visual_weight_distribution=visual_weight_distribution,
            depth_layers=depth_layers,
            perspective_complexity=perspective_complexity,
            compositional_tension=compositional_tension,
            negative_space_ratio=negative_space_ratio,
            visual_hierarchy_score=visual_hierarchy_score
        )
    
    def _calculate_rule_of_thirds(self, gray: np.ndarray) -> float:
        """Calculate rule of thirds compliance"""
        h, w = gray.shape
        
        # Define rule of thirds lines
        v_lines = [w // 3, 2 * w // 3]
        h_lines = [h // 3, 2 * h // 3]
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Check edge density near rule of thirds lines
        score = 0
        margin = 10
        
        for v_line in v_lines:
            region = edges[:, max(0, v_line-margin):min(w, v_line+margin)]
            score += np.sum(region > 0) / region.size
        
        for h_line in h_lines:
            region = edges[max(0, h_line-margin):min(h, h_line+margin), :]
            score += np.sum(region > 0) / region.size
        
        return score / 4
    
    def _calculate_symmetry(self, gray: np.ndarray) -> float:
        """Calculate symmetry score"""
        # Vertical symmetry
        flipped_v = cv2.flip(gray, 1)
        v_symmetry = 1.0 - np.mean(np.abs(gray - flipped_v)) / 255.0
        
        # Horizontal symmetry
        flipped_h = cv2.flip(gray, 0)
        h_symmetry = 1.0 - np.mean(np.abs(gray - flipped_h)) / 255.0
        
        # Diagonal symmetry
        flipped_d = cv2.transpose(gray)
        if flipped_d.shape == gray.shape:
            d_symmetry = 1.0 - np.mean(np.abs(gray - flipped_d)) / 255.0
        else:
            d_symmetry = 0.5
        
        return max(v_symmetry, h_symmetry, d_symmetry)
    
    def _calculate_balance(self, gray: np.ndarray) -> float:
        """Calculate visual balance"""
        h, w = gray.shape
        
        # Calculate center of mass
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        
        total_mass = np.sum(gray)
        if total_mass > 0:
            cx = np.sum(x_coords * gray) / total_mass
            cy = np.sum(y_coords * gray) / total_mass
            
            # Calculate distance from center
            center_x, center_y = w / 2, h / 2
            distance = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
            max_distance = np.sqrt(center_x**2 + center_y**2)
            
            balance = 1.0 - (distance / max_distance)
        else:
            balance = 0.5
        
        return balance
    
    def _detect_focal_points(self, gray: np.ndarray) -> List[Tuple[int, int]]:
        """Detect focal points using saliency"""
        # Simple saliency using Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        saliency = np.abs(laplacian)
        
        # Find peaks
        threshold = np.percentile(saliency, 95)
        peaks = saliency > threshold
        
        # Find connected components
        num_labels, labels = cv2.connectedComponents(peaks.astype(np.uint8))
        
        focal_points = []
        for i in range(1, num_labels):
            y, x = np.where(labels == i)
            if len(y) > 10:  # Minimum size
                focal_points.append((int(np.mean(x)), int(np.mean(y))))
        
        return focal_points[:10]  # Limit to 10 focal points
    
    def _calculate_visual_weight_distribution(self, gray: np.ndarray) -> float:
        """Calculate visual weight distribution uniformity"""
        # Divide into quadrants
        h, w = gray.shape
        h_mid, w_mid = h // 2, w // 2
        
        quadrants = [
            gray[:h_mid, :w_mid],
            gray[:h_mid, w_mid:],
            gray[h_mid:, :w_mid],
            gray[h_mid:, w_mid:]
        ]
        
        # Calculate weight for each quadrant
        weights = [np.mean(q) for q in quadrants]
        
        # Calculate uniformity
        if np.mean(weights) > 0:
            uniformity = 1.0 - (np.std(weights) / np.mean(weights))
        else:
            uniformity = 0.0
        
        return uniformity
    
    def _estimate_depth_layers(self, frame: np.ndarray) -> int:
        """Estimate number of depth layers"""
        # Use blur detection at different scales
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        layers = []
        for kernel_size in [3, 7, 11, 15]:
            blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
            diff = np.abs(gray.astype(float) - blurred.astype(float))
            
            # Threshold to find sharp regions
            threshold = np.percentile(diff, 75)
            sharp_regions = diff > threshold
            
            # Count connected components
            num_labels, _ = cv2.connectedComponents(sharp_regions.astype(np.uint8))
            layers.append(num_labels - 1)
        
        # Estimate depth layers from scale analysis
        depth_layers = int(np.median(layers))
        return min(depth_layers, 10)  # Cap at 10 layers
    
    def _calculate_perspective_complexity(self, gray: np.ndarray) -> float:
        """Calculate perspective complexity using line analysis"""
        edges = cv2.Canny(gray, 50, 150)
        
        # Detect lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=30, maxLineGap=10)
        
        if lines is not None and len(lines) > 2:
            # Analyze line angles
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1)
                angles.append(angle)
            
            # Check for convergence (perspective)
            angle_std = np.std(angles)
            
            # More variation in angles suggests perspective
            perspective_complexity = min(angle_std / (np.pi / 4), 1.0)
        else:
            perspective_complexity = 0.0
        
        return perspective_complexity
    
    def _calculate_compositional_tension(self, gray: np.ndarray) -> float:
        """Calculate compositional tension from element placement"""
        # Find high contrast regions
        edges = cv2.Canny(gray, 50, 150)
        
        # Calculate tension from edge distribution
        h, w = gray.shape
        
        # Divide into regions
        regions = []
        for i in range(3):
            for j in range(3):
                region = edges[i*h//3:(i+1)*h//3, j*w//3:(j+1)*w//3]
                regions.append(np.sum(region > 0))
        
        # Tension from uneven distribution
        if np.sum(regions) > 0:
            tension = np.std(regions) / np.mean(regions)
        else:
            tension = 0.0
        
        return min(tension, 1.0)
    
    def _calculate_negative_space_ratio(self, gray: np.ndarray) -> float:
        """Calculate ratio of negative (empty) space"""
        # Detect content using adaptive threshold
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
        
        # Morphological operations to connect content
        kernel = np.ones((5, 5), np.uint8)
        content = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Calculate negative space
        negative_space = np.sum(content == 0)
        total_space = content.size
        
        return negative_space / total_space
    
    def _calculate_visual_hierarchy(self, gray: np.ndarray) -> float:
        """Calculate visual hierarchy score"""
        # Detect different sized elements
        element_sizes = []
        
        for kernel_size in [3, 7, 15, 31]:
            # Top-hat transform to find elements of different sizes
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
            
            # Count significant elements
            threshold = np.percentile(tophat, 90)
            significant = tophat > threshold
            
            num_labels, _ = cv2.connectedComponents(significant.astype(np.uint8))
            element_sizes.append(num_labels - 1)
        
        # Hierarchy from size distribution
        if len(element_sizes) > 1 and np.sum(element_sizes) > 0:
            # Good hierarchy has elements at multiple scales
            hierarchy_score = len([s for s in element_sizes if s > 0]) / len(element_sizes)
        else:
            hierarchy_score = 0.0
        
        return hierarchy_score
    
    def _analyze_temporal_complexity(self) -> TemporalComplexity:
        """Analyze temporal complexity over buffer"""
        if len(self.frame_buffer) < 2:
            return TemporalComplexity()
        
        # Extract motion complexity
        motion_complexity = self._calculate_motion_complexity()
        
        # Change frequency
        change_frequency = self._calculate_change_frequency()
        
        # Temporal coherence
        temporal_coherence = self._calculate_temporal_coherence()
        
        # Rhythm score
        rhythm_score = self._calculate_rhythm_score()
        
        # Pace variance
        pace_variance = self._calculate_pace_variance()
        
        # Temporal patterns
        temporal_patterns = self._detect_temporal_patterns()
        
        # Stability score
        stability_score = self._calculate_stability_score()
        
        # Predictability
        predictability = self._calculate_predictability()
        
        return TemporalComplexity(
            motion_complexity=motion_complexity,
            change_frequency=change_frequency,
            temporal_coherence=temporal_coherence,
            rhythm_score=rhythm_score,
            pace_variance=pace_variance,
            temporal_patterns=temporal_patterns,
            stability_score=stability_score,
            predictability=predictability
        )
    
    def _calculate_motion_complexity(self) -> float:
        """Calculate complexity of motion patterns"""
        if len(self.frame_buffer) < 2:
            return 0.0
        
        # Analyze optical flow patterns
        flow_complexities = []
        
        for i in range(len(self.frame_buffer) - 1):
            if self.frame_buffer[i]['flow'] is not None:
                flow = self.frame_buffer[i]['flow']
                
                # Calculate flow statistics
                magnitude = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
                angle = np.arctan2(flow[:, :, 1], flow[:, :, 0])
                
                # Complexity from magnitude variance and angle diversity
                mag_complexity = np.std(magnitude) / (np.mean(magnitude) + 1e-10)
                
                angle_hist, _ = np.histogram(angle, bins=36, range=(-np.pi, np.pi))
                angle_probs = angle_hist / (angle_hist.sum() + 1e-10)
                angle_complexity = -np.sum(angle_probs * np.log2(angle_probs + 1e-10))
                
                flow_complexities.append((mag_complexity + angle_complexity) / 2)
        
        return np.mean(flow_complexities) if flow_complexities else 0.0
    
    def _calculate_change_frequency(self) -> float:
        """Calculate frequency of visual changes"""
        if len(self.frame_buffer) < 2:
            return 0.0
        
        changes = []
        
        for i in range(len(self.frame_buffer) - 1):
            frame1 = self.frame_buffer[i]['frame']
            frame2 = self.frame_buffer[i + 1]['frame']
            
            # Calculate frame difference
            diff = np.mean(np.abs(frame1.astype(float) - frame2.astype(float)))
            changes.append(diff > 10)  # Threshold for significant change
        
        return np.mean(changes) if changes else 0.0
    
    def _calculate_temporal_coherence(self) -> float:
        """Calculate temporal coherence"""
        if len(self.frame_buffer) < 3:
            return 1.0
        
        coherences = []
        
        for i in range(len(self.frame_buffer) - 2):
            frame1 = self.frame_buffer[i]['frame']
            frame2 = self.frame_buffer[i + 1]['frame']
            frame3 = self.frame_buffer[i + 2]['frame']
            
            # Check if changes are consistent
            diff1 = frame2.astype(float) - frame1.astype(float)
            diff2 = frame3.astype(float) - frame2.astype(float)
            
            # Correlation between consecutive differences
            correlation = np.corrcoef(diff1.flatten(), diff2.flatten())[0, 1]
            coherences.append(abs(correlation))
        
        return np.mean(coherences) if coherences else 0.0
    
    def _calculate_rhythm_score(self) -> float:
        """Calculate rhythm in visual changes"""
        if len(self.frame_buffer) < 4:
            return 0.0
        
        # Calculate inter-frame differences
        differences = []
        
        for i in range(len(self.frame_buffer) - 1):
            frame1 = self.frame_buffer[i]['frame']
            frame2 = self.frame_buffer[i + 1]['frame']
            diff = np.mean(np.abs(frame1.astype(float) - frame2.astype(float)))
            differences.append(diff)
        
        if len(differences) > 4:
            # Look for periodicity using autocorrelation
            differences = np.array(differences)
            autocorr = np.correlate(differences - np.mean(differences), 
                                   differences - np.mean(differences), mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr / autocorr[0]
            
            # Find peaks in autocorrelation
            peaks = []
            for i in range(1, len(autocorr) - 1):
                if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1] and autocorr[i] > 0.5:
                    peaks.append(i)
            
            # Rhythm score from peak regularity
            if len(peaks) > 1:
                peak_intervals = np.diff(peaks)
                rhythm_score = 1.0 - (np.std(peak_intervals) / (np.mean(peak_intervals) + 1e-10))
            else:
                rhythm_score = 0.0
        else:
            rhythm_score = 0.0
        
        return max(0, min(rhythm_score, 1.0))
    
    def _calculate_pace_variance(self) -> float:
        """Calculate variance in pacing"""
        if len(self.frame_buffer) < 3:
            return 0.0
        
        # Calculate rate of change
        change_rates = []
        
        for i in range(len(self.frame_buffer) - 2):
            frame1 = self.frame_buffer[i]['frame']
            frame2 = self.frame_buffer[i + 1]['frame']
            frame3 = self.frame_buffer[i + 2]['frame']
            
            diff1 = np.mean(np.abs(frame2.astype(float) - frame1.astype(float)))
            diff2 = np.mean(np.abs(frame3.astype(float) - frame2.astype(float)))
            
            change_rate = abs(diff2 - diff1)
            change_rates.append(change_rate)
        
        if change_rates:
            variance = np.var(change_rates) / (np.mean(change_rates)**2 + 1e-10)
            return min(variance, 1.0)
        
        return 0.0
    
    def _detect_temporal_patterns(self) -> List[str]:
        """Detect temporal patterns in the buffer"""
        patterns = []
        
        if len(self.frame_buffer) < 4:
            return patterns
        
        # Check for different pattern types
        differences = []
        for i in range(len(self.frame_buffer) - 1):
            frame1 = self.frame_buffer[i]['frame']
            frame2 = self.frame_buffer[i + 1]['frame']
            diff = np.mean(np.abs(frame1.astype(float) - frame2.astype(float)))
            differences.append(diff)
        
        differences = np.array(differences)
        
        # Static pattern
        if np.std(differences) < 5:
            patterns.append("static")
        
        # Oscillating pattern
        if len(differences) > 4:
            sign_changes = np.sum(np.diff(np.sign(np.diff(differences))) != 0)
            if sign_changes > len(differences) * 0.5:
                patterns.append("oscillating")
        
        # Increasing pattern
        if len(differences) > 3:
            trend = np.polyfit(range(len(differences)), differences, 1)[0]
            if trend > 1:
                patterns.append("accelerating")
            elif trend < -1:
                patterns.append("decelerating")
        
        # Periodic pattern
        if len(differences) > 8:
            fft = np.fft.fft(differences)
            power = np.abs(fft)**2
            freq_peak = np.argmax(power[1:len(power)//2]) + 1
            if power[freq_peak] > np.mean(power) * 3:
                patterns.append("periodic")
        
        return patterns
    
    def _calculate_stability_score(self) -> float:
        """Calculate visual stability"""
        if len(self.frame_buffer) < 2:
            return 1.0
        
        # Calculate cumulative change
        total_change = 0
        
        for i in range(len(self.frame_buffer) - 1):
            frame1 = self.frame_buffer[i]['frame']
            frame2 = self.frame_buffer[i + 1]['frame']
            
            diff = np.mean(np.abs(frame1.astype(float) - frame2.astype(float)))
            total_change += diff
        
        avg_change = total_change / (len(self.frame_buffer) - 1)
        
        # Stability inversely related to change
        stability = 1.0 - min(avg_change / 50.0, 1.0)
        
        return stability
    
    def _calculate_predictability(self) -> float:
        """Calculate predictability of visual changes"""
        if len(self.frame_buffer) < 5:
            return 0.5
        
        # Use simple prediction model
        predictions_correct = 0
        total_predictions = 0
        
        for i in range(2, len(self.frame_buffer) - 1):
            # Predict next frame based on linear extrapolation
            frame1 = self.frame_buffer[i - 2]['frame'].astype(float)
            frame2 = self.frame_buffer[i - 1]['frame'].astype(float)
            frame3 = self.frame_buffer[i]['frame'].astype(float)
            
            # Simple prediction: frame3 ≈ 2*frame2 - frame1
            predicted = 2 * frame2 - frame1
            predicted = np.clip(predicted, 0, 255)
            
            # Calculate prediction error
            error = np.mean(np.abs(predicted - frame3))
            
            # Count as correct if error is small
            if error < 20:
                predictions_correct += 1
            total_predictions += 1
        
        if total_predictions > 0:
            predictability = predictions_correct / total_predictions
        else:
            predictability = 0.5
        
        return predictability
    
    def _calculate_overall_complexity(self, 
                                    spatial: SpatialComplexity,
                                    color: ColorComplexity,
                                    information: InformationMetrics,
                                    compositional: CompositionalComplexity,
                                    temporal: TemporalComplexity) -> float:
        """Calculate overall complexity score"""
        # Weight different complexity dimensions
        weights = {
            'spatial': 0.25,
            'color': 0.20,
            'information': 0.20,
            'compositional': 0.20,
            'temporal': 0.15
        }
        
        # Calculate spatial score
        spatial_score = np.mean([
            spatial.edge_density,
            spatial.edge_distribution_entropy,
            spatial.texture_richness,
            spatial.detail_level / 1000.0,  # Normalize
            spatial.frequency_energy.get('high', 0)
        ])
        
        # Calculate color score
        color_score = np.mean([
            color.color_entropy,
            color.color_variance,
            color.hue_diversity,
            color.gradient_strength
        ])
        
        # Calculate information score
        info_score = np.mean([
            information.shannon_entropy,
            information.fractal_dimension,
            information.mutual_information
        ])
        
        # Calculate compositional score
        comp_score = np.mean([
            1.0 - compositional.symmetry_score,  # Asymmetry adds complexity
            compositional.focal_points / 5.0,  # Normalize
            compositional.perspective_complexity,
            compositional.compositional_tension
        ])
        
        # Calculate temporal score
        temp_score = np.mean([
            temporal.motion_complexity,
            temporal.change_frequency,
            1.0 - temporal.predictability
        ])
        
        # Weighted average
        overall = (weights['spatial'] * spatial_score +
                  weights['color'] * color_score +
                  weights['information'] * info_score +
                  weights['compositional'] * comp_score +
                  weights['temporal'] * temp_score)
        
        return min(overall, 1.0)
    
    def _categorize_complexity(self, overall_complexity: float) -> str:
        """Categorize complexity level"""
        if overall_complexity < self.complexity_thresholds['low']:
            return "minimal"
        elif overall_complexity < self.complexity_thresholds['medium']:
            return "low"
        elif overall_complexity < self.complexity_thresholds['high']:
            return "medium"
        elif overall_complexity < self.complexity_thresholds['very_high']:
            return "high"
        else:
            return "very_high"
    
    def _create_complexity_profile(self,
                                  spatial: SpatialComplexity,
                                  color: ColorComplexity,
                                  information: InformationMetrics,
                                  compositional: CompositionalComplexity,
                                  temporal: TemporalComplexity) -> Dict[str, float]:
        """Create detailed complexity profile"""
        return {
            'edge_complexity': spatial.edge_density,
            'texture_complexity': spatial.texture_richness,
            'detail_complexity': min(spatial.detail_level / 1000.0, 1.0),
            'color_diversity': color.color_entropy,
            'color_richness': color.hue_diversity,
            'information_density': information.shannon_entropy,
            'structural_complexity': information.fractal_dimension,
            'compositional_sophistication': 1.0 - compositional.symmetry_score,
            'visual_balance': compositional.balance_score,
            'temporal_dynamics': temporal.motion_complexity,
            'change_rate': temporal.change_frequency,
            'predictability': temporal.predictability
        }
    
    def _estimate_cognitive_load(self, profile: Dict[str, float]) -> float:
        """Estimate cognitive load from complexity profile"""
        # Factors that increase cognitive load
        load_factors = [
            profile.get('edge_complexity', 0) * 1.2,
            profile.get('texture_complexity', 0) * 1.1,
            profile.get('color_diversity', 0) * 0.9,
            profile.get('information_density', 0) * 1.3,
            profile.get('temporal_dynamics', 0) * 1.0,
            (1.0 - profile.get('predictability', 0.5)) * 1.2
        ]
        
        return min(np.mean(load_factors), 1.0)
    
    def _estimate_visual_richness(self, profile: Dict[str, float]) -> float:
        """Estimate visual richness from complexity profile"""
        richness_factors = [
            profile.get('texture_complexity', 0),
            profile.get('color_richness', 0),
            profile.get('detail_complexity', 0),
            profile.get('compositional_sophistication', 0)
        ]
        
        return min(np.mean(richness_factors) * 1.1, 1.0)
    
    def _estimate_perceptual_difficulty(self, profile: Dict[str, float]) -> float:
        """Estimate perceptual difficulty from complexity profile"""
        difficulty_factors = [
            profile.get('information_density', 0) * 1.2,
            profile.get('structural_complexity', 0) * 1.1,
            (1.0 - profile.get('visual_balance', 0.5)) * 0.8,
            profile.get('temporal_dynamics', 0) * 0.9
        ]
        
        return min(np.mean(difficulty_factors), 1.0)
    
    def _estimate_attention_demand(self, profile: Dict[str, float]) -> float:
        """Estimate attention demand from complexity profile"""
        attention_factors = [
            profile.get('edge_complexity', 0),
            profile.get('change_rate', 0) * 1.3,
            (1.0 - profile.get('predictability', 0.5)) * 1.1,
            profile.get('temporal_dynamics', 0) * 1.2
        ]
        
        return min(np.mean(attention_factors), 1.0)