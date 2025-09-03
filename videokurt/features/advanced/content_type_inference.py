"""
Content Type Inference

Infers the type of visual content using pattern recognition and statistical analysis:
- Document vs UI vs video content detection
- Animation vs live-action classification
- Presentation vs gameplay vs productivity content
- Static vs dynamic content categorization
- Content density and information type inference

This feature helps understand what kind of visual content is being analyzed
without semantic interpretation, focusing on visual patterns and statistics.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
import cv2
from scipy import stats, signal, ndimage
from scipy.spatial import distance
from scipy.fftpack import fft2, fftshift
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')


@dataclass
class VisualCharacteristics:
    """Visual characteristics of content"""
    edge_sharpness: float = 0.0
    color_uniformity: float = 0.0
    geometric_regularity: float = 0.0
    text_likelihood: float = 0.0
    gradient_smoothness: float = 0.0
    noise_level: float = 0.0
    contrast_distribution: float = 0.0
    spatial_frequency_profile: Dict[str, float] = field(default_factory=dict)
    texture_homogeneity: float = 0.0
    structural_alignment: float = 0.0


@dataclass
class MotionCharacteristics:
    """Motion characteristics of content"""
    motion_smoothness: float = 0.0
    motion_consistency: float = 0.0
    camera_motion_score: float = 0.0
    object_motion_score: float = 0.0
    motion_periodicity: float = 0.0
    stillness_ratio: float = 0.0
    motion_complexity: float = 0.0
    transition_frequency: float = 0.0
    motion_patterns: List[str] = field(default_factory=list)
    temporal_stability: float = 0.0


@dataclass
class LayoutCharacteristics:
    """Layout and composition characteristics"""
    grid_alignment: float = 0.0
    rectangular_regions: int = 0
    symmetry_score: float = 0.0
    layout_consistency: float = 0.0
    white_space_ratio: float = 0.0
    element_density: float = 0.0
    hierarchical_structure: float = 0.0
    alignment_score: float = 0.0
    repetitive_elements: int = 0
    layout_complexity: float = 0.0


@dataclass
class ColorCharacteristics:
    """Color-based content characteristics"""
    palette_type: str = ""  # "limited", "natural", "vibrant", "monochrome"
    color_consistency: float = 0.0
    saturation_profile: str = ""  # "low", "medium", "high", "mixed"
    luminance_distribution: str = ""  # "uniform", "bimodal", "normal", "skewed"
    color_temperature: str = ""  # "cool", "neutral", "warm", "mixed"
    gradient_presence: float = 0.0
    flat_color_ratio: float = 0.0
    color_transitions: str = ""  # "sharp", "smooth", "mixed"
    transparency_likelihood: float = 0.0
    color_depth_estimate: int = 0


@dataclass
class ContentPattern:
    """Detected content pattern"""
    pattern_type: str = ""
    confidence: float = 0.0
    characteristics: Dict[str, float] = field(default_factory=dict)
    spatial_distribution: np.ndarray = field(default_factory=lambda: np.array([]))
    temporal_consistency: float = 0.0


@dataclass
class ContentTypeResult:
    """Complete content type inference result"""
    primary_type: str = ""
    secondary_types: List[str] = field(default_factory=list)
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    visual_chars: VisualCharacteristics = field(default_factory=VisualCharacteristics)
    motion_chars: MotionCharacteristics = field(default_factory=MotionCharacteristics)
    layout_chars: LayoutCharacteristics = field(default_factory=LayoutCharacteristics)
    color_chars: ColorCharacteristics = field(default_factory=ColorCharacteristics)
    detected_patterns: List[ContentPattern] = field(default_factory=list)
    content_dynamics: str = ""  # "static", "semi-static", "dynamic", "highly-dynamic"
    information_density: str = ""  # "sparse", "moderate", "dense", "very-dense"
    visual_style: str = ""  # "minimalist", "detailed", "complex", "chaotic"
    interaction_likelihood: float = 0.0
    automation_likelihood: float = 0.0


class ContentTypeInferencer:
    """Infers content type from visual patterns and characteristics"""
    
    def __init__(self,
                 temporal_window: int = 30,
                 pattern_threshold: float = 0.7,
                 min_confidence: float = 0.5):
        """
        Initialize content type inferencer
        
        Args:
            temporal_window: Number of frames for temporal analysis
            pattern_threshold: Threshold for pattern detection
            min_confidence: Minimum confidence for type classification
        """
        self.temporal_window = temporal_window
        self.pattern_threshold = pattern_threshold
        self.min_confidence = min_confidence
        
        self.frame_buffer = []
        self.pattern_history = []
        self.type_history = []
        
        # Define content type signatures
        self._init_content_signatures()
        
    def _init_content_signatures(self):
        """Initialize content type signatures"""
        self.content_signatures = {
            'document': {
                'text_likelihood': (0.7, 1.0),
                'geometric_regularity': (0.7, 1.0),
                'white_space_ratio': (0.3, 0.8),
                'motion_consistency': (0.0, 0.2),
                'edge_sharpness': (0.6, 1.0)
            },
            'ui_interface': {
                'geometric_regularity': (0.6, 1.0),
                'grid_alignment': (0.5, 1.0),
                'rectangular_regions': (3, 50),
                'element_density': (0.3, 0.7),
                'flat_color_ratio': (0.4, 0.9)
            },
            'video_content': {
                'motion_smoothness': (0.5, 1.0),
                'camera_motion_score': (0.3, 1.0),
                'gradient_smoothness': (0.4, 0.9),
                'color_consistency': (0.3, 0.8),
                'temporal_stability': (0.4, 0.9)
            },
            'animation': {
                'flat_color_ratio': (0.5, 1.0),
                'edge_sharpness': (0.6, 1.0),
                'motion_smoothness': (0.6, 1.0),
                'color_consistency': (0.5, 1.0),
                'noise_level': (0.0, 0.3)
            },
            'presentation': {
                'text_likelihood': (0.5, 0.9),
                'geometric_regularity': (0.6, 1.0),
                'transition_frequency': (0.1, 0.4),
                'layout_consistency': (0.6, 1.0),
                'hierarchical_structure': (0.5, 1.0)
            },
            'gaming': {
                'motion_complexity': (0.5, 1.0),
                'object_motion_score': (0.4, 1.0),
                'color_consistency': (0.4, 0.9),
                'element_density': (0.4, 0.8),
                'temporal_stability': (0.3, 0.8)
            },
            'terminal': {
                'text_likelihood': (0.8, 1.0),
                'geometric_regularity': (0.8, 1.0),
                'flat_color_ratio': (0.8, 1.0),
                'motion_consistency': (0.0, 0.3),
                'palette_type': 'limited'
            },
            'diagram': {
                'geometric_regularity': (0.7, 1.0),
                'edge_sharpness': (0.7, 1.0),
                'white_space_ratio': (0.4, 0.8),
                'structural_alignment': (0.6, 1.0),
                'flat_color_ratio': (0.6, 1.0)
            }
        }
    
    def analyze(self, frame: np.ndarray,
                optical_flow: Optional[np.ndarray] = None) -> ContentTypeResult:
        """
        Analyze frame to infer content type
        
        Args:
            frame: Input frame (H, W, 3)
            optical_flow: Optional optical flow field
            
        Returns:
            Content type inference result
        """
        # Update buffers
        self._update_buffers(frame, optical_flow)
        
        # Extract characteristics
        visual_chars = self._analyze_visual_characteristics(frame)
        motion_chars = self._analyze_motion_characteristics(optical_flow)
        layout_chars = self._analyze_layout_characteristics(frame)
        color_chars = self._analyze_color_characteristics(frame)
        
        # Detect patterns
        patterns = self._detect_content_patterns(frame, visual_chars, layout_chars)
        
        # Infer content type
        type_scores = self._infer_content_type(
            visual_chars, motion_chars, layout_chars, color_chars, patterns
        )
        
        # Determine primary and secondary types
        primary_type, secondary_types = self._determine_types(type_scores)
        
        # Analyze content dynamics
        content_dynamics = self._analyze_content_dynamics(motion_chars)
        
        # Analyze information density
        information_density = self._analyze_information_density(
            visual_chars, layout_chars
        )
        
        # Determine visual style
        visual_style = self._determine_visual_style(
            visual_chars, color_chars, layout_chars
        )
        
        # Estimate interaction likelihood
        interaction_likelihood = self._estimate_interaction_likelihood(
            motion_chars, layout_chars, patterns
        )
        
        # Estimate automation likelihood
        automation_likelihood = self._estimate_automation_likelihood(
            motion_chars, patterns
        )
        
        # Update history
        self.type_history.append(primary_type)
        if len(self.type_history) > self.temporal_window:
            self.type_history.pop(0)
        
        return ContentTypeResult(
            primary_type=primary_type,
            secondary_types=secondary_types,
            confidence_scores=type_scores,
            visual_chars=visual_chars,
            motion_chars=motion_chars,
            layout_chars=layout_chars,
            color_chars=color_chars,
            detected_patterns=patterns,
            content_dynamics=content_dynamics,
            information_density=information_density,
            visual_style=visual_style,
            interaction_likelihood=interaction_likelihood,
            automation_likelihood=automation_likelihood
        )
    
    def _update_buffers(self, frame: np.ndarray, optical_flow: Optional[np.ndarray]):
        """Update frame and flow buffers"""
        self.frame_buffer.append({
            'frame': frame.copy(),
            'flow': optical_flow.copy() if optical_flow is not None else None,
            'timestamp': len(self.frame_buffer)
        })
        
        if len(self.frame_buffer) > self.temporal_window:
            self.frame_buffer.pop(0)
    
    def _analyze_visual_characteristics(self, frame: np.ndarray) -> VisualCharacteristics:
        """Analyze visual characteristics of the frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Edge sharpness
        edge_sharpness = self._calculate_edge_sharpness(gray)
        
        # Color uniformity
        color_uniformity = self._calculate_color_uniformity(frame)
        
        # Geometric regularity
        geometric_regularity = self._calculate_geometric_regularity(gray)
        
        # Text likelihood
        text_likelihood = self._calculate_text_likelihood(gray)
        
        # Gradient smoothness
        gradient_smoothness = self._calculate_gradient_smoothness(gray)
        
        # Noise level
        noise_level = self._calculate_noise_level(gray)
        
        # Contrast distribution
        contrast_distribution = self._calculate_contrast_distribution(gray)
        
        # Spatial frequency profile
        spatial_frequency_profile = self._analyze_spatial_frequencies(gray)
        
        # Texture homogeneity
        texture_homogeneity = self._calculate_texture_homogeneity(gray)
        
        # Structural alignment
        structural_alignment = self._calculate_structural_alignment(gray)
        
        return VisualCharacteristics(
            edge_sharpness=edge_sharpness,
            color_uniformity=color_uniformity,
            geometric_regularity=geometric_regularity,
            text_likelihood=text_likelihood,
            gradient_smoothness=gradient_smoothness,
            noise_level=noise_level,
            contrast_distribution=contrast_distribution,
            spatial_frequency_profile=spatial_frequency_profile,
            texture_homogeneity=texture_homogeneity,
            structural_alignment=structural_alignment
        )
    
    def _calculate_edge_sharpness(self, gray: np.ndarray) -> float:
        """Calculate edge sharpness metric"""
        # Sobel gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Gradient magnitude
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Sharpness from gradient statistics
        edges = magnitude > np.percentile(magnitude, 75)
        if np.sum(edges) > 0:
            edge_magnitudes = magnitude[edges]
            sharpness = np.mean(edge_magnitudes) / (np.std(edge_magnitudes) + 1e-10)
            return min(sharpness / 10.0, 1.0)
        
        return 0.0
    
    def _calculate_color_uniformity(self, frame: np.ndarray) -> float:
        """Calculate color uniformity across the frame"""
        # Divide into blocks
        h, w = frame.shape[:2]
        block_size = 32
        
        block_colors = []
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block = frame[i:i+block_size, j:j+block_size]
                mean_color = np.mean(block, axis=(0, 1))
                block_colors.append(mean_color)
        
        if len(block_colors) > 1:
            block_colors = np.array(block_colors)
            # Calculate variance across blocks
            color_variance = np.mean(np.var(block_colors, axis=0))
            uniformity = 1.0 - min(color_variance / 1000.0, 1.0)
            return uniformity
        
        return 0.5
    
    def _calculate_geometric_regularity(self, gray: np.ndarray) -> float:
        """Calculate geometric regularity from lines and shapes"""
        edges = cv2.Canny(gray, 50, 150)
        
        # Detect lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=30, maxLineGap=10)
        
        if lines is not None and len(lines) > 5:
            # Analyze line angles
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1)
                angles.append(angle)
            
            angles = np.array(angles)
            
            # Check for regular angles (0, 45, 90 degrees)
            regular_angles = [0, np.pi/4, np.pi/2, -np.pi/4, -np.pi/2]
            regularity_scores = []
            
            for reg_angle in regular_angles:
                distances = np.abs(angles - reg_angle)
                distances = np.minimum(distances, np.pi - distances)
                close_to_regular = np.sum(distances < 0.1)
                regularity_scores.append(close_to_regular / len(angles))
            
            return max(regularity_scores)
        
        return 0.0
    
    def _calculate_text_likelihood(self, gray: np.ndarray) -> float:
        """Calculate likelihood of text presence"""
        # Text typically has high frequency horizontal patterns
        
        # Apply horizontal Sobel
        sobel_h = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Look for horizontal line patterns
        h_proj = np.sum(np.abs(sobel_h), axis=1)
        
        # Analyze periodicity in projection
        if len(h_proj) > 10:
            # Autocorrelation
            autocorr = np.correlate(h_proj - np.mean(h_proj), 
                                   h_proj - np.mean(h_proj), mode='same')
            autocorr = autocorr / (autocorr[len(autocorr)//2] + 1e-10)
            
            # Look for periodic peaks (text lines)
            peaks = []
            for i in range(5, len(autocorr)//2, 5):
                if i < len(autocorr) - 1:
                    if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                        peaks.append(autocorr[i])
            
            if peaks:
                text_likelihood = np.mean(peaks)
            else:
                text_likelihood = 0.0
        else:
            text_likelihood = 0.0
        
        # Also check for small connected components (characters)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        if num_labels > 1:
            # Filter components by size
            areas = stats[1:, cv2.CC_STAT_AREA]
            small_components = np.sum((areas > 10) & (areas < 500))
            component_density = small_components / (gray.shape[0] * gray.shape[1])
            
            text_likelihood = max(text_likelihood, min(component_density * 100, 1.0))
        
        return text_likelihood
    
    def _calculate_gradient_smoothness(self, gray: np.ndarray) -> float:
        """Calculate gradient smoothness"""
        # Calculate gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Second derivatives for smoothness
        grad_xx = cv2.Sobel(grad_x, cv2.CV_64F, 1, 0, ksize=3)
        grad_yy = cv2.Sobel(grad_y, cv2.CV_64F, 0, 1, ksize=3)
        
        # Smoothness from second derivative magnitude
        second_deriv = np.sqrt(grad_xx**2 + grad_yy**2)
        first_deriv = np.sqrt(grad_x**2 + grad_y**2)
        
        if np.mean(first_deriv) > 0:
            smoothness = 1.0 - min(np.mean(second_deriv) / np.mean(first_deriv), 1.0)
        else:
            smoothness = 0.5
        
        return smoothness
    
    def _calculate_noise_level(self, gray: np.ndarray) -> float:
        """Estimate noise level in the image"""
        # Use Laplacian variance method
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        
        # Estimate noise from high-frequency components
        noise_estimate = np.std(laplacian) / 255.0
        
        # Also check for salt-and-pepper noise
        median_filtered = cv2.medianBlur(gray, 3)
        diff = np.abs(gray.astype(float) - median_filtered.astype(float))
        salt_pepper = np.mean(diff > 20) 
        
        return min(noise_estimate + salt_pepper, 1.0)
    
    def _calculate_contrast_distribution(self, gray: np.ndarray) -> float:
        """Calculate contrast distribution metric"""
        # Local contrast using standard deviation in windows
        window_size = 16
        h, w = gray.shape
        
        local_contrasts = []
        for i in range(0, h - window_size, window_size//2):
            for j in range(0, w - window_size, window_size//2):
                window = gray[i:i+window_size, j:j+window_size]
                local_contrast = np.std(window)
                local_contrasts.append(local_contrast)
        
        if local_contrasts:
            # Distribution uniformity
            contrast_std = np.std(local_contrasts)
            contrast_mean = np.mean(local_contrasts)
            
            if contrast_mean > 0:
                distribution = 1.0 - min(contrast_std / contrast_mean, 1.0)
            else:
                distribution = 0.0
        else:
            distribution = 0.5
        
        return distribution
    
    def _analyze_spatial_frequencies(self, gray: np.ndarray) -> Dict[str, float]:
        """Analyze spatial frequency content"""
        # Apply FFT
        f_transform = fft2(gray)
        f_shift = fftshift(f_transform)
        magnitude = np.abs(f_shift)
        
        h, w = gray.shape
        center_y, center_x = h // 2, w // 2
        
        # Define frequency bands
        max_radius = min(center_x, center_y)
        
        # Create radial masks
        y, x = np.ogrid[:h, :w]
        radius = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Low frequency (0-25% radius)
        low_mask = radius <= max_radius * 0.25
        low_energy = np.sum(magnitude[low_mask]**2)
        
        # Mid frequency (25-50% radius)
        mid_mask = (radius > max_radius * 0.25) & (radius <= max_radius * 0.5)
        mid_energy = np.sum(magnitude[mid_mask]**2)
        
        # High frequency (50-100% radius)
        high_mask = radius > max_radius * 0.5
        high_energy = np.sum(magnitude[high_mask]**2)
        
        total_energy = low_energy + mid_energy + high_energy + 1e-10
        
        return {
            'low': low_energy / total_energy,
            'mid': mid_energy / total_energy,
            'high': high_energy / total_energy
        }
    
    def _calculate_texture_homogeneity(self, gray: np.ndarray) -> float:
        """Calculate texture homogeneity"""
        # Use Local Binary Pattern-like analysis
        h, w = gray.shape
        
        # Calculate local variance
        window_size = 8
        homogeneity_scores = []
        
        for i in range(0, h - window_size, window_size):
            for j in range(0, w - window_size, window_size):
                window = gray[i:i+window_size, j:j+window_size]
                
                # Calculate variance within window
                variance = np.var(window)
                
                # Low variance indicates homogeneity
                homogeneity = 1.0 / (1.0 + variance / 100.0)
                homogeneity_scores.append(homogeneity)
        
        return np.mean(homogeneity_scores) if homogeneity_scores else 0.5
    
    def _calculate_structural_alignment(self, gray: np.ndarray) -> float:
        """Calculate structural alignment score"""
        edges = cv2.Canny(gray, 50, 150)
        
        # Detect dominant orientations
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=30, maxLineGap=10)
        
        if lines is not None and len(lines) > 10:
            # Extract angles
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1)
                angles.append(angle)
            
            # Cluster angles
            angles = np.array(angles).reshape(-1, 1)
            
            # Find dominant directions
            hist, bins = np.histogram(angles, bins=36, range=(-np.pi, np.pi))
            
            # Alignment score from concentration of angles
            if np.sum(hist) > 0:
                probs = hist / np.sum(hist)
                entropy = -np.sum(probs * np.log2(probs + 1e-10))
                max_entropy = np.log2(36)
                alignment = 1.0 - (entropy / max_entropy)
            else:
                alignment = 0.0
        else:
            alignment = 0.0
        
        return alignment
    
    def _analyze_motion_characteristics(self, 
                                       optical_flow: Optional[np.ndarray]) -> MotionCharacteristics:
        """Analyze motion characteristics"""
        if optical_flow is None or len(self.frame_buffer) < 2:
            return MotionCharacteristics(stillness_ratio=1.0)
        
        # Motion smoothness
        motion_smoothness = self._calculate_motion_smoothness(optical_flow)
        
        # Motion consistency
        motion_consistency = self._calculate_motion_consistency()
        
        # Camera vs object motion
        camera_motion, object_motion = self._separate_motion_types(optical_flow)
        
        # Motion periodicity
        motion_periodicity = self._calculate_motion_periodicity()
        
        # Stillness ratio
        stillness_ratio = self._calculate_stillness_ratio(optical_flow)
        
        # Motion complexity
        motion_complexity = self._calculate_motion_complexity(optical_flow)
        
        # Transition frequency
        transition_frequency = self._calculate_transition_frequency()
        
        # Motion patterns
        motion_patterns = self._detect_motion_patterns(optical_flow)
        
        # Temporal stability
        temporal_stability = self._calculate_temporal_stability()
        
        return MotionCharacteristics(
            motion_smoothness=motion_smoothness,
            motion_consistency=motion_consistency,
            camera_motion_score=camera_motion,
            object_motion_score=object_motion,
            motion_periodicity=motion_periodicity,
            stillness_ratio=stillness_ratio,
            motion_complexity=motion_complexity,
            transition_frequency=transition_frequency,
            motion_patterns=motion_patterns,
            temporal_stability=temporal_stability
        )
    
    def _calculate_motion_smoothness(self, flow: np.ndarray) -> float:
        """Calculate motion smoothness from optical flow"""
        # Calculate flow gradients
        flow_dx = np.gradient(flow[:, :, 0], axis=1)
        flow_dy = np.gradient(flow[:, :, 1], axis=0)
        
        # Smoothness from gradient magnitude
        gradient_mag = np.sqrt(flow_dx**2 + flow_dy**2)
        
        # Lower gradient indicates smoother motion
        flow_mag = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
        
        if np.mean(flow_mag) > 0.1:
            smoothness = 1.0 - min(np.mean(gradient_mag) / np.mean(flow_mag), 1.0)
        else:
            smoothness = 1.0  # No motion is perfectly smooth
        
        return smoothness
    
    def _calculate_motion_consistency(self) -> float:
        """Calculate motion consistency over time"""
        if len(self.frame_buffer) < 3:
            return 0.5
        
        # Compare consecutive flow fields
        consistencies = []
        
        for i in range(len(self.frame_buffer) - 2):
            if (self.frame_buffer[i]['flow'] is not None and 
                self.frame_buffer[i+1]['flow'] is not None):
                
                flow1 = self.frame_buffer[i]['flow']
                flow2 = self.frame_buffer[i+1]['flow']
                
                # Calculate similarity
                diff = np.mean(np.abs(flow1 - flow2))
                avg_mag = (np.mean(np.abs(flow1)) + np.mean(np.abs(flow2))) / 2
                
                if avg_mag > 0:
                    consistency = 1.0 - min(diff / avg_mag, 1.0)
                else:
                    consistency = 1.0
                
                consistencies.append(consistency)
        
        return np.mean(consistencies) if consistencies else 0.5
    
    def _separate_motion_types(self, flow: np.ndarray) -> Tuple[float, float]:
        """Separate camera motion from object motion"""
        h, w = flow.shape[:2]
        
        # Estimate global motion (camera)
        # Sample points
        sample_size = 100
        y_coords = np.random.randint(0, h, sample_size)
        x_coords = np.random.randint(0, w, sample_size)
        
        flow_samples = flow[y_coords, x_coords]
        
        # RANSAC-like approach to find dominant motion
        best_inliers = 0
        best_motion = np.array([0, 0])
        
        for _ in range(20):
            # Random sample
            idx = np.random.choice(sample_size, 10)
            candidate_motion = np.median(flow_samples[idx], axis=0)
            
            # Count inliers
            distances = np.linalg.norm(flow_samples - candidate_motion, axis=1)
            inliers = np.sum(distances < 2.0)
            
            if inliers > best_inliers:
                best_inliers = inliers
                best_motion = candidate_motion
        
        # Camera motion score
        camera_motion = min(np.linalg.norm(best_motion) / 10.0, 1.0)
        
        # Object motion (deviation from global)
        flow_centered = flow - best_motion
        object_motion = min(np.mean(np.abs(flow_centered)) / 5.0, 1.0)
        
        return camera_motion, object_motion
    
    def _calculate_motion_periodicity(self) -> float:
        """Calculate periodicity in motion patterns"""
        if len(self.frame_buffer) < 10:
            return 0.0
        
        # Extract motion magnitudes over time
        magnitudes = []
        
        for i in range(len(self.frame_buffer)):
            if self.frame_buffer[i]['flow'] is not None:
                flow = self.frame_buffer[i]['flow']
                mag = np.mean(np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2))
                magnitudes.append(mag)
        
        if len(magnitudes) > 8:
            magnitudes = np.array(magnitudes)
            
            # Autocorrelation for periodicity
            autocorr = np.correlate(magnitudes - np.mean(magnitudes),
                                   magnitudes - np.mean(magnitudes), mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr / (autocorr[0] + 1e-10)
            
            # Find peaks
            peaks = []
            for i in range(2, min(len(autocorr) - 1, 20)):
                if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1] and autocorr[i] > 0.3:
                    peaks.append(i)
            
            if peaks:
                # Regular peaks indicate periodicity
                if len(peaks) > 1:
                    intervals = np.diff(peaks)
                    periodicity = 1.0 - (np.std(intervals) / (np.mean(intervals) + 1e-10))
                else:
                    periodicity = 0.5
            else:
                periodicity = 0.0
        else:
            periodicity = 0.0
        
        return max(0, min(periodicity, 1.0))
    
    def _calculate_stillness_ratio(self, flow: np.ndarray) -> float:
        """Calculate ratio of still pixels"""
        magnitude = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
        still_pixels = magnitude < 0.5
        return np.mean(still_pixels)
    
    def _calculate_motion_complexity(self, flow: np.ndarray) -> float:
        """Calculate complexity of motion field"""
        # Analyze flow field structure
        mag = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
        angle = np.arctan2(flow[:, :, 1], flow[:, :, 0])
        
        # Complexity from magnitude variance
        mag_complexity = np.std(mag) / (np.mean(mag) + 1e-10)
        
        # Complexity from angle diversity
        angle_hist, _ = np.histogram(angle, bins=36, range=(-np.pi, np.pi))
        angle_probs = angle_hist / (angle_hist.sum() + 1e-10)
        angle_entropy = -np.sum(angle_probs * np.log2(angle_probs + 1e-10))
        angle_complexity = angle_entropy / np.log2(36)
        
        return (mag_complexity + angle_complexity) / 2
    
    def _calculate_transition_frequency(self) -> float:
        """Calculate frequency of visual transitions"""
        if len(self.frame_buffer) < 3:
            return 0.0
        
        transitions = []
        
        for i in range(len(self.frame_buffer) - 1):
            frame1 = self.frame_buffer[i]['frame']
            frame2 = self.frame_buffer[i + 1]['frame']
            
            # Calculate change magnitude
            diff = np.mean(np.abs(frame1.astype(float) - frame2.astype(float)))
            
            # Threshold for transition
            transitions.append(diff > 30)
        
        return np.mean(transitions) if transitions else 0.0
    
    def _detect_motion_patterns(self, flow: np.ndarray) -> List[str]:
        """Detect specific motion patterns"""
        patterns = []
        
        h, w = flow.shape[:2]
        center_y, center_x = h // 2, w // 2
        
        # Create coordinate grids
        y, x = np.ogrid[:h, :w]
        
        # Radial pattern (zoom/expansion)
        radial_vec = np.stack([x - center_x, y - center_y], axis=-1)
        radial_vec_norm = radial_vec / (np.linalg.norm(radial_vec, axis=-1, keepdims=True) + 1e-10)
        
        flow_norm = flow / (np.linalg.norm(flow, axis=-1, keepdims=True) + 1e-10)
        radial_similarity = np.mean(np.sum(flow_norm * radial_vec_norm, axis=-1))
        
        if abs(radial_similarity) > 0.3:
            patterns.append("radial" if radial_similarity > 0 else "convergent")
        
        # Rotational pattern
        tangent_vec = np.stack([-(y - center_y), x - center_x], axis=-1)
        tangent_vec_norm = tangent_vec / (np.linalg.norm(tangent_vec, axis=-1, keepdims=True) + 1e-10)
        
        rotation_similarity = np.mean(np.sum(flow_norm * tangent_vec_norm, axis=-1))
        
        if abs(rotation_similarity) > 0.3:
            patterns.append("rotational")
        
        # Linear pattern (translation)
        mean_flow = np.mean(flow, axis=(0, 1))
        flow_centered = flow - mean_flow
        
        if np.mean(np.linalg.norm(flow_centered, axis=-1)) < 1.0:
            if abs(mean_flow[0]) > abs(mean_flow[1]):
                patterns.append("horizontal")
            else:
                patterns.append("vertical")
        
        # Turbulent pattern
        flow_grad_x = np.gradient(flow[:, :, 0])
        flow_grad_y = np.gradient(flow[:, :, 1])
        turbulence = np.std(flow_grad_x) + np.std(flow_grad_y)
        
        if turbulence > 2.0:
            patterns.append("turbulent")
        
        return patterns
    
    def _calculate_temporal_stability(self) -> float:
        """Calculate temporal stability of visual content"""
        if len(self.frame_buffer) < 3:
            return 0.5
        
        # Calculate frame-to-frame similarities
        similarities = []
        
        for i in range(len(self.frame_buffer) - 1):
            frame1 = self.frame_buffer[i]['frame']
            frame2 = self.frame_buffer[i + 1]['frame']
            
            # Structural similarity
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            
            # Simple SSIM-like metric
            mean1, mean2 = np.mean(gray1), np.mean(gray2)
            var1, var2 = np.var(gray1), np.var(gray2)
            cov = np.mean((gray1 - mean1) * (gray2 - mean2))
            
            c1, c2 = 0.01**2, 0.03**2
            ssim = ((2 * mean1 * mean2 + c1) * (2 * cov + c2)) / \
                   ((mean1**2 + mean2**2 + c1) * (var1 + var2 + c2))
            
            similarities.append(ssim)
        
        return np.mean(similarities) if similarities else 0.5
    
    def _analyze_layout_characteristics(self, frame: np.ndarray) -> LayoutCharacteristics:
        """Analyze layout and composition characteristics"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Grid alignment
        grid_alignment = self._calculate_grid_alignment(gray)
        
        # Rectangular regions
        rectangular_regions = self._detect_rectangular_regions(gray)
        
        # Symmetry
        symmetry_score = self._calculate_symmetry(gray)
        
        # Layout consistency
        layout_consistency = self._calculate_layout_consistency()
        
        # White space ratio
        white_space_ratio = self._calculate_white_space_ratio(frame)
        
        # Element density
        element_density = self._calculate_element_density(gray)
        
        # Hierarchical structure
        hierarchical_structure = self._calculate_hierarchical_structure(gray)
        
        # Alignment score
        alignment_score = self._calculate_alignment_score(gray)
        
        # Repetitive elements
        repetitive_elements = self._detect_repetitive_elements(gray)
        
        # Layout complexity
        layout_complexity = self._calculate_layout_complexity(gray)
        
        return LayoutCharacteristics(
            grid_alignment=grid_alignment,
            rectangular_regions=rectangular_regions,
            symmetry_score=symmetry_score,
            layout_consistency=layout_consistency,
            white_space_ratio=white_space_ratio,
            element_density=element_density,
            hierarchical_structure=hierarchical_structure,
            alignment_score=alignment_score,
            repetitive_elements=repetitive_elements,
            layout_complexity=layout_complexity
        )
    
    def _calculate_grid_alignment(self, gray: np.ndarray) -> float:
        """Calculate grid alignment score"""
        edges = cv2.Canny(gray, 50, 150)
        
        # Project edges to axes
        h_projection = np.sum(edges, axis=1)
        v_projection = np.sum(edges, axis=0)
        
        # Find peaks in projections
        h_peaks = signal.find_peaks(h_projection, height=np.mean(h_projection))[0]
        v_peaks = signal.find_peaks(v_projection, height=np.mean(v_projection))[0]
        
        # Check for regular spacing
        grid_score = 0.0
        
        if len(h_peaks) > 2:
            h_spacing = np.diff(h_peaks)
            h_regularity = 1.0 - (np.std(h_spacing) / (np.mean(h_spacing) + 1e-10))
            grid_score += h_regularity * 0.5
        
        if len(v_peaks) > 2:
            v_spacing = np.diff(v_peaks)
            v_regularity = 1.0 - (np.std(v_spacing) / (np.mean(v_spacing) + 1e-10))
            grid_score += v_regularity * 0.5
        
        return max(0, min(grid_score, 1.0))
    
    def _detect_rectangular_regions(self, gray: np.ndarray) -> int:
        """Detect number of rectangular regions"""
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        rectangular_count = 0
        
        for contour in contours:
            if cv2.contourArea(contour) > 100:
                # Approximate polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Check if rectangular (4 vertices)
                if len(approx) == 4:
                    # Check angles
                    angles = []
                    for i in range(4):
                        p1 = approx[i][0]
                        p2 = approx[(i + 1) % 4][0]
                        p3 = approx[(i + 2) % 4][0]
                        
                        v1 = p1 - p2
                        v2 = p3 - p2
                        
                        angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10))
                        angles.append(angle)
                    
                    # Check if angles are close to 90 degrees
                    if all(abs(angle - np.pi/2) < 0.3 for angle in angles):
                        rectangular_count += 1
        
        return rectangular_count
    
    def _calculate_symmetry(self, gray: np.ndarray) -> float:
        """Calculate symmetry score"""
        h, w = gray.shape
        
        # Vertical symmetry
        left = gray[:, :w//2]
        right = gray[:, w//2:w//2*2]
        right_flipped = cv2.flip(right, 1)
        
        if left.shape == right_flipped.shape:
            v_symmetry = 1.0 - np.mean(np.abs(left - right_flipped)) / 255.0
        else:
            v_symmetry = 0.0
        
        # Horizontal symmetry
        top = gray[:h//2, :]
        bottom = gray[h//2:h//2*2, :]
        bottom_flipped = cv2.flip(bottom, 0)
        
        if top.shape == bottom_flipped.shape:
            h_symmetry = 1.0 - np.mean(np.abs(top - bottom_flipped)) / 255.0
        else:
            h_symmetry = 0.0
        
        return max(v_symmetry, h_symmetry)
    
    def _calculate_layout_consistency(self) -> float:
        """Calculate layout consistency over time"""
        if len(self.frame_buffer) < 3:
            return 0.5
        
        # Extract structural features from frames
        structures = []
        
        for i in range(min(5, len(self.frame_buffer))):
            frame = self.frame_buffer[-(i+1)]['frame']
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Simple structural signature
            edges = cv2.Canny(gray, 50, 150)
            h_proj = np.sum(edges, axis=1)
            v_proj = np.sum(edges, axis=0)
            
            # Normalize projections
            h_proj = h_proj / (np.sum(h_proj) + 1e-10)
            v_proj = v_proj / (np.sum(v_proj) + 1e-10)
            
            structures.append((h_proj, v_proj))
        
        # Compare structures
        if len(structures) > 1:
            similarities = []
            
            for i in range(len(structures) - 1):
                h_sim = 1.0 - np.mean(np.abs(structures[i][0] - structures[i+1][0]))
                v_sim = 1.0 - np.mean(np.abs(structures[i][1] - structures[i+1][1]))
                similarities.append((h_sim + v_sim) / 2)
            
            consistency = np.mean(similarities)
        else:
            consistency = 0.5
        
        return consistency
    
    def _calculate_white_space_ratio(self, frame: np.ndarray) -> float:
        """Calculate ratio of white/empty space"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect "empty" regions (low variance, near white/black)
        window_size = 16
        h, w = gray.shape
        
        empty_count = 0
        total_count = 0
        
        for i in range(0, h - window_size, window_size):
            for j in range(0, w - window_size, window_size):
                window = gray[i:i+window_size, j:j+window_size]
                
                # Check if uniform (low variance)
                if np.var(window) < 100:
                    # Check if near white or black
                    mean_val = np.mean(window)
                    if mean_val > 200 or mean_val < 50:
                        empty_count += 1
                
                total_count += 1
        
        return empty_count / total_count if total_count > 0 else 0.0
    
    def _calculate_element_density(self, gray: np.ndarray) -> float:
        """Calculate density of visual elements"""
        # Use edge detection and connected components
        edges = cv2.Canny(gray, 50, 150)
        
        # Morphological closing to connect nearby edges
        kernel = np.ones((3, 3), np.uint8)
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
        
        # Filter by size
        significant_elements = 0
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > 50:  # Minimum size threshold
                significant_elements += 1
        
        # Density relative to image size
        density = significant_elements / (gray.shape[0] * gray.shape[1] / 10000)
        
        return min(density, 1.0)
    
    def _calculate_hierarchical_structure(self, gray: np.ndarray) -> float:
        """Calculate hierarchical structure score"""
        # Detect elements at different scales
        scales = [1, 2, 4, 8]
        element_counts = []
        
        for scale in scales:
            # Resize image
            scaled = cv2.resize(gray, (gray.shape[1]//scale, gray.shape[0]//scale))
            
            # Detect edges
            edges = cv2.Canny(scaled, 50, 150)
            
            # Count connected components
            num_labels, _ = cv2.connectedComponents(edges)
            element_counts.append(num_labels - 1)
        
        # Hierarchical structure from scale distribution
        if len(element_counts) > 1 and element_counts[0] > 0:
            # Good hierarchy has fewer elements at coarser scales
            hierarchy_ratios = []
            for i in range(len(element_counts) - 1):
                if element_counts[i] > 0:
                    ratio = element_counts[i+1] / element_counts[i]
                    hierarchy_ratios.append(min(ratio, 1.0))
            
            if hierarchy_ratios:
                hierarchy_score = 1.0 - np.mean(hierarchy_ratios)
            else:
                hierarchy_score = 0.0
        else:
            hierarchy_score = 0.0
        
        return hierarchy_score
    
    def _calculate_alignment_score(self, gray: np.ndarray) -> float:
        """Calculate element alignment score"""
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) < 2:
            return 0.0
        
        # Get bounding boxes
        bboxes = []
        for contour in contours:
            if cv2.contourArea(contour) > 50:
                x, y, w, h = cv2.boundingRect(contour)
                bboxes.append((x, y, x+w, y+h))
        
        if len(bboxes) < 2:
            return 0.0
        
        # Check alignment
        alignment_scores = []
        
        # Check horizontal alignment
        y_positions = [(bbox[1] + bbox[3]) / 2 for bbox in bboxes]
        y_unique = list(set(y_positions))
        
        for y in y_unique:
            aligned = sum(1 for pos in y_positions if abs(pos - y) < 5)
            if aligned > 1:
                alignment_scores.append(aligned / len(bboxes))
        
        # Check vertical alignment
        x_positions = [(bbox[0] + bbox[2]) / 2 for bbox in bboxes]
        x_unique = list(set(x_positions))
        
        for x in x_unique:
            aligned = sum(1 for pos in x_positions if abs(pos - x) < 5)
            if aligned > 1:
                alignment_scores.append(aligned / len(bboxes))
        
        return max(alignment_scores) if alignment_scores else 0.0
    
    def _detect_repetitive_elements(self, gray: np.ndarray) -> int:
        """Detect number of repetitive elements"""
        # Use template matching approach
        edges = cv2.Canny(gray, 50, 150)
        
        # Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(edges, connectivity=8)
        
        # Extract component templates
        templates = []
        for i in range(1, min(num_labels, 20)):  # Limit to 20 components
            mask = (labels == i).astype(np.uint8)
            x, y, w, h = cv2.boundingRect(mask)
            
            if w > 10 and h > 10 and w < 100 and h < 100:
                template = mask[y:y+h, x:x+w]
                templates.append(template)
        
        # Find similar templates
        repetitive_count = 0
        matched = set()
        
        for i, template1 in enumerate(templates):
            if i in matched:
                continue
            
            similar_count = 1
            for j, template2 in enumerate(templates[i+1:], i+1):
                if j in matched:
                    continue
                
                # Resize to same size for comparison
                if template1.shape != template2.shape:
                    template2_resized = cv2.resize(template2, template1.shape[::-1])
                else:
                    template2_resized = template2
                
                # Calculate similarity
                similarity = 1.0 - np.mean(np.abs(template1 - template2_resized))
                
                if similarity > 0.7:
                    similar_count += 1
                    matched.add(j)
            
            if similar_count > 1:
                repetitive_count += similar_count
        
        return repetitive_count
    
    def _calculate_layout_complexity(self, gray: np.ndarray) -> float:
        """Calculate overall layout complexity"""
        # Combine multiple factors
        edges = cv2.Canny(gray, 50, 150)
        
        # Edge density
        edge_density = np.sum(edges > 0) / edges.size
        
        # Number of regions
        num_labels, _ = cv2.connectedComponents(edges)
        region_complexity = min(num_labels / 100.0, 1.0)
        
        # Orientation diversity
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=30, maxLineGap=10)
        
        if lines is not None and len(lines) > 5:
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1)
                angles.append(angle)
            
            # Angle diversity
            angle_hist, _ = np.histogram(angles, bins=18, range=(-np.pi, np.pi))
            angle_probs = angle_hist / (angle_hist.sum() + 1e-10)
            angle_entropy = -np.sum(angle_probs * np.log2(angle_probs + 1e-10))
            orientation_complexity = angle_entropy / np.log2(18)
        else:
            orientation_complexity = 0.0
        
        # Combine factors
        layout_complexity = (edge_density + region_complexity + orientation_complexity) / 3
        
        return layout_complexity
    
    def _analyze_color_characteristics(self, frame: np.ndarray) -> ColorCharacteristics:
        """Analyze color characteristics"""
        # Convert to different color spaces
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        
        # Palette type
        palette_type = self._determine_palette_type(frame)
        
        # Color consistency
        color_consistency = self._calculate_color_consistency(frame)
        
        # Saturation profile
        saturation_profile = self._determine_saturation_profile(hsv)
        
        # Luminance distribution
        luminance_distribution = self._determine_luminance_distribution(lab)
        
        # Color temperature
        color_temperature = self._determine_color_temperature(frame)
        
        # Gradient presence
        gradient_presence = self._calculate_gradient_presence(frame)
        
        # Flat color ratio
        flat_color_ratio = self._calculate_flat_color_ratio(frame)
        
        # Color transitions
        color_transitions = self._determine_color_transitions(frame)
        
        # Transparency likelihood
        transparency_likelihood = self._calculate_transparency_likelihood(frame)
        
        # Color depth estimate
        color_depth_estimate = self._estimate_color_depth(frame)
        
        return ColorCharacteristics(
            palette_type=palette_type,
            color_consistency=color_consistency,
            saturation_profile=saturation_profile,
            luminance_distribution=luminance_distribution,
            color_temperature=color_temperature,
            gradient_presence=gradient_presence,
            flat_color_ratio=flat_color_ratio,
            color_transitions=color_transitions,
            transparency_likelihood=transparency_likelihood,
            color_depth_estimate=color_depth_estimate
        )
    
    def _determine_palette_type(self, frame: np.ndarray) -> str:
        """Determine the type of color palette"""
        # Sample colors
        pixels = frame.reshape(-1, 3)
        sample_size = min(1000, len(pixels))
        sample = pixels[np.random.choice(len(pixels), sample_size, replace=False)]
        
        # Cluster colors
        try:
            kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
            kmeans.fit(sample)
            palette = kmeans.cluster_centers_
        except:
            return "unknown"
        
        # Analyze palette
        # Check if monochrome
        color_std = np.std(palette, axis=1)
        if np.mean(color_std) < 20:
            return "monochrome"
        
        # Check saturation
        hsv_palette = cv2.cvtColor(palette.reshape(1, -1, 3).astype(np.uint8), 
                                  cv2.COLOR_BGR2HSV).reshape(-1, 3)
        avg_saturation = np.mean(hsv_palette[:, 1])
        
        if avg_saturation < 50:
            return "limited"
        elif avg_saturation < 150:
            return "natural"
        else:
            return "vibrant"
    
    def _calculate_color_consistency(self, frame: np.ndarray) -> float:
        """Calculate color consistency across the frame"""
        # Divide into blocks and compare color distributions
        h, w = frame.shape[:2]
        block_size = 32
        
        block_hists = []
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block = frame[i:i+block_size, j:j+block_size]
                
                # Calculate color histogram
                hist = cv2.calcHist([block], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                hist = hist.flatten()
                hist = hist / (hist.sum() + 1e-10)
                block_hists.append(hist)
        
        if len(block_hists) > 1:
            # Calculate pairwise similarities
            similarities = []
            for i in range(len(block_hists) - 1):
                similarity = 1.0 - distance.cosine(block_hists[i], block_hists[i+1])
                similarities.append(similarity)
            
            consistency = np.mean(similarities)
        else:
            consistency = 0.5
        
        return consistency
    
    def _determine_saturation_profile(self, hsv: np.ndarray) -> str:
        """Determine saturation profile"""
        saturation = hsv[:, :, 1].flatten()
        mean_sat = np.mean(saturation)
        
        if mean_sat < 50:
            return "low"
        elif mean_sat < 100:
            return "medium"
        elif mean_sat < 180:
            return "high"
        else:
            # Check distribution
            std_sat = np.std(saturation)
            if std_sat > 50:
                return "mixed"
            else:
                return "high"
    
    def _determine_luminance_distribution(self, lab: np.ndarray) -> str:
        """Determine luminance distribution type"""
        luminance = lab[:, :, 0].flatten()
        
        # Check distribution shape
        hist, bins = np.histogram(luminance, bins=50)
        hist = hist / hist.sum()
        
        # Check for bimodal distribution
        peaks = signal.find_peaks(hist, height=0.02)[0]
        
        if len(peaks) >= 2:
            return "bimodal"
        
        # Check skewness
        skewness = stats.skew(luminance)
        
        if abs(skewness) < 0.5:
            # Check uniformity
            uniformity = np.std(hist)
            if uniformity < 0.01:
                return "uniform"
            else:
                return "normal"
        else:
            return "skewed"
    
    def _determine_color_temperature(self, frame: np.ndarray) -> str:
        """Determine color temperature"""
        # Simple color temperature estimation
        b, g, r = cv2.split(frame)
        
        avg_r = np.mean(r)
        avg_b = np.mean(b)
        
        if avg_r > avg_b * 1.2:
            return "warm"
        elif avg_b > avg_r * 1.2:
            return "cool"
        else:
            # Check variance
            temp_variance = np.var(r.flatten() - b.flatten())
            if temp_variance > 1000:
                return "mixed"
            else:
                return "neutral"
    
    def _calculate_gradient_presence(self, frame: np.ndarray) -> float:
        """Calculate presence of color gradients"""
        # Check for smooth color transitions
        h, w = frame.shape[:2]
        
        # Sample horizontal and vertical strips
        gradient_scores = []
        
        # Horizontal gradients
        for i in range(0, h, h//10):
            strip = frame[i:i+5, :]
            if strip.shape[0] > 0:
                # Check color progression
                colors = [np.mean(strip[:, j:j+10], axis=(0, 1)) 
                         for j in range(0, w-10, 10)]
                
                if len(colors) > 2:
                    # Calculate smoothness of progression
                    diffs = [np.linalg.norm(colors[i+1] - colors[i]) 
                            for i in range(len(colors)-1)]
                    
                    if diffs:
                        smoothness = 1.0 - (np.std(diffs) / (np.mean(diffs) + 1e-10))
                        gradient_scores.append(smoothness)
        
        # Vertical gradients
        for j in range(0, w, w//10):
            strip = frame[:, j:j+5]
            if strip.shape[1] > 0:
                colors = [np.mean(strip[i:i+10, :], axis=(0, 1)) 
                         for i in range(0, h-10, 10)]
                
                if len(colors) > 2:
                    diffs = [np.linalg.norm(colors[i+1] - colors[i]) 
                            for i in range(len(colors)-1)]
                    
                    if diffs:
                        smoothness = 1.0 - (np.std(diffs) / (np.mean(diffs) + 1e-10))
                        gradient_scores.append(smoothness)
        
        return np.mean(gradient_scores) if gradient_scores else 0.0
    
    def _calculate_flat_color_ratio(self, frame: np.ndarray) -> float:
        """Calculate ratio of flat color regions"""
        # Detect uniform color regions
        h, w = frame.shape[:2]
        window_size = 8
        
        flat_count = 0
        total_count = 0
        
        for i in range(0, h - window_size, window_size):
            for j in range(0, w - window_size, window_size):
                window = frame[i:i+window_size, j:j+window_size]
                
                # Check color variance
                color_std = np.mean(np.std(window, axis=(0, 1)))
                
                if color_std < 5:  # Very low variance
                    flat_count += 1
                
                total_count += 1
        
        return flat_count / total_count if total_count > 0 else 0.0
    
    def _determine_color_transitions(self, frame: np.ndarray) -> str:
        """Determine type of color transitions"""
        # Analyze edge characteristics
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate color gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Analyze gradient distribution
        strong_edges = gradient_mag > np.percentile(gradient_mag, 90)
        weak_edges = (gradient_mag > np.percentile(gradient_mag, 50)) & ~strong_edges
        
        strong_ratio = np.sum(strong_edges) / gradient_mag.size
        weak_ratio = np.sum(weak_edges) / gradient_mag.size
        
        if strong_ratio > weak_ratio * 2:
            return "sharp"
        elif weak_ratio > strong_ratio * 2:
            return "smooth"
        else:
            return "mixed"
    
    def _calculate_transparency_likelihood(self, frame: np.ndarray) -> float:
        """Calculate likelihood of transparency effects"""
        # Look for alpha blending patterns
        h, w = frame.shape[:2]
        
        transparency_indicators = []
        
        # Check for semi-transparent overlays (consistent opacity patterns)
        for i in range(0, h - 50, 50):
            for j in range(0, w - 50, 50):
                region = frame[i:i+50, j:j+50]
                
                # Check if region has consistent transparency pattern
                # (e.g., consistent darkening/lightening)
                mean_color = np.mean(region, axis=(0, 1))
                deviations = region - mean_color
                
                # Transparency often creates uniform deviations
                deviation_consistency = 1.0 - (np.std(deviations) / (np.mean(np.abs(deviations)) + 1e-10))
                transparency_indicators.append(deviation_consistency)
        
        # Check for gradient transparency (fading edges)
        edges = cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 50, 150)
        edge_blur = cv2.GaussianBlur(edges, (15, 15), 0)
        
        # Soft edges indicate transparency
        soft_edge_ratio = np.sum((edge_blur > 0) & (edge_blur < 128)) / edge_blur.size
        
        transparency_score = np.mean(transparency_indicators) * 0.7 + soft_edge_ratio * 0.3
        
        return min(transparency_score, 1.0)
    
    def _estimate_color_depth(self, frame: np.ndarray) -> int:
        """Estimate effective color depth"""
        # Count unique colors in quantized image
        quantization_levels = [256, 128, 64, 32, 16, 8, 4, 2]
        
        for level in quantization_levels:
            quantized = (frame // (256 // level)) * (256 // level)
            unique_colors = len(np.unique(quantized.reshape(-1, 3), axis=0))
            
            # Estimate bits needed
            if unique_colors < 256:
                return 8
            elif unique_colors < 4096:
                return 12
            elif unique_colors < 65536:
                return 16
        
        return 24  # True color
    
    def _detect_content_patterns(self, frame: np.ndarray,
                                visual_chars: VisualCharacteristics,
                                layout_chars: LayoutCharacteristics) -> List[ContentPattern]:
        """Detect specific content patterns"""
        patterns = []
        
        # Terminal/console pattern
        if visual_chars.text_likelihood > 0.7 and layout_chars.grid_alignment > 0.6:
            patterns.append(ContentPattern(
                pattern_type="terminal_pattern",
                confidence=min(visual_chars.text_likelihood, layout_chars.grid_alignment),
                characteristics={
                    'text_density': visual_chars.text_likelihood,
                    'grid_structure': layout_chars.grid_alignment
                }
            ))
        
        # Document pattern
        if visual_chars.text_likelihood > 0.5 and layout_chars.white_space_ratio > 0.3:
            patterns.append(ContentPattern(
                pattern_type="document_pattern",
                confidence=(visual_chars.text_likelihood + layout_chars.white_space_ratio) / 2,
                characteristics={
                    'text_presence': visual_chars.text_likelihood,
                    'whitespace': layout_chars.white_space_ratio
                }
            ))
        
        # UI pattern
        if layout_chars.rectangular_regions > 3 and visual_chars.geometric_regularity > 0.5:
            patterns.append(ContentPattern(
                pattern_type="ui_pattern",
                confidence=min(layout_chars.rectangular_regions / 10.0, 1.0) * visual_chars.geometric_regularity,
                characteristics={
                    'rectangles': layout_chars.rectangular_regions,
                    'geometry': visual_chars.geometric_regularity
                }
            ))
        
        # Video/media pattern
        if visual_chars.gradient_smoothness > 0.6 and visual_chars.noise_level < 0.3:
            patterns.append(ContentPattern(
                pattern_type="media_pattern",
                confidence=visual_chars.gradient_smoothness * (1.0 - visual_chars.noise_level),
                characteristics={
                    'smoothness': visual_chars.gradient_smoothness,
                    'quality': 1.0 - visual_chars.noise_level
                }
            ))
        
        # Presentation pattern
        if layout_chars.hierarchical_structure > 0.5 and visual_chars.text_likelihood > 0.3:
            patterns.append(ContentPattern(
                pattern_type="presentation_pattern",
                confidence=(layout_chars.hierarchical_structure + visual_chars.text_likelihood) / 2,
                characteristics={
                    'hierarchy': layout_chars.hierarchical_structure,
                    'text_content': visual_chars.text_likelihood
                }
            ))
        
        # Game/interactive pattern
        if layout_chars.element_density > 0.4 and visual_chars.edge_sharpness > 0.5:
            patterns.append(ContentPattern(
                pattern_type="game_pattern",
                confidence=(layout_chars.element_density + visual_chars.edge_sharpness) / 2,
                characteristics={
                    'density': layout_chars.element_density,
                    'sharpness': visual_chars.edge_sharpness
                }
            ))
        
        return patterns
    
    def _infer_content_type(self,
                           visual: VisualCharacteristics,
                           motion: MotionCharacteristics,
                           layout: LayoutCharacteristics,
                           color: ColorCharacteristics,
                           patterns: List[ContentPattern]) -> Dict[str, float]:
        """Infer content type from characteristics"""
        scores = {}
        
        # Check each content type signature
        for content_type, signature in self.content_signatures.items():
            score = 0.0
            count = 0
            
            # Check visual characteristics
            for attr, (min_val, max_val) in signature.items():
                if hasattr(visual, attr):
                    value = getattr(visual, attr)
                    if min_val <= value <= max_val:
                        score += 1.0
                    else:
                        score += max(0, 1.0 - abs(value - (min_val + max_val) / 2))
                    count += 1
                elif hasattr(motion, attr):
                    value = getattr(motion, attr)
                    if min_val <= value <= max_val:
                        score += 1.0
                    else:
                        score += max(0, 1.0 - abs(value - (min_val + max_val) / 2))
                    count += 1
                elif hasattr(layout, attr):
                    value = getattr(layout, attr)
                    if min_val <= value <= max_val:
                        score += 1.0
                    else:
                        score += max(0, 1.0 - abs(value - (min_val + max_val) / 2))
                    count += 1
                elif hasattr(color, attr):
                    value = getattr(color, attr)
                    if isinstance(value, str):
                        if value == signature.get(attr):
                            score += 1.0
                        count += 1
                    elif min_val <= value <= max_val:
                        score += 1.0
                    else:
                        score += max(0, 1.0 - abs(value - (min_val + max_val) / 2))
                    count += 1
            
            if count > 0:
                scores[content_type] = score / count
        
        # Boost scores based on detected patterns
        for pattern in patterns:
            if pattern.pattern_type == "terminal_pattern":
                scores['terminal'] = scores.get('terminal', 0.0) + pattern.confidence * 0.3
            elif pattern.pattern_type == "document_pattern":
                scores['document'] = scores.get('document', 0.0) + pattern.confidence * 0.3
            elif pattern.pattern_type == "ui_pattern":
                scores['ui_interface'] = scores.get('ui_interface', 0.0) + pattern.confidence * 0.3
            elif pattern.pattern_type == "media_pattern":
                scores['video_content'] = scores.get('video_content', 0.0) + pattern.confidence * 0.3
            elif pattern.pattern_type == "presentation_pattern":
                scores['presentation'] = scores.get('presentation', 0.0) + pattern.confidence * 0.3
            elif pattern.pattern_type == "game_pattern":
                scores['gaming'] = scores.get('gaming', 0.0) + pattern.confidence * 0.3
        
        # Normalize scores
        max_score = max(scores.values()) if scores else 1.0
        if max_score > 0:
            scores = {k: min(v / max_score, 1.0) for k, v in scores.items()}
        
        return scores
    
    def _determine_types(self, scores: Dict[str, float]) -> Tuple[str, List[str]]:
        """Determine primary and secondary content types"""
        if not scores:
            return "unknown", []
        
        # Sort by score
        sorted_types = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Primary type
        primary_type = sorted_types[0][0] if sorted_types[0][1] >= self.min_confidence else "unknown"
        
        # Secondary types (above threshold)
        secondary_types = [t for t, s in sorted_types[1:] if s >= self.min_confidence * 0.7]
        
        return primary_type, secondary_types
    
    def _analyze_content_dynamics(self, motion: MotionCharacteristics) -> str:
        """Analyze content dynamics"""
        if motion.stillness_ratio > 0.95:
            return "static"
        elif motion.stillness_ratio > 0.7:
            return "semi-static"
        elif motion.motion_complexity < 0.5:
            return "dynamic"
        else:
            return "highly-dynamic"
    
    def _analyze_information_density(self,
                                    visual: VisualCharacteristics,
                                    layout: LayoutCharacteristics) -> str:
        """Analyze information density"""
        density_score = (visual.edge_density * 0.3 +
                        layout.element_density * 0.3 +
                        visual.texture_homogeneity * 0.2 +
                        (1.0 - layout.white_space_ratio) * 0.2)
        
        if density_score < 0.3:
            return "sparse"
        elif density_score < 0.5:
            return "moderate"
        elif density_score < 0.7:
            return "dense"
        else:
            return "very-dense"
    
    def _determine_visual_style(self,
                               visual: VisualCharacteristics,
                               color: ColorCharacteristics,
                               layout: LayoutCharacteristics) -> str:
        """Determine visual style"""
        complexity_score = (visual.edge_density * 0.25 +
                          (1.0 - color.flat_color_ratio) * 0.25 +
                          layout.layout_complexity * 0.25 +
                          visual.texture_homogeneity * 0.25)
        
        if complexity_score < 0.3:
            return "minimalist"
        elif complexity_score < 0.5:
            return "detailed"
        elif complexity_score < 0.7:
            return "complex"
        else:
            return "chaotic"
    
    def _estimate_interaction_likelihood(self,
                                        motion: MotionCharacteristics,
                                        layout: LayoutCharacteristics,
                                        patterns: List[ContentPattern]) -> float:
        """Estimate likelihood of interactive content"""
        interaction_score = 0.0
        
        # Motion patterns suggesting interaction
        if "oscillating" in motion.motion_patterns:
            interaction_score += 0.2
        
        # UI patterns
        ui_patterns = [p for p in patterns if p.pattern_type in ["ui_pattern", "game_pattern"]]
        if ui_patterns:
            interaction_score += max(p.confidence for p in ui_patterns) * 0.4
        
        # Layout suggesting interaction
        if layout.rectangular_regions > 5:
            interaction_score += 0.2
        
        # Repetitive elements (buttons, menus)
        if layout.repetitive_elements > 3:
            interaction_score += 0.2
        
        return min(interaction_score, 1.0)
    
    def _estimate_automation_likelihood(self,
                                       motion: MotionCharacteristics,
                                       patterns: List[ContentPattern]) -> float:
        """Estimate likelihood of automated content"""
        automation_score = 0.0
        
        # Periodic motion
        automation_score += motion.motion_periodicity * 0.3
        
        # Consistent motion
        automation_score += motion.motion_consistency * 0.3
        
        # Terminal pattern (scripts running)
        terminal_patterns = [p for p in patterns if p.pattern_type == "terminal_pattern"]
        if terminal_patterns:
            automation_score += max(p.confidence for p in terminal_patterns) * 0.4
        
        return min(automation_score, 1.0)