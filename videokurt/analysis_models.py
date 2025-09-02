"""Analysis model classes for VideoKurt.

Each analysis is a configurable class that processes video frames
and returns a RawAnalysis result.
"""

import time
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import cv2

from .models import RawAnalysis


class BaseAnalysis(ABC):
    """Base class for all video analysis methods."""
    
    METHOD_NAME = None  # Must be overridden by subclasses
    
    def __init__(self, downsample: float = 1.0, **kwargs):
        """
        Args:
            downsample: Resolution scale (0.5 = half resolution)
            **kwargs: Analysis-specific parameters
        """
        if self.METHOD_NAME is None:
            raise NotImplementedError("METHOD_NAME must be defined")
        
        self.downsample = downsample
        self.config = kwargs
        
    def preprocess_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Apply generic preprocessing like downsampling."""
        if self.downsample < 1.0:
            processed = []
            for frame in frames:
                h, w = frame.shape[:2]
                new_h = int(h * self.downsample)
                new_w = int(w * self.downsample)
                resized = cv2.resize(frame, (new_w, new_h))
                processed.append(resized)
            return processed
        return frames
    
    @abstractmethod
    def analyze(self, frames: List[np.ndarray]) -> RawAnalysis:
        """Analyze video frames and return results.
        
        Args:
            frames: List of video frames as numpy arrays
            
        Returns:
            RawAnalysis object containing the analysis results
        """
        pass


# =============================================================================
# LEVEL 1: Basic Analyses (Real-time capable)
# =============================================================================

class FrameDiff(BaseAnalysis):
    """Simple frame differencing analysis."""
    
    METHOD_NAME = 'frame_diff'
    
    def __init__(self, downsample: float = 1.0, threshold: float = 0.1):
        """
        Args:
            downsample: Resolution scale (0.5 = half resolution)
            threshold: Threshold for detecting changes
        """
        super().__init__(downsample=downsample)
        self.threshold = threshold
    
    def analyze(self, frames: List[np.ndarray]) -> RawAnalysis:
        """Compute pixel-wise differences between consecutive frames."""
        start_time = time.time()
        frames = self.preprocess_frames(frames)
        
        pixel_diffs = []
        for i in range(len(frames) - 1):
            # Convert to grayscale if needed
            if len(frames[i].shape) == 3:
                gray1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(frames[i+1], cv2.COLOR_BGR2GRAY)
            else:
                gray1 = frames[i]
                gray2 = frames[i+1]
            
            diff = cv2.absdiff(gray1, gray2)
            pixel_diffs.append(diff)
        
        pixel_diff_array = np.array(pixel_diffs)
        
        return RawAnalysis(
            method=self.METHOD_NAME,
            data={'pixel_diff': pixel_diff_array},
            parameters={
                'threshold': self.threshold,
                'downsample': self.downsample
            },
            processing_time=time.time() - start_time,
            output_shapes={'pixel_diff': pixel_diff_array.shape},
            dtype_info={'pixel_diff': str(pixel_diff_array.dtype)}
        )


class EdgeCanny(BaseAnalysis):
    """Canny edge detection analysis."""
    
    METHOD_NAME = 'edge_canny'
    
    def __init__(self, downsample: float = 1.0, 
                 low_threshold: int = 50, high_threshold: int = 150):
        """
        Args:
            downsample: Resolution scale
            low_threshold: Lower threshold for edge detection
            high_threshold: Upper threshold for edge detection
        """
        super().__init__(downsample=downsample)
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
    
    def analyze(self, frames: List[np.ndarray]) -> RawAnalysis:
        """Detect edges in each frame using Canny edge detector."""
        start_time = time.time()
        frames = self.preprocess_frames(frames)
        
        # Convert to grayscale
        gray_frames = []
        for frame in frames:
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            gray_frames.append(gray)
        
        edge_maps = []
        gradient_magnitudes = []
        gradient_directions = []
        
        for gray in gray_frames:
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
            
            # Detect edges
            edges = cv2.Canny(blurred, self.low_threshold, self.high_threshold)
            edge_maps.append(edges)
            
            # Calculate gradients for additional info
            grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
            
            magnitude = np.sqrt(grad_x**2 + grad_y**2)
            direction = np.arctan2(grad_y, grad_x)
            
            gradient_magnitudes.append(magnitude.astype(np.float32))
            gradient_directions.append(direction.astype(np.float32))
        
        return RawAnalysis(
            method=self.METHOD_NAME,
            data={
                'edge_map': np.array(edge_maps),
                'gradient_magnitude': np.array(gradient_magnitudes),
                'gradient_direction': np.array(gradient_directions)
            },
            parameters={
                'downsample': self.downsample,
                'low_threshold': self.low_threshold,
                'high_threshold': self.high_threshold
            },
            processing_time=time.time() - start_time,
            output_shapes={
                'edge_map': np.array(edge_maps).shape,
                'gradient_magnitude': np.array(gradient_magnitudes).shape,
                'gradient_direction': np.array(gradient_directions).shape
            },
            dtype_info={
                'edge_map': 'uint8',
                'gradient_magnitude': 'float32',
                'gradient_direction': 'float32'
            }
        )


# =============================================================================
# LEVEL 2: Intermediate Analyses
# =============================================================================

class FrameDiffAdvanced(BaseAnalysis):
    """Advanced frame differencing with multiple techniques."""
    
    METHOD_NAME = 'frame_diff_advanced'
    
    def __init__(self, downsample: float = 1.0, 
                 window_size: int = 5, accumulate: bool = True):
        """
        Args:
            downsample: Resolution scale
            window_size: Size of temporal window for running average
            accumulate: Whether to compute accumulated differences
        """
        super().__init__(downsample=downsample)
        self.window_size = window_size
        self.accumulate = accumulate
    
    def analyze(self, frames: List[np.ndarray]) -> RawAnalysis:
        """Compute advanced frame differences including triple diff and running average."""
        start_time = time.time()
        frames = self.preprocess_frames(frames)
        
        # Convert to grayscale
        gray_frames = []
        for frame in frames:
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            gray_frames.append(gray)
        
        if len(gray_frames) < 3:
            raise ValueError("Need at least 3 frames for advanced differencing")
        
        h, w = gray_frames[0].shape
        
        # Initialize outputs
        triple_diffs = []
        running_avg_diffs = []
        accumulated_diff = np.zeros((h, w), dtype=np.float32)
        running_avg = np.float32(gray_frames[0])
        
        # Process frames
        for i in range(2, len(gray_frames)):
            # Triple frame differencing (reduces noise)
            diff1 = cv2.absdiff(gray_frames[i-2], gray_frames[i-1])
            diff2 = cv2.absdiff(gray_frames[i-1], gray_frames[i])
            triple_diff = cv2.bitwise_and(diff1, diff2)
            triple_diffs.append(triple_diff)
            
            # Running average background subtraction
            cv2.accumulateWeighted(gray_frames[i], running_avg, 0.02)  # Learning rate
            background = np.uint8(running_avg)
            diff_background = cv2.absdiff(gray_frames[i], background)
            running_avg_diffs.append(diff_background)
            
            # Accumulated differences (motion history)
            if self.accumulate:
                simple_diff = cv2.absdiff(gray_frames[i-1], gray_frames[i])
                accumulated_diff = accumulated_diff * 0.95  # Decay factor
                accumulated_diff += simple_diff.astype(np.float32) / 255.0
        
        # Normalize accumulated diff
        accumulated_norm = np.uint8(np.clip(accumulated_diff * 255, 0, 255))
        
        return RawAnalysis(
            method=self.METHOD_NAME,
            data={
                'triple_diff': np.array(triple_diffs) if triple_diffs else None,
                'running_avg_diff': np.array(running_avg_diffs) if running_avg_diffs else None,
                'accumulated_diff': accumulated_norm if self.accumulate else None
            },
            parameters={
                'downsample': self.downsample,
                'window_size': self.window_size,
                'accumulate': self.accumulate
            },
            processing_time=time.time() - start_time,
            output_shapes={
                'triple_diff': np.array(triple_diffs).shape if triple_diffs else None,
                'running_avg_diff': np.array(running_avg_diffs).shape if running_avg_diffs else None,
                'accumulated_diff': accumulated_norm.shape if self.accumulate else None
            },
            dtype_info={
                'triple_diff': 'uint8',
                'running_avg_diff': 'uint8',
                'accumulated_diff': 'uint8'
            }
        )


class ContourDetection(BaseAnalysis):
    """Contour and shape detection analysis."""
    
    METHOD_NAME = 'contour_detection'
    
    def __init__(self, downsample: float = 0.5,  # Often downsample for contours
                 threshold: int = 127, max_contours: int = 100):
        """
        Args:
            downsample: Resolution scale (default 0.5 for performance)
            threshold: Binary threshold for contour detection
            max_contours: Maximum number of contours to keep per frame
        """
        super().__init__(downsample=downsample)
        self.threshold = threshold
        self.max_contours = max_contours
    
    def analyze(self, frames: List[np.ndarray]) -> RawAnalysis:
        """Find and track shape boundaries in each frame."""
        start_time = time.time()
        frames = self.preprocess_frames(frames)
        
        # Convert to grayscale
        gray_frames = []
        for frame in frames:
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            gray_frames.append(gray)
        
        all_contours = []
        all_hierarchies = []
        
        # Process each frame pair
        for i in range(1, len(gray_frames)):
            # Calculate frame difference
            diff = cv2.absdiff(gray_frames[i-1], gray_frames[i])
            
            # Apply threshold to get binary mask
            _, thresh = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)
            
            # Apply morphological operations to clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)  # Close gaps
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)   # Remove noise
            
            # Find contours
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter and keep only significant contours
            significant_contours = []
            significant_hierarchy = []
            
            for j, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area > 100:  # Minimum area threshold
                    significant_contours.append(contour)
                    if hierarchy is not None:
                        significant_hierarchy.append(hierarchy[0][j])
            
            # Sort by area and keep only top N
            if len(significant_contours) > self.max_contours:
                areas = [cv2.contourArea(c) for c in significant_contours]
                indices = np.argsort(areas)[-self.max_contours:]
                significant_contours = [significant_contours[i] for i in indices]
                if significant_hierarchy:
                    significant_hierarchy = [significant_hierarchy[i] for i in indices]
            
            all_contours.append(significant_contours)
            all_hierarchies.append(np.array(significant_hierarchy) if significant_hierarchy else None)
        
        return RawAnalysis(
            method=self.METHOD_NAME,
            data={
                'contours': all_contours,
                'hierarchy': all_hierarchies
            },
            parameters={
                'downsample': self.downsample,
                'threshold': self.threshold,
                'max_contours': self.max_contours
            },
            processing_time=time.time() - start_time,
            output_shapes={
                'contours': f"List[{len(all_contours)}][varies]",
                'hierarchy': f"List[{len(all_hierarchies)}]"
            },
            dtype_info={
                'contours': 'List[np.ndarray]',
                'hierarchy': 'List[np.ndarray]'
            }
        )


# =============================================================================
# LEVEL 3: Advanced Analyses
# =============================================================================

class BackgroundMOG2(BaseAnalysis):
    """MOG2 background subtraction analysis."""
    
    METHOD_NAME = 'background_mog2'
    
    def __init__(self, downsample: float = 0.5,  # Often downsample for speed
                 history: int = 120, var_threshold: float = 16.0,
                 detect_shadows: bool = True):
        """
        Args:
            downsample: Resolution scale (default 0.5 for performance)
            history: Number of frames for background model
            var_threshold: Variance threshold for background model
            detect_shadows: Whether to detect shadows
        """
        super().__init__(downsample=downsample)
        self.history = history
        self.var_threshold = var_threshold
        self.detect_shadows = detect_shadows
    
    def analyze(self, frames: List[np.ndarray]) -> RawAnalysis:
        """Learn background model and detect foreground objects."""
        start_time = time.time()
        frames = self.preprocess_frames(frames)
        
        # Create MOG2 background subtractor
        mog2 = cv2.createBackgroundSubtractorMOG2(
            detectShadows=self.detect_shadows,
            varThreshold=self.var_threshold,
            history=self.history
        )
        
        foreground_masks = []
        
        for frame in frames:
            # Apply background subtraction
            mask = mog2.apply(frame)
            
            # Remove shadows if detected (shadows are 127, foreground is 255)
            if self.detect_shadows:
                mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)[1]
            
            # Apply morphological operations to reduce noise
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            foreground_masks.append(mask)
        
        foreground_array = np.array(foreground_masks)
        
        return RawAnalysis(
            method=self.METHOD_NAME,
            data={'foreground_mask': foreground_array},
            parameters={
                'downsample': self.downsample,
                'history': self.history,
                'var_threshold': self.var_threshold,
                'detect_shadows': self.detect_shadows
            },
            processing_time=time.time() - start_time,
            output_shapes={'foreground_mask': foreground_array.shape},
            dtype_info={'foreground_mask': str(foreground_array.dtype)}
        )


class BackgroundKNN(BaseAnalysis):
    """KNN background subtraction analysis."""
    
    METHOD_NAME = 'background_knn'
    
    def __init__(self, downsample: float = 0.5,
                 history: int = 200, dist2_threshold: float = 400.0,
                 detect_shadows: bool = False):
        """
        Args:
            downsample: Resolution scale
            history: Number of frames for background model
            dist2_threshold: Distance threshold for KNN
            detect_shadows: Whether to detect shadows
        """
        super().__init__(downsample=downsample)
        self.history = history
        self.dist2_threshold = dist2_threshold
        self.detect_shadows = detect_shadows
    
    def analyze(self, frames: List[np.ndarray]) -> RawAnalysis:
        """K-nearest neighbors background model."""
        start_time = time.time()
        frames = self.preprocess_frames(frames)
        
        # Create KNN background subtractor
        knn = cv2.createBackgroundSubtractorKNN(
            detectShadows=self.detect_shadows,
            dist2Threshold=self.dist2_threshold,
            history=self.history
        )
        
        foreground_masks = []
        
        for frame in frames:
            # Apply background subtraction
            mask = knn.apply(frame)
            
            # Remove shadows if detected
            if self.detect_shadows:
                mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)[1]
            
            # Apply morphological operations to reduce noise
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            foreground_masks.append(mask)
        
        foreground_array = np.array(foreground_masks)
        
        return RawAnalysis(
            method=self.METHOD_NAME,
            data={'foreground_mask': foreground_array},
            parameters={
                'downsample': self.downsample,
                'history': self.history,
                'dist2_threshold': self.dist2_threshold,
                'detect_shadows': self.detect_shadows
            },
            processing_time=time.time() - start_time,
            output_shapes={'foreground_mask': foreground_array.shape},
            dtype_info={'foreground_mask': str(foreground_array.dtype)}
        )


class OpticalFlowSparse(BaseAnalysis):
    """Lucas-Kanade sparse optical flow analysis."""
    
    METHOD_NAME = 'optical_flow_sparse'
    
    def __init__(self, downsample: float = 1.0,  # Usually full res for accuracy
                 max_corners: int = 100, quality_level: float = 0.3,
                 min_distance: int = 7):
        """
        Args:
            downsample: Resolution scale (default 1.0 for accuracy)
            max_corners: Maximum number of corners to track
            quality_level: Quality level for corner detection
            min_distance: Minimum distance between corners
        """
        super().__init__(downsample=downsample)
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
    
    def analyze(self, frames: List[np.ndarray]) -> RawAnalysis:
        """Track specific feature points across frames."""
        start_time = time.time()
        frames = self.preprocess_frames(frames)
        
        if len(frames) < 2:
            raise ValueError("Need at least 2 frames for optical flow")
        
        # Convert to grayscale
        gray_frames = []
        for frame in frames:
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            gray_frames.append(gray)
        
        # Parameters for ShiTomasi corner detection
        feature_params = dict(
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            blockSize=7
        )
        
        # Parameters for Lucas-Kanade optical flow
        lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # Track points across all frames
        all_tracked_points = []
        all_point_status = []
        
        # Detect initial features in first frame
        p0 = cv2.goodFeaturesToTrack(gray_frames[0], mask=None, **feature_params)
        
        if p0 is not None:
            for i in range(1, len(gray_frames)):
                # Calculate optical flow
                p1, st, err = cv2.calcOpticalFlowPyrLK(
                    gray_frames[i-1], gray_frames[i], p0, None, **lk_params
                )
                
                # Select good points
                if p1 is not None:
                    good_new = p1[st == 1]
                    good_old = p0[st == 1]
                    
                    # Store tracked points
                    tracked = []
                    for j, (new, old) in enumerate(zip(good_new, good_old)):
                        tracked.append({
                            'id': j,
                            'old_pos': old.tolist(),
                            'new_pos': new.tolist(),
                            'dx': float(new[0] - old[0]),
                            'dy': float(new[1] - old[1])
                        })
                    
                    all_tracked_points.append(tracked)
                    all_point_status.append(st.flatten())
                    
                    # Update points for next iteration
                    p0 = good_new.reshape(-1, 1, 2)
                    
                    # Redetect features if too few points remain
                    if len(p0) < 20:
                        p0 = cv2.goodFeaturesToTrack(gray_frames[i], mask=None, **feature_params)
                else:
                    all_tracked_points.append([])
                    all_point_status.append(np.array([]))
        
        return RawAnalysis(
            method=self.METHOD_NAME,
            data={
                'tracked_points': all_tracked_points,
                'point_status': all_point_status
            },
            parameters={
                'downsample': self.downsample,
                'max_corners': self.max_corners,
                'quality_level': self.quality_level,
                'min_distance': self.min_distance
            },
            processing_time=time.time() - start_time,
            output_shapes={
                'tracked_points': f"List[{len(all_tracked_points)}]",
                'point_status': f"List[{len(all_point_status)}]"
            },
            dtype_info={
                'tracked_points': 'List[Dict]',
                'point_status': 'List[np.ndarray]'
            }
        )


# =============================================================================
# LEVEL 4: Complex Analyses (Computationally intensive)
# =============================================================================

class OpticalFlowDense(BaseAnalysis):
    """Farneback dense optical flow analysis."""
    
    METHOD_NAME = 'optical_flow_dense'
    
    def __init__(self, downsample: float = 0.25,  # Heavy computation, downsample!
                 pyr_scale: float = 0.5, levels: int = 3,
                 winsize: int = 15, iterations: int = 3):
        """
        Args:
            downsample: Resolution scale (default 0.25 for performance)
            pyr_scale: Pyramid scale factor
            levels: Number of pyramid levels
            winsize: Window size for averaging
            iterations: Number of iterations at each pyramid level
        """
        super().__init__(downsample=downsample)
        self.pyr_scale = pyr_scale
        self.levels = levels
        self.winsize = winsize
        self.iterations = iterations
    
    def analyze(self, frames: List[np.ndarray]) -> RawAnalysis:
        """Calculate motion vectors for every pixel between frames."""
        start_time = time.time()
        frames = self.preprocess_frames(frames)
        
        # Convert to grayscale
        gray_frames = []
        for frame in frames:
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            gray_frames.append(gray)
        
        # Compute optical flow
        flow_fields = []
        for i in range(len(gray_frames) - 1):
            flow = cv2.calcOpticalFlowFarneback(
                gray_frames[i], gray_frames[i+1], None,
                self.pyr_scale, self.levels, self.winsize,
                self.iterations, 5, 1.2, 0
            )
            flow_fields.append(flow)
        
        flow_array = np.array(flow_fields)
        
        return RawAnalysis(
            method=self.METHOD_NAME,
            data={'flow_field': flow_array},
            parameters={
                'downsample': self.downsample,
                'pyr_scale': self.pyr_scale,
                'levels': self.levels,
                'winsize': self.winsize,
                'iterations': self.iterations
            },
            processing_time=time.time() - start_time,
            output_shapes={'flow_field': flow_array.shape},
            dtype_info={'flow_field': str(flow_array.dtype)}
        )


class MotionHeatmap(BaseAnalysis):
    """Motion heatmap accumulation analysis."""
    
    METHOD_NAME = 'motion_heatmap'
    
    def __init__(self, downsample: float = 0.25,  # Heavy memory usage
                 decay_factor: float = 0.95, snapshot_interval: int = 30):
        """
        Args:
            downsample: Resolution scale (default 0.25 for memory)
            decay_factor: Decay factor for weighted heatmap
            snapshot_interval: Frames between snapshots
        """
        super().__init__(downsample=downsample)
        self.decay_factor = decay_factor
        self.snapshot_interval = snapshot_interval
    
    def analyze(self, frames: List[np.ndarray]) -> RawAnalysis:
        """Accumulate motion over time to find activity zones."""
        start_time = time.time()
        frames = self.preprocess_frames(frames)
        
        if len(frames) < 2:
            raise ValueError("Need at least 2 frames for motion heatmap")
        
        # Convert to grayscale
        gray_frames = []
        for frame in frames:
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            gray_frames.append(gray)
        
        h, w = gray_frames[0].shape
        
        # Initialize heatmaps
        cumulative_heatmap = np.zeros((h, w), dtype=np.float32)
        weighted_heatmap = np.zeros((h, w), dtype=np.float32)
        heatmap_snapshots = []
        
        # Process frame pairs
        for i in range(1, len(gray_frames)):
            # Calculate frame difference for motion
            diff = cv2.absdiff(gray_frames[i-1], gray_frames[i])
            motion_mask = diff.astype(np.float32) / 255.0
            
            # Update heatmaps
            cumulative_heatmap += motion_mask
            weighted_heatmap = weighted_heatmap * self.decay_factor + motion_mask
            
            # Take snapshots at intervals
            if i % self.snapshot_interval == 0:
                snapshot_time = i / len(gray_frames)
                heatmap_snapshots.append((snapshot_time, weighted_heatmap.copy()))
        
        # Normalize heatmaps
        if cumulative_heatmap.max() > 0:
            cumulative_heatmap = cumulative_heatmap / cumulative_heatmap.max()
        if weighted_heatmap.max() > 0:
            weighted_heatmap = weighted_heatmap / weighted_heatmap.max()
        
        # Convert to uint8 for storage
        cumulative_uint8 = (cumulative_heatmap * 255).astype(np.uint8)
        weighted_uint8 = (weighted_heatmap * 255).astype(np.uint8)
        
        return RawAnalysis(
            method=self.METHOD_NAME,
            data={
                'cumulative': cumulative_uint8,
                'weighted': weighted_uint8,
                'snapshots': heatmap_snapshots
            },
            parameters={
                'downsample': self.downsample,
                'decay_factor': self.decay_factor,
                'snapshot_interval': self.snapshot_interval
            },
            processing_time=time.time() - start_time,
            output_shapes={
                'cumulative': cumulative_uint8.shape,
                'weighted': weighted_uint8.shape,
                'snapshots': f"List[{len(heatmap_snapshots)}]"
            },
            dtype_info={
                'cumulative': 'uint8',
                'weighted': 'uint8',
                'snapshots': 'List[Tuple[float, np.ndarray]]'
            }
        )


class FrequencyFFT(BaseAnalysis):
    """Frequency analysis using FFT."""
    
    METHOD_NAME = 'frequency_fft'
    
    def __init__(self, downsample: float = 0.1,  # Very small for FFT
                 window_size: int = 64, overlap: float = 0.5):
        """
        Args:
            downsample: Resolution scale (default 0.1 for FFT)
            window_size: Size of temporal window for FFT
            overlap: Overlap between windows (0.0 to 1.0)
        """
        super().__init__(downsample=downsample)
        self.window_size = window_size
        self.overlap = overlap
    
    def analyze(self, frames: List[np.ndarray]) -> RawAnalysis:
        """Analyze temporal frequency of pixel changes."""
        start_time = time.time()
        frames = self.preprocess_frames(frames)
        
        if len(frames) < self.window_size:
            raise ValueError(f"Need at least {self.window_size} frames for FFT analysis")
        
        # Convert to grayscale
        gray_frames = []
        for frame in frames:
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            gray_frames.append(gray.astype(np.float32) / 255.0)
        
        gray_array = np.array(gray_frames)  # [T, H, W]
        
        # Sample pixels for FFT (too expensive to do all pixels)
        h, w = gray_array.shape[1:]
        step = max(1, int(1 / np.sqrt(self.downsample)))  # Sample pixels based on downsample
        sampled_pixels = gray_array[:, ::step, ::step]  # [T, H', W']
        
        # Prepare for FFT analysis
        frequency_spectrums = []
        phase_spectrums = []
        
        # Sliding window FFT
        stride = int(self.window_size * (1 - self.overlap))
        
        for start in range(0, len(gray_frames) - self.window_size + 1, stride):
            window = sampled_pixels[start:start + self.window_size]
            
            # Apply FFT along time axis for each pixel
            fft_result = np.fft.fft(window, axis=0)
            
            # Get magnitude and phase
            magnitude = np.abs(fft_result)
            phase = np.angle(fft_result)
            
            # Average across spatial dimensions for summary
            avg_magnitude = np.mean(magnitude, axis=(1, 2))
            avg_phase = np.mean(phase, axis=(1, 2))
            
            frequency_spectrums.append(avg_magnitude[:self.window_size//2])  # Keep positive frequencies
            phase_spectrums.append(avg_phase[:self.window_size//2])
        
        return RawAnalysis(
            method=self.METHOD_NAME,
            data={
                'frequency_spectrum': np.array(frequency_spectrums),
                'phase_spectrum': np.array(phase_spectrums)
            },
            parameters={
                'downsample': self.downsample,
                'window_size': self.window_size,
                'overlap': self.overlap
            },
            processing_time=time.time() - start_time,
            output_shapes={
                'frequency_spectrum': np.array(frequency_spectrums).shape,
                'phase_spectrum': np.array(phase_spectrums).shape
            },
            dtype_info={
                'frequency_spectrum': 'float64',
                'phase_spectrum': 'float64'
            }
        )


class FlowHSVViz(BaseAnalysis):
    """HSV visualization of optical flow."""
    
    METHOD_NAME = 'flow_hsv_viz'
    
    def __init__(self, downsample: float = 0.5,
                 max_magnitude: float = 20.0, saturation_boost: float = 1.5):
        """
        Args:
            downsample: Resolution scale
            max_magnitude: Maximum flow magnitude for normalization
            saturation_boost: Boost factor for saturation
        """
        super().__init__(downsample=downsample)
        self.max_magnitude = max_magnitude
        self.saturation_boost = saturation_boost
    
    def analyze(self, frames: List[np.ndarray]) -> RawAnalysis:
        """Convert optical flow to HSV color representation."""
        start_time = time.time()
        frames = self.preprocess_frames(frames)
        
        if len(frames) < 2:
            raise ValueError("Need at least 2 frames for flow visualization")
        
        # Convert to grayscale
        gray_frames = []
        for frame in frames:
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            gray_frames.append(gray)
        
        hsv_flows = []
        
        for i in range(1, len(gray_frames)):
            # Calculate optical flow
            flow = cv2.calcOpticalFlowFarneback(
                gray_frames[i-1], gray_frames[i], None,
                pyr_scale=0.5, levels=2, winsize=15,
                iterations=2, poly_n=5, poly_sigma=1.1, flags=0
            )
            
            # Convert flow to HSV
            h, w = flow.shape[:2]
            fx, fy = flow[:,:,0], flow[:,:,1]
            
            # Calculate magnitude and angle
            mag, ang = cv2.cartToPolar(fx, fy)
            
            # Create HSV image
            hsv = np.zeros((h, w, 3), dtype=np.uint8)
            
            # Hue = direction (shifted for better colors)
            hsv[:,:,0] = (ang * 180 / np.pi / 2 + 90) % 180
            
            # Saturation = magnitude (more motion = more saturated)
            sat_normalized = np.minimum(mag * 20 * self.saturation_boost, 255)
            hsv[:,:,1] = sat_normalized.astype(np.uint8)
            
            # Value = magnitude (normalized)
            normalized_mag = np.minimum(mag * self.max_magnitude, 255)
            hsv[:,:,2] = normalized_mag.astype(np.uint8)
            
            # Convert HSV to BGR for storage
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            hsv_flows.append(bgr)
        
        hsv_array = np.array(hsv_flows)
        
        return RawAnalysis(
            method=self.METHOD_NAME,
            data={'hsv_flow': hsv_array},
            parameters={
                'downsample': self.downsample,
                'max_magnitude': self.max_magnitude,
                'saturation_boost': self.saturation_boost
            },
            processing_time=time.time() - start_time,
            output_shapes={'hsv_flow': hsv_array.shape},
            dtype_info={'hsv_flow': str(hsv_array.dtype)}
        )


# =============================================================================
# Analysis Registry
# =============================================================================

ANALYSIS_REGISTRY = {
    # Level 1: Basic (Real-time capable)
    'frame_diff': FrameDiff,
    'edge_canny': EdgeCanny,
    
    # Level 2: Intermediate
    'frame_diff_advanced': FrameDiffAdvanced,
    'contour_detection': ContourDetection,
    
    # Level 3: Advanced
    'background_mog2': BackgroundMOG2,
    'background_knn': BackgroundKNN,
    'optical_flow_sparse': OpticalFlowSparse,
    
    # Level 4: Complex (Computationally intensive)
    'optical_flow_dense': OpticalFlowDense,
    'motion_heatmap': MotionHeatmap,
    'frequency_fft': FrequencyFFT,
    'flow_hsv_viz': FlowHSVViz
}

# Default configurations for common use cases
FAST_CONFIG = {
    'frame_diff': {'downsample': 0.5},
    'optical_flow_dense': {'downsample': 0.25, 'levels': 2},
    'motion_heatmap': {'downsample': 0.25},
    'contour_detection': {'downsample': 0.25, 'max_contours': 50},
}

QUALITY_CONFIG = {
    'frame_diff': {'downsample': 1.0},
    'optical_flow_dense': {'downsample': 0.5, 'levels': 5, 'iterations': 5},
    'motion_heatmap': {'downsample': 0.5, 'decay_factor': 0.98},
    'contour_detection': {'downsample': 0.75, 'max_contours': 200},
}

DEFAULT_PIPELINE = ['frame_diff']

RECOMMENDED_PIPELINE = [
    'frame_diff',           # Level 1: Quick activity detection
    'contour_detection',    # Level 2: Structure analysis
    'background_mog2',      # Level 3: New element detection
    'optical_flow_dense'    # Level 4: Motion patterns
]