"""Simple pixel-wise frame differencing implementation."""

from typing import Optional
import numpy as np
from .base import FrameDifferencer, DifferenceResult


class SimpleFrameDiff(FrameDifferencer):
    """Basic pixel-wise frame differencing.
    
    Fast and simple algorithm suitable for detecting any visual change.
    Best for real-time processing and initial activity detection.
    """
    
    def __init__(
        self,
        blur_kernel: int = 5,
        noise_threshold: int = 10,
        normalize: bool = True,
        **config
    ):
        """Initialize simple frame differencer.
        
        Args:
            blur_kernel: Gaussian blur kernel size for noise reduction
            noise_threshold: Minimum pixel difference to consider
            normalize: Whether to normalize output scores
        """
        self.blur_kernel = blur_kernel
        self.noise_threshold = noise_threshold
        self.normalize = normalize
        super().__init__(**config)
    
    def compute_difference(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> DifferenceResult:
        """Compute simple pixel-wise difference."""
        # Preprocess frames
        gray1, gray2 = self._preprocess_frames(frame1, frame2)
        
        # Apply Gaussian blur for noise reduction
        if self.blur_kernel > 0:
            try:
                import cv2
                gray1 = cv2.GaussianBlur(gray1, (self.blur_kernel, self.blur_kernel), 0)
                gray2 = cv2.GaussianBlur(gray2, (self.blur_kernel, self.blur_kernel), 0)
            except ImportError:
                # Simple box filter fallback if cv2 not available
                from scipy.ndimage import uniform_filter
                gray1 = uniform_filter(gray1, size=self.blur_kernel).astype(np.uint8)
                gray2 = uniform_filter(gray2, size=self.blur_kernel).astype(np.uint8)
        
        # Compute absolute difference
        diff = np.abs(gray1.astype(float) - gray2.astype(float)).astype(np.uint8)
        
        # Apply noise threshold
        diff[diff < self.noise_threshold] = 0
        
        # Apply mask if provided
        if mask is not None:
            diff = diff * (mask > 0).astype(np.uint8)
        
        # Calculate metrics
        total_pixels = diff.shape[0] * diff.shape[1]
        
        # Compute scores
        # Always use full frame for mean to make scores comparable
        mean_diff = np.mean(diff)
        max_diff = np.max(diff)
        changed_pixels = np.sum(diff > self.noise_threshold)
        
        # Normalize score
        if self.normalize:
            score = mean_diff / 255.0
        else:
            score = mean_diff
        
        metadata = {
            'mean_diff': mean_diff,
            'max_diff': max_diff,
            'changed_pixels': changed_pixels,
            'changed_ratio': changed_pixels / total_pixels if total_pixels > 0 else 0,
            'algorithm': 'simple'
        }
        
        return DifferenceResult(score=score, diff_mask=diff, metadata=metadata)