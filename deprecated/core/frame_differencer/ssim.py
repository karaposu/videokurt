"""Structural Similarity Index (SSIM) based frame comparison."""

from typing import Optional
import numpy as np
from .base import FrameDifferencer, DifferenceResult


class SSIMFrameDiff(FrameDifferencer):
    """Structural Similarity Index (SSIM) based comparison.
    
    Perceptual similarity metric that considers luminance, contrast, and structure.
    More aligned with human perception but computationally heavier.
    """
    
    def __init__(
        self,
        window_size: int = 7,
        gaussian_weights: bool = True,
        multichannel: bool = False,
        **config
    ):
        """Initialize SSIM differencer.
        
        Args:
            window_size: Size of the sliding window for SSIM
            gaussian_weights: Use Gaussian weighting for the window
            multichannel: Process color channels separately
        """
        self.window_size = window_size
        self.gaussian_weights = gaussian_weights
        self.multichannel = multichannel
        super().__init__(**config)
    
    def compute_difference(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> DifferenceResult:
        """Compute SSIM-based difference."""
        try:
            from skimage.metrics import structural_similarity as ssim
            
            # Preprocess frames
            if not self.multichannel:
                gray1, gray2 = self._preprocess_frames(frame1, frame2)
                
                # Compute SSIM
                ssim_score, ssim_image = ssim(
                    gray1,
                    gray2,
                    win_size=self.window_size,
                    gaussian_weights=self.gaussian_weights,
                    full=True
                )
                
                # Convert SSIM to difference (SSIM is similarity, we want difference)
                score = 1.0 - ssim_score
                
                # Create difference mask from SSIM image
                diff_mask = (1.0 - ssim_image * 255).astype(np.uint8)
            else:
                # Process each channel
                ssim_scores = []
                for i in range(3):
                    channel_ssim, _ = ssim(
                        frame1[:, :, i],
                        frame2[:, :, i],
                        win_size=self.window_size,
                        gaussian_weights=self.gaussian_weights,
                        full=True
                    )
                    ssim_scores.append(channel_ssim)
                
                ssim_score = np.mean(ssim_scores)
                score = 1.0 - ssim_score
                
                # Simple diff mask for multichannel
                gray1, gray2 = self._preprocess_frames(frame1, frame2)
                diff_mask = np.abs(gray1.astype(float) - gray2.astype(float)).astype(np.uint8)
                
        except ImportError:
            # Fallback SSIM implementation without skimage
            gray1, gray2 = self._preprocess_frames(frame1, frame2)
            
            # Simple SSIM approximation
            c1 = (0.01 * 255) ** 2
            c2 = (0.03 * 255) ** 2
            
            # Calculate means
            mu1 = self._window_average(gray1, self.window_size)
            mu2 = self._window_average(gray2, self.window_size)
            
            mu1_sq = mu1 ** 2
            mu2_sq = mu2 ** 2
            mu1_mu2 = mu1 * mu2
            
            # Calculate variances and covariance
            sigma1_sq = self._window_average(gray1 ** 2, self.window_size) - mu1_sq
            sigma2_sq = self._window_average(gray2 ** 2, self.window_size) - mu2_sq
            sigma12 = self._window_average(gray1 * gray2, self.window_size) - mu1_mu2
            
            # SSIM formula
            ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
                       ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
            
            ssim_score = np.mean(ssim_map)
            score = 1.0 - ssim_score
            
            # Create difference mask
            diff_mask = ((1.0 - ssim_map) * 255).astype(np.uint8)
        
        # Apply mask if provided
        if mask is not None:
            diff_mask = diff_mask * (mask > 0).astype(np.uint8)
        
        metadata = {
            'ssim_score': ssim_score if 'ssim_score' in locals() else 1.0 - score,
            'window_size': self.window_size,
            'gaussian_weights': self.gaussian_weights,
            'multichannel': self.multichannel,
            'algorithm': 'ssim'
        }
        
        return DifferenceResult(score=score, diff_mask=diff_mask, metadata=metadata)
    
    def _window_average(self, image: np.ndarray, window_size: int) -> np.ndarray:
        """Calculate windowed average for SSIM computation."""
        from scipy.ndimage import uniform_filter
        return uniform_filter(image.astype(float), size=window_size)