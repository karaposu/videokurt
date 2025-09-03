"""Histogram-based frame comparison implementation."""

from typing import Optional
import numpy as np
from .base import FrameDifferencer, DifferenceResult


class HistogramFrameDiff(FrameDifferencer):
    """Histogram-based frame comparison.
    
    Compares color/brightness distributions between frames.
    More robust to minor spatial changes but sensitive to lighting.
    """
    
    def __init__(
        self,
        bins: int = 256,
        channels: str = 'gray',
        distance_metric: str = 'correlation',
        **config
    ):
        """Initialize histogram differencer.
        
        Args:
            bins: Number of histogram bins
            channels: Color channels to use ('gray', 'rgb', 'hsv')
            distance_metric: Histogram comparison method
        """
        self.bins = bins
        self.channels = channels
        self.distance_metric = distance_metric
        super().__init__(**config)
    
    def compute_difference(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> DifferenceResult:
        """Compute histogram-based difference."""
        try:
            import cv2
            
            # Prepare frames based on channel selection
            if self.channels == 'gray':
                gray1, gray2 = self._preprocess_frames(frame1, frame2)
                hist1 = cv2.calcHist([gray1], [0], mask, [self.bins], [0, 256])
                hist2 = cv2.calcHist([gray2], [0], mask, [self.bins], [0, 256])
            elif self.channels == 'hsv':
                hsv1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
                hsv2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)
                hist1 = cv2.calcHist([hsv1], [0, 1], mask, [50, 60], [0, 180, 0, 256])
                hist2 = cv2.calcHist([hsv2], [0, 1], mask, [50, 60], [0, 180, 0, 256])
            else:  # rgb
                hist1_b = cv2.calcHist([frame1], [0], mask, [self.bins], [0, 256])
                hist1_g = cv2.calcHist([frame1], [1], mask, [self.bins], [0, 256])
                hist1_r = cv2.calcHist([frame1], [2], mask, [self.bins], [0, 256])
                hist2_b = cv2.calcHist([frame2], [0], mask, [self.bins], [0, 256])
                hist2_g = cv2.calcHist([frame2], [1], mask, [self.bins], [0, 256])
                hist2_r = cv2.calcHist([frame2], [2], mask, [self.bins], [0, 256])
                hist1 = np.concatenate([hist1_b, hist1_g, hist1_r])
                hist2 = np.concatenate([hist2_b, hist2_g, hist2_r])
            
            # Normalize histograms
            hist1 = cv2.normalize(hist1, hist1).flatten()
            hist2 = cv2.normalize(hist2, hist2).flatten()
            
            # Compute histogram distance
            if self.distance_metric == 'correlation':
                distance = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
                score = 1.0 - distance  # Convert correlation to difference
            elif self.distance_metric == 'chi_square':
                distance = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
                score = min(distance / 100.0, 1.0)  # Normalize chi-square
            elif self.distance_metric == 'intersection':
                distance = cv2.compareHist(hist1, hist2, cv2.HISTCMP_INTERSECT)
                score = 1.0 - distance
            else:  # bhattacharyya
                distance = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
                score = distance
            
        except ImportError:
            # Fallback without cv2
            gray1, gray2 = self._preprocess_frames(frame1, frame2)
            
            # Simple histogram calculation
            hist1, _ = np.histogram(gray1.flatten(), bins=self.bins, range=(0, 256))
            hist2, _ = np.histogram(gray2.flatten(), bins=self.bins, range=(0, 256))
            
            # Normalize
            hist1 = hist1.astype(float) / (hist1.sum() + 1e-10)
            hist2 = hist2.astype(float) / (hist2.sum() + 1e-10)
            
            # Simple distance metric
            if self.distance_metric == 'correlation':
                correlation = np.corrcoef(hist1, hist2)[0, 1]
                score = 1.0 - correlation
                distance = correlation
            else:
                # Bhattacharyya distance fallback
                bc = np.sum(np.sqrt(hist1 * hist2))
                distance = -np.log(bc + 1e-10)
                score = min(distance, 1.0)
        
        # Create simple diff mask for compatibility
        gray1, gray2 = self._preprocess_frames(frame1, frame2)
        diff_mask = np.abs(gray1.astype(float) - gray2.astype(float)).astype(np.uint8)
        
        metadata = {
            'histogram_distance': distance,
            'distance_metric': self.distance_metric,
            'channels': self.channels,
            'bins': self.bins,
            'algorithm': 'histogram'
        }
        
        return DifferenceResult(score=score, diff_mask=diff_mask, metadata=metadata)