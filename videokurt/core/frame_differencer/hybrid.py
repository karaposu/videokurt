"""Hybrid frame differencing combining multiple methods."""

from typing import Optional, Dict
import numpy as np
from .base import FrameDifferencer, DifferenceResult
from .simple import SimpleFrameDiff
from .histogram import HistogramFrameDiff
from .ssim import SSIMFrameDiff


class HybridFrameDiff(FrameDifferencer):
    """Hybrid approach combining multiple differencing methods.
    
    Combines simple, histogram, and SSIM methods with configurable weights.
    Provides most robust detection at the cost of performance.
    """
    
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        voting_threshold: float = 0.05,
        **config
    ):
        """Initialize hybrid differencer.
        
        Args:
            weights: Weight for each method {'simple': 0.4, 'histogram': 0.3, 'ssim': 0.3}
            voting_threshold: Threshold for voting-based decision
        """
        self.weights = weights or {'simple': 0.4, 'histogram': 0.3, 'ssim': 0.3}
        self.voting_threshold = voting_threshold
        
        # Initialize component differencers
        self.simple_diff = SimpleFrameDiff()
        self.histogram_diff = HistogramFrameDiff()
        self.ssim_diff = SSIMFrameDiff()
        
        super().__init__(**config)
    
    def compute_difference(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> DifferenceResult:
        """Compute hybrid difference using multiple methods."""
        results = {}
        
        # Compute all differences
        results['simple'] = self.simple_diff.compute_difference(frame1, frame2, mask)
        results['histogram'] = self.histogram_diff.compute_difference(frame1, frame2, mask)
        results['ssim'] = self.ssim_diff.compute_difference(frame1, frame2, mask)
        
        # Weighted average of scores
        weighted_score = 0.0
        for method, weight in self.weights.items():
            weighted_score += results[method].score * weight
        
        # Voting mechanism
        votes = sum(1 for r in results.values() if r.score > self.voting_threshold)
        consensus = votes >= 2
        
        # Use simple diff mask as the combined mask
        diff_mask = results['simple'].diff_mask
        
        metadata = {
            'individual_scores': {k: v.score for k, v in results.items()},
            'weights': self.weights,
            'votes': votes,
            'consensus': consensus,
            'algorithm': 'hybrid'
        }
        
        return DifferenceResult(score=weighted_score, diff_mask=diff_mask, metadata=metadata)