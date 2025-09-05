"""Histogram statistics from color histogram analysis."""

import numpy as np
from typing import Dict, Any

from ..base import BaseFeature


class HistogramStatistics(BaseFeature):
    """Compute statistics from color histograms."""
    
    FEATURE_NAME = 'histogram_statistics'
    REQUIRED_ANALYSES = ['color_histogram']
    
    def __init__(self, compute_spread: bool = True):
        """
        Args:
            compute_spread: Whether to compute histogram spread
        """
        super().__init__()
        self.compute_spread = compute_spread
    
    def compute(self, analysis_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Compute histogram statistics.
        
        Returns:
            Dict with mean, peak, and optionally spread
        """
        self.validate_inputs(analysis_data)
        
        histogram_analysis = analysis_data['color_histogram']
        histograms = histogram_analysis.data['histograms']
        
        stats = {
            'mean': [],
            'peak': [],
        }
        
        if self.compute_spread:
            stats['spread'] = []
        
        for hist in histograms:
            # Compute weighted mean
            bins = np.arange(len(hist))
            if np.sum(hist) > 0:
                mean_val = np.average(bins, weights=hist)
                stats['mean'].append(mean_val)
                
                # Find peak (mode)
                peak_idx = np.argmax(hist)
                stats['peak'].append(peak_idx)
                
                # Compute spread (standard deviation)
                if self.compute_spread:
                    variance = np.average((bins - mean_val)**2, weights=hist)
                    spread = np.sqrt(variance)
                    stats['spread'].append(spread)
            else:
                stats['mean'].append(0)
                stats['peak'].append(0)
                if self.compute_spread:
                    stats['spread'].append(0)
        
        # Convert lists to arrays
        for key in stats:
            stats[key] = np.array(stats[key], dtype=np.float32)
        
        return stats