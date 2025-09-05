"""Activity burst detection from temporal patterns."""

import numpy as np
from typing import Dict, Any, List

from ..base import BaseFeature


class ActivityBursts(BaseFeature):
    """Detect periods of high activity followed by low activity."""
    
    FEATURE_NAME = 'activity_bursts'
    REQUIRED_ANALYSES = ['frame_diff']
    
    def __init__(self, burst_threshold: float = 0.5, 
                 min_burst_length: int = 3,
                 smoothing_window: int = 5):
        """
        Args:
            burst_threshold: Threshold for high activity (normalized)
            min_burst_length: Minimum frames for a burst
            smoothing_window: Window size for activity smoothing
        """
        super().__init__()
        self.burst_threshold = burst_threshold
        self.min_burst_length = min_burst_length
        self.smoothing_window = smoothing_window
    
    def compute(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect activity bursts from frame differences.
        
        Returns:
            Dict with burst events and statistics
        """
        self.validate_inputs(analysis_data)
        
        frame_diff_data = analysis_data['frame_diff'].data['pixel_diff']
        
        # Compute activity level per frame
        activity_levels = []
        for diff in frame_diff_data:
            activity = np.mean(diff)
            activity_levels.append(activity)
        
        activity_levels = np.array(activity_levels)
        
        if len(activity_levels) == 0:
            return {
                'bursts': [],
                'num_bursts': 0,
                'burst_ratio': 0,
                'avg_burst_intensity': 0
            }
        
        # Smooth activity signal
        smoothed = self._smooth_signal(activity_levels, self.smoothing_window)
        
        # Normalize to 0-1
        if smoothed.max() > 0:
            normalized = smoothed / smoothed.max()
        else:
            normalized = smoothed
        
        # Detect bursts
        bursts = self._detect_bursts(normalized)
        
        # Compute burst statistics
        burst_frames = sum(b['duration'] for b in bursts)
        burst_ratio = burst_frames / len(activity_levels) if len(activity_levels) > 0 else 0
        
        avg_intensity = np.mean([b['peak_intensity'] for b in bursts]) if bursts else 0
        
        # Classify burst patterns
        burst_patterns = self._classify_patterns(bursts, len(activity_levels))
        
        return {
            'bursts': bursts,
            'num_bursts': len(bursts),
            'burst_ratio': burst_ratio,
            'avg_burst_intensity': avg_intensity,
            'burst_patterns': burst_patterns,
            'activity_timeline': normalized
        }
    
    def _smooth_signal(self, signal: np.ndarray, window: int) -> np.ndarray:
        """Apply moving average smoothing."""
        if len(signal) < window:
            return signal
        
        kernel = np.ones(window) / window
        # Pad signal to handle edges
        padded = np.pad(signal, (window//2, window//2), mode='edge')
        smoothed = np.convolve(padded, kernel, mode='valid')
        
        return smoothed[:len(signal)]
    
    def _detect_bursts(self, activity: np.ndarray) -> List[Dict]:
        """Detect burst periods in activity signal."""
        bursts = []
        in_burst = False
        burst_start = 0
        
        for i, level in enumerate(activity):
            if not in_burst and level >= self.burst_threshold:
                # Start of burst
                in_burst = True
                burst_start = i
            elif in_burst and level < self.burst_threshold:
                # End of burst
                duration = i - burst_start
                if duration >= self.min_burst_length:
                    burst_activity = activity[burst_start:i]
                    bursts.append({
                        'start': burst_start,
                        'end': i,
                        'duration': duration,
                        'peak_intensity': float(np.max(burst_activity)),
                        'avg_intensity': float(np.mean(burst_activity))
                    })
                in_burst = False
        
        # Handle burst that extends to end
        if in_burst:
            duration = len(activity) - burst_start
            if duration >= self.min_burst_length:
                burst_activity = activity[burst_start:]
                bursts.append({
                    'start': burst_start,
                    'end': len(activity),
                    'duration': duration,
                    'peak_intensity': float(np.max(burst_activity)),
                    'avg_intensity': float(np.mean(burst_activity))
                })
        
        return bursts
    
    def _classify_patterns(self, bursts: List[Dict], total_frames: int) -> Dict:
        """Classify burst patterns."""
        if not bursts:
            return {'pattern': 'none', 'regularity': 0}
        
        if len(bursts) == 1:
            return {'pattern': 'single', 'regularity': 0}
        
        # Check for regular patterns
        intervals = []
        for i in range(1, len(bursts)):
            interval = bursts[i]['start'] - bursts[i-1]['end']
            intervals.append(interval)
        
        if intervals:
            avg_interval = np.mean(intervals)
            std_interval = np.std(intervals)
            regularity = 1.0 - (std_interval / (avg_interval + 1e-6))
            
            if regularity > 0.7:
                pattern = 'periodic'
            elif len(bursts) > 5:
                pattern = 'frequent'
            else:
                pattern = 'sporadic'
        else:
            pattern = 'single'
            regularity = 0
        
        return {
            'pattern': pattern,
            'regularity': float(regularity) if intervals else 0,
            'avg_interval': float(avg_interval) if intervals else 0
        }