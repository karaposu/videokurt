"""Temporal activity pattern analysis."""

import numpy as np
from typing import Dict, Any, List

from ..base import MiddleFeature


class TemporalActivityPatterns(MiddleFeature):
    """Extract temporal patterns from activity data."""
    
    FEATURE_NAME = 'temporal_activity_patterns'
    REQUIRED_ANALYSES = ['frame_diff']
    
    def __init__(self, window_size: int = 30, overlap: float = 0.5):
        """
        Args:
            window_size: Size of temporal window
            overlap: Overlap between windows (0-1)
        """
        super().__init__()
        self.window_size = window_size
        self.overlap = overlap
    
    def _compute_middle(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract temporal activity patterns.
        
        Returns:
            Dict with temporal patterns and statistics
        """
        frame_diff_data = analysis_data['frame_diff'].data['pixel_diff']
        
        # Compute activity timeline
        activity_timeline = []
        for diff in frame_diff_data:
            activity = np.mean(diff)
            activity_timeline.append(activity)
        
        activity_timeline = np.array(activity_timeline)
        
        if len(activity_timeline) < self.window_size:
            return {
                'patterns': [],
                'pattern_changes': [],
                'activity_phases': []
            }
        
        # Extract windowed patterns
        patterns = self._extract_patterns(activity_timeline)
        
        # Detect pattern changes
        pattern_changes = self._detect_pattern_changes(patterns)
        
        # Identify activity phases
        activity_phases = self._identify_phases(activity_timeline)
        
        return {
            'patterns': patterns,
            'pattern_changes': pattern_changes,
            'activity_phases': activity_phases,
            'mean_activity': float(np.mean(activity_timeline)),
            'activity_variance': float(np.var(activity_timeline))
        }
    
    def _extract_patterns(self, timeline: np.ndarray) -> List[Dict]:
        """Extract patterns from temporal windows."""
        patterns = []
        stride = int(self.window_size * (1 - self.overlap))
        
        for i in range(0, len(timeline) - self.window_size + 1, stride):
            window = timeline[i:i + self.window_size]
            
            # Compute pattern features
            pattern = {
                'start': i,
                'end': i + self.window_size,
                'mean': float(np.mean(window)),
                'std': float(np.std(window)),
                'trend': self._compute_trend(window),
                'peaks': self._count_peaks(window)
            }
            patterns.append(pattern)
        
        return patterns
    
    def _compute_trend(self, window: np.ndarray) -> str:
        """Compute trend in window."""
        if len(window) < 2:
            return 'flat'
        
        # Fit linear trend
        x = np.arange(len(window))
        coeffs = np.polyfit(x, window, 1)
        slope = coeffs[0]
        
        if abs(slope) < 0.01:
            return 'flat'
        elif slope > 0:
            return 'increasing'
        else:
            return 'decreasing'
    
    def _count_peaks(self, window: np.ndarray) -> int:
        """Count peaks in window."""
        peaks = 0
        for i in range(1, len(window) - 1):
            if window[i] > window[i-1] and window[i] > window[i+1]:
                peaks += 1
        return peaks
    
    def _detect_pattern_changes(self, patterns: List[Dict]) -> List[Dict]:
        """Detect significant pattern changes."""
        changes = []
        
        for i in range(1, len(patterns)):
            prev = patterns[i-1]
            curr = patterns[i]
            
            # Check for significant change
            mean_change = abs(curr['mean'] - prev['mean']) / (prev['mean'] + 1e-6)
            
            if mean_change > 0.5 or prev['trend'] != curr['trend']:
                changes.append({
                    'frame': curr['start'],
                    'type': 'trend_change' if prev['trend'] != curr['trend'] else 'intensity_change',
                    'magnitude': mean_change
                })
        
        return changes
    
    def _identify_phases(self, timeline: np.ndarray) -> List[Dict]:
        """Identify distinct activity phases."""
        # Simple thresholding approach
        mean_activity = np.mean(timeline)
        std_activity = np.std(timeline)
        
        high_threshold = mean_activity + std_activity
        low_threshold = mean_activity - std_activity
        
        phases = []
        current_phase = None
        phase_start = 0
        
        for i, activity in enumerate(timeline):
            if activity > high_threshold:
                phase_type = 'high'
            elif activity < low_threshold:
                phase_type = 'low'
            else:
                phase_type = 'medium'
            
            if current_phase != phase_type:
                if current_phase is not None:
                    phases.append({
                        'type': current_phase,
                        'start': phase_start,
                        'end': i,
                        'duration': i - phase_start
                    })
                current_phase = phase_type
                phase_start = i
        
        # Add final phase
        if current_phase is not None:
            phases.append({
                'type': current_phase,
                'start': phase_start,
                'end': len(timeline),
                'duration': len(timeline) - phase_start
            })
        
        return phases
