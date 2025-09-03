"""Binary timeline representing active/inactive periods."""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import numpy as np


@dataclass
class ActivityPeriod:
    """A period of activity or inactivity."""
    start_frame: int
    end_frame: int
    is_active: bool
    confidence: float = 1.0
    
    @property
    def duration_frames(self) -> int:
        """Duration in frames."""
        return self.end_frame - self.start_frame + 1


class BinaryTimeline:
    """Binary representation of activity over time.
    
    This is the core VideoKurt output - a simple binary timeline
    showing when the video is "active" (user doing something) vs "inactive".
    """
    
    def __init__(self, total_frames: int):
        """Initialize binary timeline.
        
        Args:
            total_frames: Total number of frames
        """
        self.binary_array = np.zeros(total_frames, dtype=bool)
        self.confidence_array = np.ones(total_frames, dtype=np.float32)
        
    def mark_active(
        self,
        start_frame: int,
        end_frame: int,
        confidence: float = 1.0
    ) -> None:
        """Mark a range of frames as active.
        
        Args:
            start_frame: Start frame index
            end_frame: End frame index
            confidence: Confidence level
        """
        start = max(0, min(start_frame, len(self.binary_array) - 1))
        end = max(start, min(end_frame, len(self.binary_array) - 1))
        
        self.binary_array[start:end+1] = True
        
        # Update confidence (take maximum)
        self.confidence_array[start:end+1] = np.maximum(
            self.confidence_array[start:end+1],
            confidence
        )
    
    def mark_inactive(
        self,
        start_frame: int,
        end_frame: int,
        confidence: float = 1.0
    ) -> None:
        """Mark a range of frames as inactive.
        
        Args:
            start_frame: Start frame index
            end_frame: End frame index  
            confidence: Confidence level
        """
        start = max(0, min(start_frame, len(self.binary_array) - 1))
        end = max(start, min(end_frame, len(self.binary_array) - 1))
        
        self.binary_array[start:end+1] = False
        
        # Update confidence
        self.confidence_array[start:end+1] = np.maximum(
            self.confidence_array[start:end+1],
            confidence
        )
    
    def from_timeline(self, timeline: 'Timeline') -> None:
        """Populate from a Timeline object.
        
        Args:
            timeline: Timeline with events
        """
        from .timeline_builder import EventType
        
        # Reset arrays
        self.binary_array.fill(False)
        self.confidence_array.fill(0.0)
        
        for entry in timeline.entries:
            # Determine if this event represents activity
            is_active = entry.event_type not in [EventType.IDLE, EventType.VIDEO_PLAYBACK]
            
            # Special case: video playback is inactive if user is not interacting
            if entry.event_type == EventType.VIDEO_PLAYBACK:
                # Check metadata for user activity
                is_active = entry.metadata.get('user_active', False)
            
            if is_active:
                self.mark_active(entry.frame_start, entry.frame_end, entry.confidence)
            else:
                # Don't override existing active marks with inactive
                # Only mark inactive if not already marked active
                for frame in range(entry.frame_start, entry.frame_end + 1):
                    if not self.binary_array[frame]:
                        self.binary_array[frame] = False
                        self.confidence_array[frame] = max(
                            self.confidence_array[frame],
                            entry.confidence
                        )
    
    def get_periods(self, min_duration_frames: int = 1) -> List[ActivityPeriod]:
        """Get list of active/inactive periods.
        
        Args:
            min_duration_frames: Minimum duration to report
            
        Returns:
            List of ActivityPeriod objects
        """
        periods = []
        
        if len(self.binary_array) == 0:
            return periods
        
        # Find transitions
        current_state = self.binary_array[0]
        current_start = 0
        
        for i in range(1, len(self.binary_array)):
            if self.binary_array[i] != current_state:
                # State changed, record period
                duration = i - current_start
                
                if duration >= min_duration_frames:
                    # Calculate average confidence for this period
                    avg_confidence = np.mean(self.confidence_array[current_start:i])
                    
                    periods.append(ActivityPeriod(
                        start_frame=current_start,
                        end_frame=i - 1,
                        is_active=current_state,
                        confidence=float(avg_confidence)
                    ))
                
                current_state = self.binary_array[i]
                current_start = i
        
        # Add final period
        duration = len(self.binary_array) - current_start
        if duration >= min_duration_frames:
            avg_confidence = np.mean(self.confidence_array[current_start:])
            
            periods.append(ActivityPeriod(
                start_frame=current_start,
                end_frame=len(self.binary_array) - 1,
                is_active=current_state,
                confidence=float(avg_confidence)
            ))
        
        return periods
    
    def smooth(self, window_size: int = 5) -> None:
        """Smooth the timeline to remove noise.
        
        Args:
            window_size: Size of smoothing window
        """
        if window_size < 2 or len(self.binary_array) < window_size:
            return
        
        # Use majority voting in sliding window
        smoothed = np.zeros_like(self.binary_array)
        
        for i in range(len(self.binary_array)):
            start = max(0, i - window_size // 2)
            end = min(len(self.binary_array), i + window_size // 2 + 1)
            
            # Majority vote
            window = self.binary_array[start:end]
            smoothed[i] = np.sum(window) > len(window) // 2
        
        self.binary_array = smoothed
    
    def fill_short_gaps(self, max_gap_frames: int = 10) -> None:
        """Fill short gaps between active periods.
        
        Args:
            max_gap_frames: Maximum gap to fill
        """
        periods = self.get_periods()
        
        # Look for inactive periods between active ones
        for i in range(len(periods) - 2):
            # Check for pattern: active -> inactive -> active
            if (periods[i].is_active and 
                not periods[i + 1].is_active and 
                i + 2 < len(periods) and
                periods[i + 2].is_active):
                
                gap_duration = periods[i + 1].duration_frames
                
                if gap_duration <= max_gap_frames:
                    # Fill the gap
                    start = periods[i + 1].start_frame
                    end = periods[i + 1].end_frame
                    
                    self.binary_array[start:end+1] = True
                    # Override confidence for filled gaps
                    self.confidence_array[start:end+1] = 0.5
    
    def remove_short_periods(self, min_duration_frames: int = 5) -> None:
        """Remove very short active/inactive periods (likely noise).
        
        Args:
            min_duration_frames: Minimum duration to keep
        """
        periods = self.get_periods(min_duration_frames=1)
        
        for period in periods:
            if period.duration_frames < min_duration_frames:
                # Flip the state
                if period.is_active:
                    self.mark_inactive(period.start_frame, period.end_frame)
                else:
                    self.mark_active(period.start_frame, period.end_frame)
    
    def get_activity_ratio(self) -> float:
        """Get ratio of active frames to total frames."""
        if len(self.binary_array) == 0:
            return 0.0
        return np.sum(self.binary_array) / len(self.binary_array)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the timeline."""
        periods = self.get_periods()
        
        active_periods = [p for p in periods if p.is_active]
        inactive_periods = [p for p in periods if not p.is_active]
        
        total_active_frames = sum(p.duration_frames for p in active_periods)
        total_inactive_frames = sum(p.duration_frames for p in inactive_periods)
        
        stats = {
            'total_frames': len(self.binary_array),
            'fps': 30.0,
            'duration_seconds': len(self.binary_array) / 30.0 if 30.0 > 0 else 0,
            'activity_ratio': self.get_activity_ratio(),
            'num_periods': len(periods),
            'num_active_periods': len(active_periods),
            'num_inactive_periods': len(inactive_periods),
            'total_active_frames': total_active_frames,
            'total_inactive_frames': total_inactive_frames,
            'total_active_seconds': total_active_frames / 30.0 if 30.0 > 0 else 0,
            'total_inactive_seconds': total_inactive_frames / 30.0 if 30.0 > 0 else 0,
        }
        
        # Add period statistics
        if active_periods:
            active_durations = [p.duration_frames for p in active_periods]
            stats['avg_active_duration_frames'] = np.mean(active_durations)
            stats['max_active_duration_frames'] = np.max(active_durations)
            stats['min_active_duration_frames'] = np.min(active_durations)
        
        if inactive_periods:
            inactive_durations = [p.duration_frames for p in inactive_periods]
            stats['avg_inactive_duration_frames'] = np.mean(inactive_durations)
            stats['max_inactive_duration_frames'] = np.max(inactive_durations)
            stats['min_inactive_duration_frames'] = np.min(inactive_durations)
        
        return stats
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        periods = self.get_periods()
        
        return {
            'total_frames': len(self.binary_array),
            'fps': 30.0,
            'duration_seconds': len(self.binary_array) / 30.0 if 30.0 > 0 else 0,
            'activity_ratio': self.get_activity_ratio(),
            'periods': [
                {
                    'start_frame': p.start_frame,
                    'end_frame': p.end_frame,
                    'start_seconds': p.start_frame / 30.0 if 30.0 > 0 else 0,
                    'end_seconds': p.end_frame / 30.0 if 30.0 > 0 else 0,
                    'duration_frames': p.duration_frames,
                    'duration_seconds': p.duration_frames / 30.0 if 30.0 > 0 else 0,
                    'is_active': p.is_active,
                    'confidence': p.confidence
                }
                for p in periods
            ],
            'statistics': self.get_statistics(),
            'metadata': {
                'version': '1.0',
                'type': 'binary_timeline'
            }
        }