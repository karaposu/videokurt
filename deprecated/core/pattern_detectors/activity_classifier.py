"""Activity classification combining all pattern detection signals.

The final arbiter of whether the user is active or the screen is just showing
ambient motion (like videos).
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from ..frame_differencer.base import DifferenceResult
from .video_detector import VideoRegion
from .scroll_detector import ScrollEvent


@dataclass
class ActivityState:
    """Current activity state assessment."""
    user_active: bool
    screen_active: bool
    activity_type: str  # 'idle', 'scrolling', 'interacting', 'watching_video'
    confidence: float
    frame_range: Tuple[int, int]
    metadata: Dict[str, Any]


class ActivityClassifier:
    """Classifies user and screen activity by combining pattern detection signals.
    
    This is the key component that solves the Instagram video problem:
    - Screen can be active (video playing) while user is idle
    - User scrolling means user is definitely active
    - No activity means both idle
    """
    
    def __init__(
        self,
        idle_threshold: float = 0.02,
        active_threshold: float = 0.1
    ):
        """Initialize activity classifier.
        
        Args:
            idle_threshold: Below this, consider idle
            active_threshold: Above this, consider active
        """
        self.idle_threshold = idle_threshold
        self.active_threshold = active_threshold
    
    def classify(
        self,
        diff_sequence: List[DifferenceResult],
        video_regions: List[VideoRegion],
        scroll_events: List[ScrollEvent],
        window_size: int = 30
    ) -> ActivityState:
        """Classify activity state from all signals.
        
        Args:
            diff_sequence: Raw frame differences
            video_regions: Detected video playback regions
            scroll_events: Detected scrolling events
            window_size: Number of frames to consider
            
        Returns:
            Current activity state
        """
        if not diff_sequence:
            return ActivityState(
                user_active=False,
                screen_active=False,
                activity_type='idle',
                confidence=1.0,
                frame_range=(0, 0),
                metadata={}
            )
        
        # Analyze the window
        window_end = min(window_size, len(diff_sequence))
        window_diffs = diff_sequence[:window_end]
        
        # 1. Check for scrolling (definite user activity)
        active_scrolls = [s for s in scroll_events 
                         if s.frame_start < window_end]
        
        if active_scrolls:
            return ActivityState(
                user_active=True,
                screen_active=True,
                activity_type='scrolling',
                confidence=max(s.confidence for s in active_scrolls),
                frame_range=(0, window_end),
                metadata={
                    'scroll_count': len(active_scrolls),
                    'scroll_directions': [s.direction for s in active_scrolls]
                }
            )
        
        # 2. Check for video playback
        active_videos = [v for v in video_regions
                        if v.frame_start < window_end]
        
        # 3. Calculate overall activity level
        activity_scores = [d.score for d in window_diffs]
        mean_activity = np.mean(activity_scores)
        
        # 4. Classify based on patterns
        if active_videos and mean_activity > self.idle_threshold:
            # Video playing - check if user is also active
            video_coverage = self._calculate_video_coverage(active_videos, window_diffs[0])
            non_video_activity = self._calculate_non_video_activity(
                window_diffs, active_videos
            )
            
            if non_video_activity > self.active_threshold:
                # User active while video plays (browsing with autoplay)
                return ActivityState(
                    user_active=True,
                    screen_active=True,
                    activity_type='interacting',
                    confidence=0.8,
                    frame_range=(0, window_end),
                    metadata={
                        'video_regions': len(active_videos),
                        'video_coverage': video_coverage,
                        'non_video_activity': non_video_activity
                    }
                )
            else:
                # Just video playing, user idle (Instagram scenario!)
                return ActivityState(
                    user_active=False,  # Key insight: user is idle
                    screen_active=True,  # But screen is active
                    activity_type='watching_video',
                    confidence=max(v.confidence for v in active_videos),
                    frame_range=(0, window_end),
                    metadata={
                        'video_regions': len(active_videos),
                        'video_coverage': video_coverage
                    }
                )
        
        # 5. No special patterns detected
        if mean_activity > self.active_threshold:
            # General activity without specific pattern
            return ActivityState(
                user_active=True,
                screen_active=True,
                activity_type='interacting',
                confidence=0.6,
                frame_range=(0, window_end),
                metadata={'mean_activity': mean_activity}
            )
        else:
            # Everything idle
            return ActivityState(
                user_active=False,
                screen_active=False,
                activity_type='idle',
                confidence=0.9,
                frame_range=(0, window_end),
                metadata={'mean_activity': mean_activity}
            )
    
    def _calculate_video_coverage(
        self,
        video_regions: List[VideoRegion],
        sample_diff: DifferenceResult
    ) -> float:
        """Calculate what fraction of screen is covered by video regions.
        
        Returns:
            Coverage ratio 0.0-1.0
        """
        if not video_regions:
            return 0.0
        
        h, w = sample_diff.diff_mask.shape
        total_area = h * w
        
        # Calculate union of video regions (simplified - doesn't handle overlap)
        video_area = sum(v.width * v.height for v in video_regions)
        
        return min(1.0, video_area / total_area)
    
    def _calculate_non_video_activity(
        self,
        diff_sequence: List[DifferenceResult],
        video_regions: List[VideoRegion]
    ) -> float:
        """Calculate activity level outside of video regions.
        
        This helps distinguish:
        - User browsing while video plays (high non-video activity)
        - User idle with autoplay (low non-video activity)
        
        Returns:
            Activity score outside video regions
        """
        if not video_regions:
            # No videos, return total activity
            return np.mean([d.score for d in diff_sequence])
        
        # Create mask for non-video areas
        h, w = diff_sequence[0].diff_mask.shape
        video_mask = np.zeros((h, w), dtype=bool)
        
        for video in video_regions:
            y1, y2 = video.y, min(video.y + video.height, h)
            x1, x2 = video.x, min(video.x + video.width, w)
            video_mask[y1:y2, x1:x2] = True
        
        non_video_mask = ~video_mask
        
        # Calculate activity in non-video regions
        activities = []
        for diff_result in diff_sequence:
            non_video_activity = diff_result.diff_mask[non_video_mask]
            if len(non_video_activity) > 0:
                activities.append(np.mean(non_video_activity) / 255.0)
        
        return np.mean(activities) if activities else 0.0