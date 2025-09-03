"""Pattern detection modules for VideoKurt.

Analyzes frame differences to identify specific motion patterns like
video playback, scrolling, and user interactions.
"""

from .video_detector import VideoDetector, VideoRegion
from .scroll_detector import ScrollDetector, ScrollEvent
from .activity_classifier import ActivityClassifier, ActivityState

__all__ = [
    'VideoDetector',
    'VideoRegion',
    'ScrollDetector', 
    'ScrollEvent',
    'ActivityClassifier',
    'ActivityState'
]