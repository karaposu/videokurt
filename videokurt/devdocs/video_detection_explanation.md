# Video Detection and Idle State Differentiation - Explanation

## The Problem You've Identified

Your Instagram scrolling example perfectly illustrates a critical distinction that VideoKurt needs to handle:

**Screen Activity ≠ User Activity**

When a user opens Instagram and stops scrolling, they may be idle (not interacting), but the screen remains highly active due to:
- Autoplaying videos in the feed
- Animated ads
- Stories previews
- Live streams
- GIF comments

## Why This Matters for VideoKurt

### 1. **False Positive Problem**
Without video detection, VideoKurt would mark these periods as "active" even though the user is idle. This breaks the fundamental purpose of VideoKurt - identifying when users are actually doing something vs when they've stepped away.

### 2. **Cost Implications**
If VideoQuery processes every frame where a video is playing as "important activity," it wastes significant compute resources analyzing frames that contain no user intent or action.

### 3. **Accuracy of Analysis**
For VideoQuery to answer questions like "What did the user do?", it needs to distinguish between:
- User scrolling through content (active)
- Video playing while user is away (passive)
- User watching a video intentionally (active but stationary)

## Two Distinct Concepts

### 1. **Screen Activity**
- Pixel changes on screen
- Could be from videos, animations, notifications
- Detected by basic frame differencing
- High activity doesn't mean user is present

### 2. **User Activity**
- Actual user interactions
- Scrolling, clicking, typing, navigating
- Requires pattern recognition beyond pixel changes
- The true signal VideoQuery needs

## How Video Detection Helps

Video detection allows VideoKurt to:

1. **Identify Repeating Patterns**: Videos have characteristic patterns:
   - Consistent frame rate changes (24/30/60 fps)
   - Localized to specific screen regions
   - Predictable motion patterns
   - Often have player controls visible

2. **Separate Signal from Noise**: 
   - Mark video regions as "ambient motion"
   - Focus on changes outside video areas
   - Detect when user interacts with video (pause/play/seek)

3. **Create Richer Metadata**:
   ```python
   {
       "screen_active": true,
       "user_active": false,
       "video_regions": [(100, 200, 640, 480)],
       "activity_type": "video_playback",
       "confidence": 0.85
   }
   ```

## Architecture Recommendation - BALANCED APPROACH

### The Right Level of Engineering
VideoKurt needs to be smart enough to distinguish different types of motion, but not so complex it becomes unmaintainable. We need specific detectors for specific patterns because:
- Video playback has distinct characteristics (frame rate, localized region)
- Scrolling has distinct characteristics (vertical motion, content shift)
- Each pattern needs different detection logic

### Modular but Focused

```
videokurt/
├── core/
│   ├── frame_differencer/     # KEEP AS IS - pixel changes
│   │   ├── simple.py
│   │   ├── histogram.py
│   │   └── ssim.py
│   │
│   └── pattern_detectors/      # NEW: Specific pattern detection
│       ├── __init__.py
│       ├── video_detector.py  # Detect video playback regions
│       ├── scroll_detector.py # Detect scrolling motion
│       └── activity_classifier.py # Combine signals
```

## Focused Implementation

### Video Detector - Specific Logic for Video
```python
class VideoDetector:
    """
    Detects video playback regions in screen recordings.
    Videos have specific patterns we can identify.
    """
    
    def detect_video_regions(self, 
                            frame_sequence: List[np.ndarray],
                            fps: float = 30.0) -> List[VideoRegion]:
        """
        Identifies regions with video playback characteristics:
        - Consistent temporal changes at video frame rates
        - Localized to rectangular regions
        - Different from scrolling or user interaction
        """
        # Real video detection logic
        # Not just "repeating motion"
```

### Scroll Detector - Specific Logic for Scrolling
```python
class ScrollDetector:
    """
    Detects scrolling motion in screen recordings.
    Scrolling has distinct patterns different from video.
    """
    
    def detect_scroll(self,
                     frame_diffs: List[DifferenceResult]) -> ScrollEvent:
        """
        Identifies scrolling patterns:
        - Vertical or horizontal content shift
        - Consistent direction
        - Affects large screen area
        """
        # Real scroll detection logic
```

### Why We Need Separate Detectors

1. **Video vs Scrolling are DIFFERENT**:
   - Video: Localized, consistent frame rate, bounded region
   - Scrolling: Full-width/height motion, directional, content shift

2. **Different Detection Methods**:
   - Video: Temporal frequency analysis, region stability
   - Scrolling: Optical flow, edge continuity

3. **Different Metadata Needed**:
   - Video: Region bounds, playback detected
   - Scrolling: Direction, speed, distance

### Integration with VideoQuery
```python
# VideoKurt provides rich metadata
metadata = {
    "timestamp": 1234567890,
    "screen_activity": 0.8,      # High due to video
    "user_activity": 0.1,         # Low, user idle
    "video_detected": True,
    "video_regions": [...],
    "recommended_for_analysis": False  # Save VideoQuery compute
}
```

## Benefits of This Approach

1. **Accurate User Presence Detection**
   - No false positives from autoplaying videos
   - Better detection of actual user departure

2. **Smarter Resource Allocation**
   - VideoQuery skips frames with only video playback
   - Focuses compute on frames with user intent

3. **Richer Context for VideoQuery**
   - "User watched video for 3 minutes"
   - "User scrolled past 5 videos without watching"
   - "Background video played while user was away"

4. **Extensible Pattern Library**
   - Can add detection for:
     - Slideshow/carousel detection
     - Animation detection
     - Notification detection
     - Screen saver detection

## Key Insight - Pattern-Specific Detection

Your observation is correct: **Screen activity ≠ User activity**

VideoKurt needs to recognize different motion patterns because they mean different things:
- **Video playback**: Screen active, user might be idle
- **Scrolling**: User definitely active
- **Static with video**: Mixed signals need careful interpretation

## Practical Next Steps

1. Keep `frame_differencer` as the foundation (pixel-level changes)
2. Add `pattern_detectors` module with specific detectors:
   - `video_detector.py` - Identify video regions
   - `scroll_detector.py` - Detect scrolling events
   - `activity_classifier.py` - Combine signals into user/screen state
3. VideoQuery gets rich metadata:
   - What type of activity (video, scroll, click, idle)
   - Confidence levels
   - Specific regions affected

## The Balance

VideoKurt should:
- Be **specific enough** to correctly identify video vs scrolling vs user interaction
- Be **simple enough** to run efficiently as a preprocessing step
- Provide **rich enough** metadata for VideoQuery to make smart decisions

This solves your Instagram problem properly - detecting that a video is playing in a specific region while the user is idle, which is very different from the user actively scrolling through their feed.