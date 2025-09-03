# VideoKurt

## Overview

VideoKurt is a low-level video analysis module that uses OpenCV to detect mechanical visual changes and predefined event patterns in video content. It serves as the computer vision foundation for VideoQuery, marking scene boundaries and activity periods without understanding semantic meaning.

## Core Purpose

VideoKurt **marks visual changes and detects predefined mechanical events** in video content.

It provides:
- Scene change boundaries (when frames differ significantly)
- Binary activity tracking (active vs inactive periods)
- Detection of specific visual patterns (scrolls, popups, loading indicators)
- Frame-level metadata without semantic interpretation

VideoKurt does NOT tell you what is happening or why - it only detects mechanical, visual changes based on pixel analysis and predefined patterns.

## Architecture

### Binary Activity Timeline

VideoKurt tracks activity as a simple binary state over time:

```python
{
    "activity": [
        {"active": true, "start": 0.0, "end": 3.5},    # User interacting
        {"active": false, "start": 3.5, "end": 5.0},   # Idle/waiting
        {"active": true, "start": 5.0, "end": 8.2},    # Activity resumes
        {"active": false, "start": 8.2, "end": 9.0}    # Loading/processing
    ]
}
```

**Active = True**: Significant visual changes, user interactions, content updates
**Active = False**: Static screens, loading states, idle periods

### Event Detection

Alongside activity tracking, VideoKurt identifies specific mechanical events:

```python
{
    "events": [
        {
            "type": "scene_change",
            "start": 0.0,
            "end": 0.1,
            "confidence": 0.95,
            "metadata": {
                "transition_type": "cut",
                "delta_score": 0.87
            }
        },
        {
            "type": "scroll",
            "start": 2.3,
            "end": 3.1,
            "confidence": 0.92,
            "metadata": {
                "direction": "down",
                "velocity": 250,  # pixels/second
                "total_distance": 750
            }
        },
        {
            "type": "popup",
            "start": 5.0,
            "end": 8.2,
            "confidence": 0.88,
            "metadata": {
                "bounds": {"x": 100, "y": 200, "width": 400, "height": 300},
                "overlay_type": "modal",
                "background_dimmed": true
            }
        },
        {
            "type": "prompted_wait",
            "start": 8.2,
            "end": 9.0,
            "confidence": 0.91,
            "metadata": {
                "indicator": "spinner",
                "location": "center"
            }
        }
    ]
}
```

## Event Types

### 1. Scene Change
- **What**: Significant visual transition between frames
- **Detection**: Frame differencing, histogram comparison, SSIM scoring
- **Use Case**: Identify navigation, screen switches, content updates
- **Metadata**: Transition type (cut/fade), change magnitude

### 2. Scroll
- **What**: Vertical or horizontal content movement
- **Detection**: Optical flow analysis, edge tracking
- **Use Case**: User browsing content, searching through lists
- **Metadata**: Direction, velocity, distance traveled

### 3. Idle Wait
- **What**: No significant visual changes for extended period
- **Detection**: Frame similarity above threshold for minimum duration
- **Use Case**: Skip unnecessary processing, identify stuck states
- **Metadata**: Duration, last activity before idle

### 4. Prompted Wait
- **What**: System-indicated loading or processing
- **Detection**: Spinner/progress bar detection, loading indicator patterns
- **Use Case**: Distinguish intentional waits from idle periods
- **Metadata**: Indicator type, position, animation pattern

### 5. Popup/Modal
- **What**: Overlay content appearing on screen
- **Detection**: New bounded region detection, background dimming, z-index changes
- **Use Case**: Track dialogs, alerts, overlays
- **Metadata**: Bounds, overlay type, background state

### 6. Text Input
- **What**: Active typing or text field interaction
- **Detection**: Cursor blinking, character appearance, input field focus
- **Use Case**: Capture form filling, search queries
- **Metadata**: Field location, input rate

### 7. Click/Tap
- **What**: Point interaction with UI
- **Detection**: Visual feedback (ripple effects, color changes), element state changes
- **Use Case**: Track user interactions, button presses
- **Metadata**: Coordinates, target element bounds

### 8. Persistent UI Frame (Advanced)
- **What**: Fixed navigation elements that remain constant (bottom tabs, headers, sidebars)
- **Detection**: Template matching of UI chrome, layout analysis, region stability over time
- **Use Case**: Confirm app continuity, track navigation state, detect context switches
- **Metadata**: Layout type, active tab/section, app identifier, stability score

## Calibration System

VideoKurt uses calibration profiles to tune detection for different contexts:

```python
calibration = {
    "scene_change": {
        "threshold": 0.3,        # 0-1, lower = more sensitive
        "min_duration": 0.1,     # Minimum seconds to register
        "method": "histogram"    # histogram, ssim, or hybrid
    },
    "scroll": {
        "min_velocity": 50,      # pixels/second
        "detect_method": "optical_flow",
        "smooth_factor": 0.8     # Smoothing for noisy detection
    },
    "idle_wait": {
        "min_duration": 1.5,     # Seconds before marking idle
        "similarity_threshold": 0.95
    },
    "prompted_wait": {
        "detect_spinners": true,
        "detect_progress": true,
        "detect_skeleton": true,
        "custom_patterns": []    # Custom loading indicators
    },
    "popup": {
        "overlay_detection": true,
        "min_coverage": 0.1,     # Minimum % of screen
        "detect_dimming": true
    },
    "persistent_ui": {
        "detect_navigation": true,
        "track_layout": true,
        "regions": ["bottom", "top", "sidebar"],
        "stability_threshold": 0.9
    }
}
```

## Detection Methods

### Frame Differencing
- Pixel-wise comparison between consecutive frames
- Fast, good for detecting any change
- Used for: Activity detection, scene changes

### Optical Flow
- Tracks motion vectors between frames
- Computationally heavier but precise
- Used for: Scroll detection, gesture tracking

### Template Matching
- Finds known patterns in frames
- Reliable for specific UI elements
- Used for: Loading indicators, standard UI components

### Histogram Analysis
- Compares color/brightness distributions
- Robust to minor changes
- Used for: Scene change detection, lighting changes

### Structural Similarity (SSIM)
- Perceptual similarity metric
- Balances luminance, contrast, structure
- Used for: Idle detection, quality assessment

### Image Detection (Advanced Feature)
- Multi-scale template matching for user-provided images
- Feature-based matching (SIFT/ORB) for robust detection
- Tracks appearance/disappearance of specific UI elements
- Returns precise timestamps when target images appear

## Output Format

Complete VideoKurt analysis output:

```python
{
    "video_info": {
        "duration": 45.5,
        "fps": 30,
        "resolution": [1920, 1080],
        "total_frames": 1365
    },
    
    "activity": [
        {"active": true, "start": 0.0, "end": 12.5},
        {"active": false, "start": 12.5, "end": 15.0},
        {"active": true, "start": 15.0, "end": 45.5}
    ],
    
    "events": [...],  # As shown above
    
    "statistics": {
        "active_time": 40.5,
        "idle_time": 5.0,
        "activity_ratio": 0.89,
        "total_events": 23,
        "dominant_event": "scroll"
    },
    
    "segments": [
        {
            "start": 0.0,
            "end": 5.0,
            "activity_score": 0.92,
            "primary_events": ["scene_change", "scroll"],
            "recommended_sampling": "high"
        },
        {
            "start": 5.0,
            "end": 10.0,
            "activity_score": 0.15,
            "primary_events": ["idle_wait"],
            "recommended_sampling": "minimal"
        }
    ]
}
```

## Performance Considerations

### Processing Modes

1. **Fast Mode**
   - Lower resolution processing (480p)
   - Skip every other frame
   - Basic detection only
   - ~10x faster than real-time

2. **Balanced Mode**
   - Medium resolution (720p)
   - Adaptive frame skipping
   - All standard detections
   - ~5x faster than real-time

3. **Thorough Mode**
   - Full resolution
   - Every frame analyzed
   - All detection methods
   - ~1-2x real-time speed

### Optimization Strategies

- **Pyramidal Processing**: Analyze at multiple resolutions
- **Region of Interest**: Focus on active screen areas
- **Adaptive Sampling**: Increase analysis during high activity
- **Early Termination**: Stop when sufficient events detected
- **Caching**: Store computed features for reuse

## Integration with VideoQuery

VideoKurt provides VideoQuery with:

1. **Intelligent Frame Selection**: Which frames to send to LLM
2. **Context Hints**: What type of activity is occurring
3. **Skip Zones**: Periods to ignore entirely
4. **Event Boundaries**: Natural cutting points for analysis
5. **Activity Intensity**: How densely to sample frames

VideoQuery uses this information to:
- Reduce LLM API calls by 80-90%
- Improve accuracy by focusing on relevant moments
- Provide better temporal context in results
- Optimize cost and performance

