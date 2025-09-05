# Motion Magnitude Feature - Trigger Logic

## What This Feature Detects

Motion magnitude measures the overall amount of movement in each frame, providing a simple scalar value representing total activity level.

## What TRIGGERS High Motion Magnitude

### 1. Large-Scale Changes
- Full page scrolling
- Window maximizing/minimizing
- Screen transitions
- Video playback
- Slideshow advances
- Application switching

### 2. Multiple Simultaneous Changes
- Multiple windows moving
- Animated backgrounds + user interaction
- Live dashboards with multiple updates
- Multi-panel interfaces updating
- Parallel animations

### 3. Rapid Interactions
- Fast mouse movements
- Quick dragging operations
- Rapid clicking/tapping
- Gesture inputs
- Gaming actions

### 4. Content Updates
- Video streaming
- Live data feeds
- Animation playback
- Screen sharing
- Broadcasting content

## What TRIGGERS Low Motion Magnitude

### 1. Static Viewing
- Reading text
- Examining images
- Watching paused video
- Reviewing documents
- Idle time

### 2. Minimal Interactions
- Slow typing
- Occasional clicks
- Hover effects only
- Cursor movements only
- Small UI updates

### 3. Localized Changes
- Blinking cursor
- Small spinner
- Status bar updates
- Clock ticking
- Single button highlights

## Magnitude Levels

| Level | Value Range | Typical Scenario |
|-------|-------------|------------------|
| None | 0 | Completely static |
| Minimal | 0-0.1 | Cursor movement only |
| Low | 0.1-0.3 | Typing, small updates |
| Medium | 0.3-0.6 | Scrolling, transitions |
| High | 0.6-0.8 | Video, animations |
| Very High | 0.8-1.0 | Rapid changes, gaming |

## Common Patterns

### Document Work
```
[Low] → [Medium spike] → [Low]
Typing → Scrolling → Reading
```

### Video Watching
```
[High continuous]
Steady high magnitude during playback
```

### Web Browsing
```
[Medium] → [Low] → [Medium]
Navigate → Read → Navigate
```

### Gaming
```
[Very High with variations]
Action-dependent spikes
```

### Idle
```
[Near zero]
Minimal or no change
```

## Advantages

1. **Simple metric** - Single value per frame
2. **Universal** - Works for any content type
3. **Intuitive** - Higher = more activity
4. **Efficient** - Low computational cost
5. **Reliable** - No false patterns

## Limitations

1. **No direction info** - Just amount, not where/how
2. **No semantic meaning** - Can't distinguish scroll from video
3. **Treats all motion equal** - Cursor same as window move
4. **Resolution dependent** - Values change with video size
5. **Noise sensitive** - Compression affects values

## Use Cases

1. **Activity detection** - When is user active vs idle
2. **Engagement measurement** - How much interaction
3. **Performance monitoring** - Detecting lag/stuttering
4. **Content classification** - Static vs dynamic content
5. **Segmentation** - Finding scene boundaries

## Comparison with Other Features

| Feature | Motion Magnitude | Alternative |
|---------|-----------------|-------------|
| Direction | ❌ No | ✅ Optical flow |
| Object tracking | ❌ No | ✅ Blob tracking |
| Pattern type | ❌ No | ✅ Scrolling detection |
| Specific regions | ❌ No | ✅ Zone activity |
| Semantic meaning | ❌ No | ✅ Scene detection |

## Best Practices

1. **Combine with other features** for context
2. **Use thresholds** appropriate to content type
3. **Smooth values** to reduce noise
4. **Normalize** across different resolutions
5. **Consider patterns** not just individual values

## Typical Output

```python
{
    'magnitude': 0.42,  # Current frame
    'timeline': [0.1, 0.1, 0.4, 0.8, 0.7, ...],  # Per frame
    'average': 0.35,
    'peak': 0.92,
    'std_dev': 0.18
}
```

## Key Insight

Motion magnitude is the "temperature reading" of video activity - it tells you how much is happening but not what or why. Best used as a complementary metric alongside more specific features.