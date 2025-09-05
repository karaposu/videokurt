# Activity Bursts Feature - Trigger Logic

## What This Feature Detects

Activity bursts identifies periods of intense activity followed by calmer periods. It detects when screen activity suddenly increases above a threshold.

## What TRIGGERS Burst Detection in Screen Recordings

### 1. Rapid Navigation
- Quick clicking through multiple pages
- Fast tab switching
- Rapid menu navigation
- Quick file browsing
- Speed scrolling through content

### 2. Intensive Interactions
- Drag and drop operations
- Window resizing/repositioning
- Multi-select operations
- Rapid form filling
- Quick sketch/drawing

### 3. Animations and Transitions
- Page transition animations
- Slideshow transitions
- Video seeking/scrubbing
- Animated popups appearing
- Loading sequences

### 4. Content Updates
- Live data dashboards updating
- Chat messages arriving rapidly
- Notification floods
- Stock tickers updating
- Social media feed refreshes

### 5. User Actions
- Fast typing bursts
- Rapid clicking (gaming, testing)
- Gesture sequences
- Keyboard shortcuts in succession
- Macro executions

## What DOESN'T Trigger Bursts

### 1. Steady Activity
- Continuous scrolling at constant speed
- Video playback
- Smooth animations
- Consistent typing pace
- Gradual transitions

### 2. Static Periods
- Reading time
- Idle periods
- Paused video
- Static presentations
- Form review before submission

### 3. Slow Changes
- Gradual fade effects
- Slow page loads
- Progressive rendering
- Subtle hover effects

## Burst Characteristics

### Typical Burst Patterns
- **Duration**: 3-20 frames for UI bursts
- **Intensity**: 2-5x baseline activity
- **Recovery**: Quick return to baseline
- **Frequency**: Irregular, user-driven

### Activity Types
1. **Navigation bursts**: Page changes, tab switches
2. **Interaction bursts**: Clicking, dragging
3. **Content bursts**: Updates, refreshes
4. **Input bursts**: Typing, shortcuts
5. **Animation bursts**: Transitions, effects

## Example Results Interpretation

```
Frequent short bursts = Active user interaction
Few long bursts = Major transitions or operations
Regular bursts = Automated or periodic updates
No bursts = Steady or static usage
```

## Common Patterns in Screen Recordings

### Web Browsing
- Bursts when: Opening links, switching tabs
- Calm when: Reading content
- Pattern: Irregular bursts

### Document Editing
- Bursts when: Formatting, rapid typing
- Calm when: Thinking, reading
- Pattern: Clustered bursts

### Video Watching
- Bursts when: Seeking, play/pause
- Calm when: Watching
- Pattern: Rare bursts

### Gaming
- Bursts when: Action sequences
- Calm when: Menus, cutscenes
- Pattern: Game-dependent

### Coding
- Bursts when: Autocomplete, refactoring
- Calm when: Thinking, reading code
- Pattern: Periodic bursts

## Threshold Considerations

| Threshold | Effect |
|-----------|--------|
| Low (0.3) | Detects subtle changes, many false positives |
| Medium (0.5) | Balanced detection, good for general use |
| High (0.7) | Only major changes, may miss interactions |

## Best Use Cases

1. **User behavior analysis** - Identifying active vs passive periods
2. **Performance monitoring** - Detecting UI lag or stuttering
3. **Workflow analysis** - Finding intensive work periods
4. **Engagement tracking** - Measuring user interaction levels
5. **Content analysis** - Identifying dynamic vs static content

## Limitations

- Cannot distinguish between user-driven and system-driven bursts
- May miss subtle but important interactions
- Threshold-dependent results
- Treats all activity types equally
- No semantic understanding of burst causes

## Typical Output

```python
{
    'bursts': [
        {'start': 45, 'end': 52, 'intensity': 0.8},
        {'start': 120, 'end': 128, 'intensity': 0.6}
    ],
    'num_bursts': 12,
    'burst_ratio': 0.15,  # 15% of time in burst state
    'avg_burst_intensity': 0.65,
    'avg_burst_duration': 6.5  # frames
}
```