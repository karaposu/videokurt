# Stability Score Feature - Trigger Logic

## What This Feature Detects

Stability score measures how unchanged/stable each frame is compared to previous frames, outputting a score from 0 (completely changed) to 1 (identical).

## What TRIGGERS High Stability (0.8-1.0)

### 1. Idle Periods
- User reading content
- Paused video
- Inactive application
- Waiting for response
- Form review before submission

### 2. Static Content
- Fixed dashboards
- Static images
- Paused presentations
- Idle desktop
- Login screens waiting

### 3. Minimal Changes
- Clock ticking only
- Blinking cursor only
- Small status updates
- Subtle animations
- Progress indicators

### 4. Stable UI States
- Modal dialogs open
- Dropdown menus persistent
- Tooltips staying visible
- Error messages displayed
- Confirmation screens

## What TRIGGERS Low Stability (0.0-0.3)

### 1. Rapid Changes
- Fast scrolling
- Video playback
- Screen transitions
- Window switching
- Tab changes

### 2. Animations
- Loading animations
- Transition effects
- Animated backgrounds
- Particle effects
- Screensavers

### 3. Active Interaction
- Dragging operations
- Drawing/painting
- Window resizing
- Multi-selection
- Rapid typing

### 4. Dynamic Content
- Live streams
- Real-time data
- Auto-refreshing feeds
- Gaming
- Screen sharing

## What TRIGGERS Medium Stability (0.3-0.8)

### 1. Slow Changes
- Slow scrolling
- Gradual fades
- Smooth transitions
- Steady typing
- Gentle animations

### 2. Partial Updates
- Sidebar updates only
- Header changes only
- Single panel refresh
- Localized animations
- Status bar updates

### 3. Intermittent Activity
- Periodic updates
- Occasional clicks
- Sporadic typing
- Brief movements
- Quick checks

## Stability Patterns

### Reading Pattern
```
[High → High → High → Low → High]
Stable reading → Quick scroll → Continue reading
```

### Active Work Pattern
```
[Low → Medium → Low → Medium]
Continuous activity with brief pauses
```

### Video Pattern
```
[Low continuous]
Constant change during playback
```

### Idle Pattern
```
[High continuous]
Extended period of no activity
```

## Use Cases

1. **Idle detection** - Finding when user is inactive
2. **Reading time** - Measuring content consumption
3. **Interaction gaps** - Finding pauses in activity
4. **Performance issues** - Detecting frozen screens
5. **Content type** - Static vs dynamic classification

## Relationship to Other Features

| Stability Score | Motion Magnitude | Interpretation |
|-----------------|------------------|----------------|
| High | Low | True idle/reading |
| High | High | Error/contradiction |
| Low | High | Active movement |
| Low | Low | Subtle changes |

## Advantages

1. **Clear interpretation** - High = stable, Low = changing
2. **Normalized** - Always 0-1 regardless of content
3. **Sensitive** - Detects even small changes
4. **Universal** - Works for any content type
5. **Intuitive** - Matches user perception

## Limitations

1. **No change information** - Just amount, not what changed
2. **Frame-to-frame only** - No long-term patterns
3. **Can't distinguish change types** - Scroll vs video vs animation
4. **Resolution sensitive** - More pixels = more potential change
5. **Compression affects values** - Artifacts reduce stability

## Best Practices

1. **Use moving average** - Smooth out micro-variations
2. **Set appropriate thresholds** - Based on content type
3. **Combine with motion magnitude** - For complete picture
4. **Consider context** - Idle in video ≠ idle in document
5. **Track patterns** - Not just individual values

## Common Misinterpretations

1. **High stability ≠ No activity** - Could be reading
2. **Low stability ≠ User interaction** - Could be video
3. **Medium stability ≠ Medium activity** - Could be various things
4. **Sudden drops ≠ Problems** - Normal for transitions

## Typical Output

```python
{
    'score': 0.92,  # Current frame stability
    'timeline': [0.9, 0.91, 0.89, 0.3, 0.35, ...],
    'average': 0.65,
    'stable_periods': [
        {'start': 0, 'end': 45, 'avg_stability': 0.9},
        {'start': 120, 'end': 200, 'avg_stability': 0.85}
    ]
}
```

## Key Insight

Stability score is the inverse of change - it measures "how much stayed the same" rather than "how much changed". This makes it excellent for detecting idle periods, reading time, and stable UI states in screen recordings.