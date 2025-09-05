# Repetition Indicator Feature - Trigger Logic

## What This Feature Detects

Repetition indicator identifies patterns that repeat over time, detecting cyclic or periodic behavior in screen recordings.

## What TRIGGERS Repetition Detection

### 1. Loading Animations
- Spinning progress indicators
- Loading bar cycles
- Pulsing buttons
- Bouncing dots
- Skeleton screen animations
- Throbbing effects

### 2. UI Animations
- Blinking cursors
- Breathing/pulse effects
- Notification badges pulsing
- Hover effect loops
- Banner rotations
- Carousel auto-play

### 3. Status Indicators
- Network activity icons
- Recording indicators
- Live badges flashing
- Alert beacons
- Warning blinks
- Status LED simulations

### 4. Media Loops
- GIF animations
- Video loops
- Audio visualizers
- Screen savers
- Demo reels
- Background animations

### 5. User Patterns
- Repetitive scrolling up/down
- Tab cycling (Alt+Tab repeatedly)
- Undo/redo sequences
- Copy-paste patterns
- Repeated clicks
- Gesture repetitions

### 6. Automated Actions
- Auto-refresh cycles
- Polling updates
- Slideshow advances
- Test automation runs
- Macro executions
- Script loops

## What DOESN'T Trigger Detection

### 1. One-Time Events
- Single page loads
- Individual clicks
- One-way scrolling
- Linear progressions
- Unique transitions

### 2. Random Changes
- Live video streams
- Random animations
- Noise/static
- Unpredictable updates
- Chaotic motion

### 3. Gradual Changes
- Slow fades
- Progressive loading
- Continuous scrolling
- Smooth transitions
- Non-repeating animations

## Repetition Characteristics

### Pattern Types
1. **Exact repetition** - Identical frames repeat
2. **Similar repetition** - Nearly identical with small variations
3. **Periodic motion** - Regular movement patterns
4. **Cyclic behavior** - Actions that loop back

### Time Scales
| Period | Typical Cause |
|--------|--------------|
| 0.5-1 sec | Cursor blink, pulse effects |
| 1-2 sec | Loading spinners |
| 2-5 sec | Progress indicators |
| 5-10 sec | Auto-refresh, slideshows |
| 10+ sec | Demo loops, screen savers |

## Common Repetitive Patterns

### Development
- Build/compile progress
- Test suite runs
- Linting indicators
- Hot reload cycles
- Debugger stepping
- Log tailing

### Communication
- Typing indicators
- Message status updates
- Call connecting animations
- Screen share indicators
- Presence updates
- Notification pulses

### Browsing
- Auto-playing videos
- Ad rotations
- Infinite scroll loading
- Ajax spinners
- Live data updates
- Social media refreshes

### Gaming
- Idle animations
- UI element pulses
- Countdown timers
- Respawn cycles
- Menu backgrounds
- Achievement notifications

## Detection Parameters

### Key Metrics
- **Period length** - Frames between repetitions
- **Similarity threshold** - How close patterns must be
- **Minimum cycles** - Required repetitions to confirm
- **Confidence score** - Strength of repetition

## Use Cases

1. **Loading detection** - Finding waiting times
2. **Animation analysis** - Identifying UI animations
3. **Automation detection** - Finding scripted actions
4. **Performance issues** - Detecting stuck processes
5. **User behavior** - Identifying repetitive tasks

## Interpretation

```
High repetition + Short period = Active loading/animation
High repetition + Long period = Slideshow/demo
Low repetition + Regular = Periodic updates
Variable repetition = User-driven patterns
```

## Advantages

1. **Pattern recognition** - Finds hidden cycles
2. **Automation detection** - Identifies scripts/macros
3. **Performance indicator** - Stuck/looping detection
4. **UI analysis** - Animation inventory
5. **Behavior patterns** - User habit detection

## Limitations

1. **False positives** - Similar but unrelated frames
2. **Period detection** - May miss irregular cycles
3. **Partial repetitions** - Incomplete cycles ignored
4. **Noise sensitivity** - Small changes break detection
5. **Memory intensive** - Must compare many frames

## Typical Output

```python
{
    'repetition_score': 0.75,  # 0-1 score
    'dominant_period': 45,  # Frames
    'repetition_count': 8,  # Number of cycles
    'confidence': 0.82,
    'pattern_type': 'periodic',
    'repetitive_regions': [...],  # Areas with repetition
    'timeline': [...]  # Repetition strength over time
}
```

## Best Practices

1. **Set appropriate thresholds** for similarity
2. **Consider multiple periods** - May have nested cycles
3. **Filter out cursor blinks** - Too common
4. **Look for meaningful periods** - Not just any repetition
5. **Combine with other features** - Context matters

## Key Insight

Repetition indicator excels at finding "waiting states" and "automated behaviors" in screen recordings - from loading animations to macro executions. It reveals both intentional patterns (animations) and unintentional ones (user habits, stuck processes).