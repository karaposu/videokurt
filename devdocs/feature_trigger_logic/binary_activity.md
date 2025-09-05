# Binary Activity Feature - Trigger Logic

## What This Feature Detects

Binary activity provides a simple yes/no answer for whether significant activity is occurring in each frame, converting complex motion into a binary signal.

## What TRIGGERS Activity (1/True)

### 1. Any Visible Movement
- Scrolling (any speed)
- Window movements
- Cursor motion (if significant)
- Typing appearing on screen
- Animation playing
- Video content

### 2. UI Changes
- Page loads
- Menu opens/closes
- Dialog appears
- Tab switches
- Notifications
- Popups

### 3. Content Updates
- Text changes
- Image loads
- Data refreshes
- Progress updates
- Status changes
- Live content

### 4. Transitions
- Fade effects
- Slide animations
- Screen switches
- App launches
- Window focus changes

## What TRIGGERS Inactivity (0/False)

### 1. Complete Stillness
- Reading without scrolling
- Paused video
- Idle desktop
- Static image viewing
- Waiting screens

### 2. Minimal Changes
- Clock tick only
- Cursor blink only
- Single pixel changes
- Sub-threshold movement
- Compression noise only

## Binary Activity Uses

### Simple State Detection
```python
if binary_activity == 1:
    # User is doing something
else:
    # User is idle/reading
```

### Activity Timeline
```
[0,0,0,1,1,1,0,0,0,1,1,0,0]
 Idle  |Active|Idle |Act|Idle
```

### Common Patterns

| Pattern | Meaning |
|---------|---------|
| All 0s | Completely idle session |
| All 1s | Continuous activity |
| 0→1→0 | Brief interaction |
| Alternating | Start-stop behavior |
| Mostly 1s | Active session |
| Mostly 0s | Reading/watching |

## Threshold Effects

### Low Threshold
- More sensitive
- Detects subtle changes
- More false positives
- Good for: Ensuring no activity missed

### High Threshold  
- Less sensitive
- Only major changes
- More false negatives
- Good for: Filtering out noise

## Use Cases

1. **Idle detection** - Simple presence detection
2. **Activity summary** - Percentage active time
3. **Segmentation** - Active vs passive periods
4. **Attention tracking** - When user engaged
5. **Compression** - Reduce data to binary

## Relationship to Other Features

| Feature | Binary Activity | Comparison |
|---------|-----------------|------------|
| Motion Magnitude | Binary version | Simplified |
| Stability Score | Inverse relationship | Opposite |
| Frame Diff | Thresholded version | Binary |

## Advantages

1. **Simplicity** - Just yes/no
2. **Low storage** - 1 bit per frame
3. **Fast processing** - Simple threshold
4. **Clear interpretation** - Active or not
5. **Easy visualization** - Timeline bars

## Limitations

1. **Loss of detail** - No intensity information
2. **Threshold dependent** - Results vary
3. **No direction** - Just presence
4. **No semantics** - What kind of activity
5. **Binary only** - No gradients

## Typical Output

```python
{
    'activity': 1,  # Current frame
    'timeline': [0, 0, 1, 1, 1, 0, ...],  # Binary array
    'activity_ratio': 0.65,  # 65% active
    'active_periods': [
        {'start': 10, 'end': 35},
        {'start': 67, 'end': 89}
    ]
}
```

## Key Insight

Binary activity is the simplest possible activity metric - perfect for when you just need to know "is something happening?" without caring about what or how much. It's the foundation for more complex activity analysis.