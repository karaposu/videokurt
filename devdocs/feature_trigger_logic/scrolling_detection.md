# Scrolling Detection Feature - Trigger Logic

## What This Feature Detects

Scrolling detection identifies vertical or horizontal scrolling patterns by analyzing consistent directional movement of content across frames.

## What TRIGGERS Detection

### 1. Vertical Scrolling
- Web page scrolling up/down
- Document scrolling in editors
- Social media feed browsing
- Chat message scrolling
- Code editor navigation
- PDF document reading
- Email list browsing
- File explorer navigation

### 2. Horizontal Scrolling
- Spreadsheet navigation
- Timeline scrubbing
- Image gallery swiping
- Code editor (long lines)
- Table/data grid scrolling
- Carousel/slider navigation
- Map panning
- Video timeline seeking

### 3. Smooth Scrolling
- Momentum/inertial scrolling
- Animated smooth scroll
- Autoscroll features
- Parallax scrolling effects
- Infinite scroll loading
- Smooth scroll to anchor

### 4. Discrete Scrolling
- Page up/down
- Mouse wheel clicks
- Arrow key navigation
- Space bar scrolling
- Jump to top/bottom

## What DOESN'T Trigger Detection

### 1. Static Content
- Reading without scrolling
- Video playback
- Static presentations
- Idle periods
- Form filling without scrolling

### 2. Other Movements
- Window dragging (not scrolling)
- Zoom in/out operations
- Rotation animations
- Fade transitions
- Pop-up appearances

### 3. Instant Jumps
- Page navigation (new page)
- Tab switches
- Search result jumps
- Bookmark navigation
- URL changes

## Detection Parameters

### Key Indicators
- **Consistent direction**: Uniform vertical/horizontal flow
- **Content coherence**: Text/elements moving together
- **Edge behavior**: New content entering, old exiting
- **Velocity patterns**: Consistent or decelerating speed

### Scrolling Types
1. **Continuous**: Smooth, uninterrupted movement
2. **Stepped**: Discrete jumps (line-by-line)
3. **Accelerated**: Speed changes (inertial)
4. **Bidirectional**: Up/down in same session

## Example Results Interpretation

```
High vertical flow = Active reading/browsing
High horizontal flow = Spreadsheet/timeline work
Alternating direction = Searching/reviewing
No scrolling = Static viewing or editing
Mixed patterns = Complex navigation
```

## Common Patterns by Application

### Web Browsers
- Vertical dominates (reading articles)
- Smooth with momentum
- Periodic pauses (reading)
- Quick returns to top

### Code Editors
- Both vertical and horizontal
- Jump scrolling common
- Search-driven jumps
- Split-pane scrolling

### Social Media
- Continuous vertical
- Infinite scroll patterns
- Brief pauses on posts
- Quick scroll past ads

### Documents
- Page-by-page common
- Smooth during review
- Jump to sections
- Find/search jumps

### Spreadsheets
- Horizontal for columns
- Vertical for rows
- Cell-to-cell jumps
- Freeze pane effects

## Scrolling Metrics

| Metric | Meaning |
|--------|---------|
| Speed | Pixels per frame moved |
| Direction | Up/down/left/right |
| Consistency | How uniform the movement |
| Duration | Continuous scroll time |
| Coverage | Portion of content scrolled |

## Best Use Cases

1. **Reading behavior** - How users consume content
2. **Navigation patterns** - How users explore interfaces
3. **Content engagement** - What gets scrolled past vs read
4. **Usability issues** - Excessive scrolling needs
5. **Performance** - Scroll smoothness/jank detection

## Advantages Over Other Features

- Purpose-built for UI interaction
- Understands viewport concept
- Distinguishes from other motion types
- Provides semantic meaning
- Works well with screen recordings

## Typical Output

```python
{
    'is_scrolling': True,
    'direction': 'vertical',
    'confidence': 0.85,
    'velocity': 120,  # pixels/frame
    'smooth_score': 0.9,  # smoothness
    'scroll_events': [
        {'start': 30, 'end': 95, 'direction': 'down'},
        {'start': 150, 'end': 180, 'direction': 'up'}
    ]
}
```

## Key Advantages

This is one of the most useful features for screen recordings because:
- Specifically designed for UI patterns
- High accuracy for common user action
- Provides actionable insights
- Works across all applications
- Clear interpretation