# Blob Stability Feature - Trigger Logic

## What This Feature Detects

Blob stability measures how long detected blobs persist across frames. It identifies which moving regions remain trackable over time versus those that appear briefly.

## What TRIGGERS High Stability in Screen Recordings

### 1. Persistent UI Elements
- Floating toolbars that remain visible
- Picture-in-picture windows
- Persistent notification badges
- Pinned overlays or widgets
- Status bars with animated elements

### 2. Long-Running Animations
- Loading spinners that run continuously
- Progress bars during long operations
- Animated backgrounds or wallpapers
- Screensavers with moving elements
- Continuous particle effects

### 3. Stable Media Players
- Video player controls during playback
- Audio visualizers
- Embedded video content
- GIF loops playing repeatedly
- Live stream overlays

### 4. Drawing/Annotation Sessions
- Continuous pen strokes
- Shape drawing operations
- Screen annotation tools in use
- Whiteboard drawing sessions
- Signature capture

## What TRIGGERS Low Stability

### 1. UI Flicker
- Hover effects appearing/disappearing
- Tooltips popping up briefly
- Menu items highlighting
- Button press animations
- Focus indicators

### 2. Text Operations
- Cursor blinking
- Text selection/deselection
- Typing creating momentary changes
- Autocomplete dropdowns
- Search suggestions

### 3. Rapid Transitions
- Page loads
- Tab switches
- Screen refreshes
- Window switching
- App transitions

## Stability Metrics

### High Stability Indicators
- Persistence > 20 frames
- Consistent size across frames
- Smooth trajectory
- Regular appearance pattern

### Low Stability Indicators
- Persistence < 5 frames
- Rapidly changing size
- Erratic position changes
- Intermittent appearance

## Example Results Interpretation

```
High stable blob ratio = Persistent UI elements or media
Low stable blob ratio = Dynamic UI with frequent changes
Mixed stability = Normal application usage
Zero stable blobs = Highly dynamic or static content
```

## Screen Recording Patterns

### Typical Patterns
1. **Static UI**: Few blobs, those detected are stable
2. **Active browsing**: Many unstable blobs from page changes
3. **Video watching**: One large stable blob (video area)
4. **Gaming**: Multiple stable blobs (game elements)
5. **Typing/coding**: Mostly unstable blobs from text changes

## Stability Score Interpretation

| Score | Meaning in Screen Recording |
|-------|----------------------------|
| 0-5 frames | UI noise, hover effects, artifacts |
| 5-20 frames | Brief animations, transitions |
| 20-60 frames | Persistent UI elements |
| 60+ frames | Static overlays, continuous media |

## False Positives

Common sources of incorrectly "stable" blobs:
- Compression artifacts in same location
- Static UI elements detected as moving
- Screen tearing creating persistent edges
- Dead pixels or screen defects
- Watermarks or logos

## Best Use Cases for Screen Recordings

1. **Media detection** - Identifying video/animation regions
2. **Overlay analysis** - Finding persistent UI overlays
3. **Tool usage** - Detecting continuous tool usage
4. **Performance issues** - Identifying stuck or frozen elements

## Limitations

- Cannot identify what stable blobs represent
- Confuses static UI elements with meaningful persistence
- High false positive rate from rendering artifacts
- Depends heavily on threshold settings
- Not useful for scrolling or text-heavy content

## Typical Output

```python
{
    'persistence_scores': [0, 0, 5, 0, 7, 0, ...],  # Mostly low
    'stable_blobs': [...],  # Often UI artifacts
    'stability_timeline': [...],  # Highly variable
    'gap_statistics': {...}  # Many gaps from flickering
}
```