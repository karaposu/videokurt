# Blob Tracking Feature - Trigger Logic

## What This Feature Detects

Blob tracking identifies and tracks connected regions of moving pixels (blobs) across frames. It uses background subtraction to find foreground objects and tracks them over time.

## What TRIGGERS Detection in Screen Recordings

### 1. Window Operations
- Windows being dragged across screen
- Pop-up windows appearing/moving
- Dialog boxes sliding into view
- Floating panels being repositioned
- Picture-in-picture video windows moving

### 2. Large UI Animations
- Loading spinners (if large enough)
- Animated logos or graphics
- Progress indicators with moving parts
- Animated buttons or controls
- Slideshow transitions with moving elements

### 3. Cursor/Pointer (if visible enough)
- Large custom cursors
- Cursor with drag preview attached
- Screen annotation tools
- Laser pointer in presentations
- Drawing/painting brush strokes

### 4. Media Content
- Video players with moving content
- GIF animations in web pages
- Game elements moving on screen
- Animated advertisements
- Screen recordings within recordings

### 5. Notification Elements
- Toast notifications sliding in
- Banner alerts moving across screen
- Floating badges or counters
- System notifications with animation
- Chat bubbles appearing/floating

## What DOESN'T Trigger Detection

### 1. Text Changes
- Typing in text fields (too small/scattered)
- Scrolling text (appears/disappears instantly)
- Text selection highlighting
- Blinking cursor (too small)

### 2. Static UI Updates
- Color changes without movement
- Instant page loads
- Static button presses
- Menu appearances without animation

### 3. Subtle Changes
- Hover effects
- Small icon changes
- Thin border highlights
- Minor pixel shifts

## Detection Characteristics

### Blob Properties
- **Size**: Must exceed minimum area threshold (default 100px)
- **Persistence**: Must appear in multiple consecutive frames
- **Connectivity**: Pixels must form connected region
- **Movement**: Must show displacement between frames

### Common False Positives in Screen Recordings
- Compression artifacts creating "phantom" blobs
- Anti-aliasing changes detected as movement
- Screen refresh patterns
- Video encoding noise

## Example Results Interpretation

```
Many short-lived blobs = UI with lots of small animations or noise
Few persistent blobs = Stable windows or media players
Trajectories present = Elements actually moving across screen
No blobs = Static content or movement below threshold
```

## Screen Recording vs Physical Video

| Aspect | Screen Recording | Physical Video |
|--------|-----------------|----------------|
| Typical blob count | High (many UI elements) | Low-moderate |
| Blob persistence | Very short (frames) | Longer (seconds) |
| Trajectories | Erratic/short | Smooth/continuous |
| Noise level | High | Low |
| Usefulness | Limited | High |

## Best Use Cases for Screen Recordings

1. **Window tracking** - Following dialog boxes or floating windows
2. **Media analysis** - Detecting video players or animations
3. **Cursor analysis** - Tracking large cursors or drawing tools
4. **Game analysis** - Following game sprites or characters

## Limitations

- Cannot distinguish between UI elements and actual objects
- Generates many false positives from UI rendering
- Loses track when objects stop moving
- Poor at handling scrolling or text
- Treats every moving pixel group as an "object"

## Typical Output

```python
{
    'counts': [45, 52, 38, ...],  # Blobs per frame (often noisy)
    'sizes': [[100, 245], ...],   # Blob sizes (highly variable)
    'trajectories': [...],         # Usually short, fragmented paths
    'centroids': [...]             # Positions (scattered across UI)
}
```