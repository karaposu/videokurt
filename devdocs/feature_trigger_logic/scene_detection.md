# Scene Detection Feature - Trigger Logic

## What This Feature Detects

Scene detection identifies significant visual changes that represent transitions between different scenes, screens, or contexts.

## What TRIGGERS Scene Changes in Screen Recordings

### 1. Application Switches
- Alt-tab between programs
- Clicking taskbar items
- Opening new applications
- Closing applications
- Switching virtual desktops

### 2. Page Navigation
- Clicking links (new page)
- Browser back/forward
- Bookmark navigation
- URL bar navigation
- Opening new tabs

### 3. Major UI Transitions
- Fullscreen mode toggle
- Login/logout screens
- Splash screens
- Loading screens
- Error screens

### 4. Content Switches
- Video chapter changes
- Presentation slide changes
- Document page breaks
- Image gallery navigation
- Tab panel switches

### 5. Window Operations
- Maximizing/minimizing windows
- Opening/closing dialogs
- Modal popups appearing
- Screen resolution changes
- Display mode changes

## What DOESN'T Trigger Scene Detection

### 1. Continuous Actions
- Scrolling (same content)
- Typing in same field
- Gradual animations
- Smooth transitions
- Video playback (same video)

### 2. Minor UI Changes
- Hover effects
- Small popups
- Status updates
- Progress bars
- Notification toasts

### 3. Incremental Updates
- Live data updating
- Chat messages arriving
- Progressive loading
- Partial refreshes
- Ajax updates

## Detection Thresholds

| Change Level | Similarity | Triggers Detection | Example |
|--------------|------------|-------------------|---------|
| Major | < 0.3 | Yes | New application |
| Significant | 0.3-0.5 | Yes | New page |
| Moderate | 0.5-0.7 | Maybe | Large modal |
| Minor | 0.7-0.9 | No | Scrolling |
| Minimal | > 0.9 | No | Cursor movement |

## Common Scene Boundaries

### Web Browsing
- Site to site navigation
- Major page changes
- Full page refreshes
- Tab switches
- Browser to other app

### Document Work
- Document opens/closes
- Major section changes
- View mode changes
- Different documents
- Print preview

### Communication
- Chat window opens
- Video call starts/ends
- Screen share begins
- Different conversations
- Email to calendar

### Development
- IDE to browser
- Terminal to editor
- Debugger activation
- Different projects
- Documentation lookup

## Scene Types in Screen Recordings

1. **Application scenes** - Different programs
2. **Content scenes** - Different documents/pages
3. **Modal scenes** - Dialogs, popups
4. **Transition scenes** - Loading, splash screens
5. **Error scenes** - Error messages, crashes

## Detection Methods

### Visual Similarity
- Pixel-wise comparison
- Histogram matching
- Structural similarity
- Color distribution
- Layout changes

### Temporal Patterns
- Sudden changes
- Fade transitions
- Cut transitions
- Dissolve effects
- Swipe animations

## Use Cases

1. **Workflow segmentation** - Breaking work into tasks
2. **Navigation analysis** - Understanding user journey
3. **Context switches** - Measuring multitasking
4. **Error detection** - Finding when errors occur
5. **Content summarization** - Creating video chapters

## Advantages

1. **Context awareness** - Understands major changes
2. **Application agnostic** - Works across all software
3. **Automated segmentation** - No manual marking needed
4. **Meaningful boundaries** - Aligns with user perception
5. **Efficient summarization** - Reduces video to key frames

## Limitations

1. **Threshold sensitivity** - May miss or over-detect
2. **No semantic understanding** - Can't identify what the scene is
3. **Gradual transitions** - May not detect slow changes
4. **False positives** - Large popups might trigger
5. **Animation confusion** - Full-screen animations detected as scenes

## Typical Output

```python
{
    'scene_boundaries': [0, 145, 389, 567, 821],  # Frame numbers
    'num_scenes': 5,
    'scene_durations': [145, 244, 178, 254],  # Frames per scene
    'confidence_scores': [0.95, 0.88, 0.92, 0.87],
    'scene_thumbnails': [...],  # Representative frames
    'similarity_scores': [0.12, 0.23, 0.18, 0.15]  # Change magnitude
}
```

## Best Practices

1. **Adjust thresholds** based on content type
2. **Filter short scenes** - Likely false positives
3. **Combine with stability** - Confirm scene boundaries
4. **Use for indexing** - Create video chapters
5. **Validate key frames** - Ensure meaningful scenes

## Key Insight

Scene detection in screen recordings effectively identifies "context switches" - when the user moves between different tasks, applications, or content. This is invaluable for understanding workflow and creating navigable video segments.