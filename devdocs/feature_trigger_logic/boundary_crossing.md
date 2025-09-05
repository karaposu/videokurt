# Boundary Crossing Feature - Trigger Logic

## What This Feature Detects

The boundary crossing feature sets up virtual "lines" or boundaries in the video frame and detects when objects cross these lines. It tracks persistent objects that maintain their identity while moving across defined boundaries.

## Default Boundaries

1. **Custom boundaries**: User-defined horizontal/vertical lines at specific coordinates
2. **Frame edge boundaries**: Lines at all 4 edges (top, bottom, left, right) for entry/exit detection

## What TRIGGERS Detection in Screen Recordings

### 1. Dragging Elements Across Screen
- Dragging files from one side to another
- Moving windows across center lines
- Dragging images between locations
- Reordering list items by dragging
- Drag-and-drop operations between panels

### 2. Animated Transitions
- Slide transitions in presentations (slides entering from edges)
- Mobile app screen swipes (new screens sliding in)
- Carousel/slider animations (images sliding horizontally)
- Toast notifications sliding in from edges
- Page transitions with sliding effects

### 3. Video/Game Content
- Playing videos where objects move across screen
- Gaming footage with characters/enemies crossing boundaries
- Screen recordings of sports replay analysis
- Animated tutorials with moving arrows/pointers
- Embedded animations with moving elements

### 4. Floating Elements
- Cursor movements across boundary lines (if cursor is large/visible enough)
- Floating action buttons animating in/out
- Pop-ups sliding in from screen edges
- Dock/taskbar auto-hide animations
- Tooltips appearing/disappearing at edges

### 5. Progressive Loading
- Progress bars growing across boundaries
- Loading animations expanding across lines
- Charts/graphs animating from left to right
- Timeline scrubbers moving across screen
- Expanding menus crossing boundaries

## What DOESN'T Trigger Detection

### 1. Scrolling
- **Why not**: Entire viewport shifts together
- Text appears/disappears at edges instantaneously
- No trackable object maintains identity while crossing
- Content reflows rather than moves as objects

### 2. Static Content
- Idle periods with no movement
- Static UI elements that don't move
- Background changes without object movement

### 3. Instant Transitions
- Jump cuts between screens
- Instant page loads
- Teleporting elements
- Content that appears/disappears without traveling

## Key Requirements for Detection

For boundary crossings to be detected, you need:

1. **Persistent objects** - Elements that maintain identity while moving
2. **Continuous motion** - Objects must travel across boundaries, not appear/disappear
3. **Sufficient size** - Objects must be large enough to be detected as blobs
4. **Crossing motion** - Objects must actually cross defined boundaries, not just approach them

## Example Results Interpretation

```
No crossings detected = Static content or movement within zones
Few crossings = Occasional drag operations or transitions
Many crossings = Active dragging, animations, or game content
```

## Screen Recording vs Physical Video

| Aspect | Screen Recording | Physical Video |
|--------|-----------------|----------------|
| Common triggers | Dragged windows, transitions | People walking, vehicles |
| Detection quality | Poor (UI elements aren't objects) | Good (real objects) |
| Typical use case | Detecting drag operations | Counting people/vehicles |
| Scrolling | Not detected | N/A |

## Best Use Cases for Screen Recordings

1. **Drag operation detection** - Identifying when users drag elements
2. **Transition analysis** - Detecting sliding animations
3. **Game analysis** - Tracking game objects crossing zones
4. **Presentation analysis** - Detecting slide transitions

## Limitations

- Cannot distinguish between different types of objects
- Requires objects to maintain identity (not good for scrolling text)
- May miss small UI elements like cursors
- Better suited for physical world videos than UI analysis