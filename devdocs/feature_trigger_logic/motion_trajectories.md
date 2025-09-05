# Motion Trajectories Feature - Trigger Logic

## What This Feature Detects

Motion trajectories tracks the paths of moving elements by following optical flow vectors over time, revealing movement patterns and directions.

## What CREATES Trajectories in Screen Recordings

### 1. Dragging Operations
- Window dragging paths
- File drag-and-drop arcs
- Selection rectangles
- Drawing/painting strokes
- Slider movements
- Map panning paths

### 2. Animated Elements
- Loading spinner rotations
- Progress bar movements
- Animated icon paths
- Transition effects
- Particle animations
- Floating elements

### 3. Scrolling Patterns
- Content flow lines during scroll
- Parallax layer movements
- Carousel sliding paths
- List reordering animations
- Smooth scroll trajectories

### 4. Cursor Movements
- Mouse paths (if visible/large)
- Touch gesture traces
- Pen/stylus strokes
- Drag preview paths
- Annotation drawings

### 5. UI Animations
- Menu slide-outs
- Panel transitions
- Tooltip movements
- Notification slides
- Modal appearances
- Drawer animations

## Trajectory Characteristics

### Path Types
1. **Linear** - Straight drag operations
2. **Curved** - Natural mouse movements
3. **Circular** - Rotational animations
4. **Zigzag** - Back-and-forth scrolling
5. **Converging** - Elements moving to point
6. **Diverging** - Elements spreading out

### Trajectory Quality
| Quality | Characteristics | Indicates |
|---------|----------------|-----------|
| Smooth | Continuous path | Intentional movement |
| Jerky | Broken segments | Lag or stuttering |
| Dense | Many trajectories | High activity |
| Sparse | Few trajectories | Limited movement |
| Parallel | Similar directions | Scrolling/panning |
| Chaotic | Random directions | Noise or transitions |

## Common Trajectory Patterns

### Scrolling
```
Parallel vertical lines
Direction: Uniform up/down
Length: Full frame height
Pattern: Content flow
```

### Window Dragging
```
Single curved path
Direction: User-defined
Length: Variable
Pattern: Start → End position
```

### Loading Animation
```
Circular trajectories
Direction: Rotating
Length: Continuous loops
Pattern: Repeated cycles
```

### Text Selection
```
Horizontal lines
Direction: Left to right
Length: Text width
Pattern: Line by line
```

## Use Cases

1. **Interaction analysis** - How users manipulate UI
2. **Animation detection** - Finding moving elements
3. **Scroll pattern analysis** - Scroll speed/direction
4. **Gesture recognition** - Touch/mouse patterns
5. **Performance issues** - Detecting stuttering

## Trajectory Metrics

| Metric | Meaning | Use Case |
|--------|---------|----------|
| Path length | Distance traveled | Movement amount |
| Direction | Angle/vector | Movement type |
| Velocity | Speed along path | Interaction speed |
| Curvature | Path smoothness | Natural vs mechanical |
| Density | Trajectories per area | Activity level |

## Advantages Over Basic Motion

1. **Path information** - Not just presence but route
2. **Direction tracking** - Where things move
3. **Pattern recognition** - Identifies gestures
4. **Temporal continuity** - Links movement over time
5. **Velocity data** - Speed and acceleration

## Limitations

1. **Fragmentation** - Paths break easily
2. **Occlusion** - Hidden elements lose tracking
3. **Noise** - Many false trajectories
4. **Short paths** - Brief movements missed
5. **No object identity** - Just anonymous paths

## Relationship to Other Features

| Feature | Motion Trajectories | Difference |
|---------|-------------------|------------|
| Optical Flow | Builds on flow | Temporal integration |
| Blob Tracking | Anonymous paths | No object identity |
| Motion Magnitude | Path details | Not just amount |
| Scrolling Detection | General paths | Not scroll-specific |

## Typical Output

```python
{
    'trajectories': [
        {
            'points': [(x1,y1), (x2,y2), ...],
            'length': 145.2,
            'direction': 'northeast',
            'velocity': 12.3,
            'frames': [10, 11, 12, ...]
        }
    ],
    'num_trajectories': 23,
    'avg_length': 67.8,
    'dominant_direction': 'vertical',
    'trajectory_density': 0.34
}
```

## Best Practices

1. **Filter short trajectories** - Remove noise
2. **Smooth paths** - Reduce jitter
3. **Cluster similar paths** - Group related movement
4. **Set minimum length** - Ignore brief motion
5. **Visualize on frame** - Overlay for context

## Key Insight

Motion trajectories reveal the "gestures" of screen interaction - from deliberate drags to automated animations. Unlike simple motion detection, trajectories show not just that something moved, but how it moved through space over time.