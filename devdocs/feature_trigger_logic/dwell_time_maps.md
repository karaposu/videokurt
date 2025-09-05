# Dwell Time Maps Feature - Trigger Logic

## What This Feature Detects

Dwell time maps create heatmaps showing how long content remains unchanged at each pixel location, revealing where UI elements are static vs dynamic.

## What CREATES High Dwell Time (Red/Hot Areas)

### 1. Static UI Elements
- Menu bars
- Navigation headers
- Sidebars
- Status bars
- Tool palettes
- Fixed footers

### 2. Persistent Content
- Logo/branding areas
- Copyright notices
- Page titles
- Breadcrumbs
- Tab bars
- Window frames

### 3. Stable Regions
- Document margins
- Reading panes
- Form labels
- Static images
- Background areas
- Decorative elements

### 4. Inactive Areas
- Unused screen space
- Disabled controls
- Empty panels
- Padding areas
- Fixed layouts
- Modal backgrounds

## What CREATES Low Dwell Time (Blue/Cold Areas)

### 1. Active Content Areas
- Main content region
- Scrolling areas
- Video players
- Animation zones
- Live data displays
- Chat windows

### 2. Interactive Elements
- Text input fields
- Dropdown menus
- Hover zones
- Button areas
- Sliders/controls
- Canvas/drawing areas

### 3. Dynamic Regions
- Notification areas
- Progress indicators
- Status updates
- Live feeds
- Ticker displays
- Loading zones

### 4. User Activity Zones
- Mouse trail areas
- Frequently clicked regions
- Drag paths
- Gesture areas
- Selection zones
- Cursor hot spots

## Common Dwell Patterns

### Web Applications
```
High: Header, sidebar, footer
Low: Central content area
Pattern: Frame stable, content dynamic
```

### Code Editors
```
High: Line numbers, file tree, tabs
Low: Code editing area
Pattern: UI stable, code changes
```

### Video Players
```
High: Player controls, title
Low: Video area
Pattern: Controls static, video dynamic
```

### Documents
```
High: Toolbar, margins
Low: Text area during scrolling
Pattern: Chrome stable, content moves
```

### Dashboards
```
High: Layout structure
Low: Data visualization areas
Pattern: Structure stable, data updates
```

## Heatmap Interpretation

| Color | Dwell Time | Meaning |
|-------|------------|---------|
| Dark Red | Very High | Never changes |
| Red | High | Rarely changes |
| Orange | Medium-High | Occasional changes |
| Yellow | Medium | Moderate activity |
| Green | Medium-Low | Frequent changes |
| Blue | Low | Constant change |
| Dark Blue | Very Low | Always changing |

## Use Cases

1. **UI stability analysis** - Which parts are static
2. **Attention mapping** - Where users focus activity
3. **Layout optimization** - Finding dead zones
4. **Performance analysis** - Identifying update regions
5. **Design validation** - Confirming stable elements

## Pattern Recognition

### Healthy Patterns
- Clear separation between stable UI and content
- Consistent navigation areas
- Predictable active zones
- Logical stability distribution

### Problem Patterns
- Entire screen low dwell (too much change)
- Entire screen high dwell (too static)
- Unexpected hot spots (stuck elements)
- Fragmented patterns (poor layout)

## Advantages

1. **Visual summary** - Instant understanding
2. **Spatial analysis** - Location-based insights
3. **Cumulative view** - Long-term patterns
4. **Design feedback** - UI effectiveness
5. **Problem detection** - Stuck or dead areas

## Limitations

1. **No temporal info** - When changes occurred
2. **No change type** - What kind of changes
3. **Resolution dependent** - Pixel-level accuracy
4. **Memory intensive** - Tracks every pixel
5. **Noise accumulation** - Small changes add up

## Relationship to Other Features

| Feature | Dwell Time Maps | Comparison |
|---------|-----------------|------------|
| Stability Score | Per-pixel stability | Frame-level stability |
| Zone Activity | Heatmap visualization | Numerical zones |
| Motion Magnitude | Spatial distribution | Single value |
| Spatial Occupancy | Where activity occurs | How long static |

## Typical Output

```python
{
    'dwell_map': np.array([[...]]),  # 2D heatmap
    'max_dwell': 500,  # Frames
    'min_dwell': 0,
    'avg_dwell': 125,
    'static_regions': [
        {'area': (0, 0, 1920, 80), 'dwell': 450}  # Header
    ],
    'dynamic_regions': [
        {'area': (200, 100, 1520, 900), 'dwell': 20}  # Content
    ]
}
```

## Visualization Tips

1. **Use color gradients** - Red=stable, Blue=dynamic
2. **Apply smoothing** - Reduce pixel noise
3. **Set thresholds** - Focus on meaningful differences
4. **Overlay on screenshot** - Context for regions
5. **Create time slices** - Evolution of patterns

## Key Insight

Dwell time maps reveal the "skeleton" of user interfaces - the stable framework versus dynamic content areas. This is invaluable for understanding UI design effectiveness and user interaction patterns in screen recordings.