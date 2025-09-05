# Edge Density Feature - Trigger Logic

## What This Feature Detects

Edge density measures the amount of visual edges/boundaries in each frame, indicating visual complexity and detail level.

## What CREATES High Edge Density

### 1. Text-Heavy Content
- Code editors with syntax highlighting
- Dense documentation
- Terminal/console output
- Spreadsheets with data
- Email lists
- Chat conversations

### 2. Detailed UI Elements
- Complex forms
- Data tables
- Tree views/file explorers
- Menu systems
- Toolbar collections
- Icon grids

### 3. Structured Layouts
- Grid layouts
- Card-based designs
- Tiled interfaces
- Dashboard widgets
- Multi-panel views
- Split screens

### 4. High-Detail Content
- Technical diagrams
- Architectural drawings
- Maps with streets/borders
- Graphs and charts
- CAD designs
- Circuit diagrams

### 5. Busy Interfaces
- IDEs with multiple panels
- Trading platforms
- Monitoring dashboards
- Audio/video editors
- 3D modeling software
- Database managers

## What CREATES Low Edge Density

### 1. Minimal Content
- Splash screens
- Login pages
- Empty states
- Loading screens
- Full-screen videos
- Solid backgrounds

### 2. Simple Layouts
- Single column text
- Centered content
- Hero sections
- Landing pages
- Presentation slides
- Minimalist designs

### 3. Smooth Content
- Gradient backgrounds
- Blurred images
- Solid color areas
- Whitespace-heavy design
- Abstract wallpapers
- Fog/overlay effects

### 4. Media Content
- Video playback
- Photo viewing
- Smooth animations
- Particle effects
- Soft shadows
- Gaussian blurs

## Edge Density Patterns

### By Application Type

| Application | Typical Density | Why |
|-------------|-----------------|-----|
| Code Editor | Very High | Syntax, line numbers, brackets |
| Web Browser | Medium-High | Text, borders, images |
| Video Player | Low | Smooth video content |
| Image Editor | Variable | Depends on image |
| Spreadsheet | High | Grid lines, data |
| Terminal | High | Text characters |
| Games | Variable | UI vs gameplay |

### By Content State

| State | Edge Density | Characteristics |
|-------|--------------|-----------------|
| Reading | High | Text creates many edges |
| Watching | Low | Video is smooth |
| Coding | Very High | Syntax highlighting |
| Browsing | Medium | Mixed content |
| Gaming | Variable | UI elements vs game world |

## Use Cases

1. **Content classification** - Text vs media detection
2. **Complexity analysis** - Simple vs complex UI
3. **Reading detection** - High edges suggest text
4. **Quality assessment** - Low quality = fewer edges
5. **Activity type** - Different tasks have signatures

## Edge Density Indicators

### High Density Indicates
- Text-based work
- Data analysis
- Coding/development
- Complex interfaces
- Detailed content

### Low Density Indicates
- Media consumption
- Minimal interfaces
- Loading/waiting states
- Artistic content
- Simple layouts

### Sudden Changes
- **Increase**: Transition to detailed content
- **Decrease**: Video starts, fullscreen mode
- **Fluctuation**: Mixed content browsing

## Relationship to Activity

| Edge Density | User Activity | Likely Scenario |
|--------------|---------------|-----------------|
| High + Static | Reading | Documentation, code review |
| High + Dynamic | Active work | Coding, data entry |
| Low + Static | Waiting | Loading, idle |
| Low + Dynamic | Watching | Video, animation |

## Advantages

1. **Content awareness** - Distinguishes text from media
2. **Complexity metric** - UI complexity measure
3. **Fast computation** - Simple edge detection
4. **Resolution independent** - Normalized metric
5. **Intuitive** - More edges = more detail

## Limitations

1. **No semantic meaning** - Edges don't reveal content
2. **Style dependent** - Font/theme affects values
3. **Noise sensitive** - Compression creates false edges
4. **Can't distinguish edge types** - Text vs borders
5. **Lighting affects results** - Shadows create edges

## Typical Output

```python
{
    'density': 0.72,  # Current frame (0-1)
    'timeline': [0.8, 0.75, 0.71, ...],  # Per frame
    'average': 0.68,
    'regions': {
        'high_density_areas': [...],  # Text regions
        'low_density_areas': [...]    # Media regions
    }
}
```

## Best Practices

1. **Normalize across sessions** - Different content scales
2. **Smooth timeline** - Reduce frame-to-frame noise
3. **Use thresholds carefully** - Content-dependent
4. **Combine with other features** - Context matters
5. **Consider display settings** - DPI affects edges

## Key Insight

Edge density serves as a "complexity thermometer" for screen content - high values indicate information-dense interfaces (code, data, text) while low values suggest media consumption or minimal interfaces. This makes it excellent for categorizing user activity types.