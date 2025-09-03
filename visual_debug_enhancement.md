# Visual Debug Enhancement for VideoKurt

## Overview

When extracting features (scene detection, scroll detection, binary activity, etc.), we need to visualize these on the original video for debugging and verification purposes. This document outlines the interface design for saving videos with visual debugging overlays.

## Core Requirements

1. **Overlay extracted features on original video**
2. **Support multiple visualization types** (text, markers, graphs, heatmaps)
3. **Configurable visualization styles**
4. **Performance-conscious** (shouldn't slow down analysis significantly)
5. **Easy to use** for common cases, flexible for advanced needs

## Proposed Interface Design

### Basic Usage - Simple Text Overlay

```python
from videokurt import VideoKurt

vk = VideoKurt()
vk.add_analysis('frame_diff')
vk.add_feature('scene_detection')
vk.add_feature('binary_activity')
vk.configure(frame_step=2, resolution_scale=0.5)

# Analyze video
results = vk.analyze('input_video.mp4')

# Save with debug visualization
vk.save_debug_video(
    input_path='input_video.mp4',
    output_path='debug_output.mp4',
    results=results,
    features=['scene_detection', 'binary_activity'],  # Which features to show
    style='text'  # Simple text overlay
)
```

### Advanced Usage - Custom Visualization

```python
# More control over visualization
vk.save_debug_video(
    input_path='input_video.mp4',
    output_path='debug_output.mp4',
    results=results,
    visualizations=[
        {
            'type': 'text',
            'feature': 'scene_detection',
            'position': 'top-left',
            'color': (0, 255, 0),
            'font_scale': 0.7
        },
        {
            'type': 'binary_bar',
            'feature': 'binary_activity',
            'position': 'bottom',
            'height': 30,
            'active_color': (0, 255, 0),
            'inactive_color': (128, 128, 128)
        },
        {
            'type': 'graph',
            'data': results.analyses['frame_diff'].data['pixel_diff'],
            'position': 'top-right',
            'size': (200, 100),
            'color': (255, 255, 0)
        }
    ]
)
```

## Visualization Types

### 1. Text Overlays
Display feature values and states as text on video.

```python
{
    'type': 'text',
    'feature': 'scene_detection',  # or custom text function
    'position': 'top-left',  # or (x, y) coordinates
    'color': (B, G, R),
    'font_scale': 0.5,
    'font_thickness': 1,
    'background_color': (0, 0, 0),  # Optional background box
    'background_alpha': 0.5
}
```

**Position options:**
- 'top-left', 'top-center', 'top-right'
- 'center-left', 'center', 'center-right'
- 'bottom-left', 'bottom-center', 'bottom-right'
- (x, y) tuple for exact positioning

### 2. Binary Activity Bar
Show binary states (active/inactive) as a horizontal bar.

```python
{
    'type': 'binary_bar',
    'feature': 'binary_activity',
    'position': 'bottom',  # or 'top'
    'height': 30,  # pixels
    'active_color': (0, 255, 0),
    'inactive_color': (128, 128, 128),
    'border': True,
    'label': 'Activity'
}
```

### 3. Timeline Graph
Display time-series data as a line graph overlay.

```python
{
    'type': 'graph',
    'data': results.analyses['frame_diff'].data['pixel_diff'],
    'position': 'top-right',
    'size': (200, 100),  # width, height in pixels
    'color': (255, 255, 0),
    'line_thickness': 2,
    'fill': True,  # Fill area under line
    'fill_alpha': 0.3,
    'y_range': 'auto',  # or (min, max)
    'label': 'Frame Diff',
    'grid': True
}
```

### 4. Heatmap Overlay
Overlay heatmap data (e.g., motion heatmap) on the video.

```python
{
    'type': 'heatmap',
    'data': results.analyses['motion_heatmap'].data['cumulative'],
    'alpha': 0.5,  # Transparency of overlay
    'colormap': 'jet',  # OpenCV colormap
    'normalize': True
}
```

### 5. Bounding Boxes
Draw boxes for detected regions (e.g., contours, motion areas).

```python
{
    'type': 'boxes',
    'feature': 'contour_detection',  # or custom box provider
    'color': (0, 255, 0),
    'thickness': 2,
    'label_format': 'area: {area}',  # Optional labels
    'min_area': 100  # Filter small boxes
}
```

### 6. Markers and Points
Show tracked points or markers.

```python
{
    'type': 'markers',
    'feature': 'optical_flow_sparse',  # or custom point provider
    'marker_type': 'circle',  # 'circle', 'cross', 'square'
    'size': 5,
    'color': (255, 0, 0),
    'tracks': True  # Show point trails
}
```

### 7. Side-by-Side Comparison
Show original and processed frames side by side.

```python
{
    'type': 'split_view',
    'mode': 'horizontal',  # or 'vertical', 'grid'
    'views': [
        {'source': 'original', 'label': 'Original'},
        {'source': 'frame_diff', 'label': 'Motion'},
        {'source': 'edge_canny', 'label': 'Edges'}
    ]
}
```

### 8. Status Dashboard
Comprehensive status panel with multiple metrics.

```python
{
    'type': 'dashboard',
    'position': 'top',
    'height': 100,
    'metrics': [
        {'label': 'Scene', 'feature': 'scene_detection'},
        {'label': 'Activity', 'feature': 'binary_activity'},
        {'label': 'Motion', 'value': lambda f: f['motion_level']},
        {'label': 'Frame', 'value': lambda f: f['frame_number']}
    ],
    'style': 'modern',  # or 'minimal', 'detailed'
    'background_alpha': 0.7
}
```

## Implementation Details

### Main Method: save_debug_video

```python
def save_debug_video(
    self,
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    results: RawAnalysisResults,
    features: List[str] = None,
    visualizations: List[dict] = None,
    style: str = 'auto',
    fps: float = None,
    codec: str = 'mp4v',
    frame_callback: Callable = None
) -> bool:
    """
    Save video with debug visualizations overlaid.
    
    Args:
        input_path: Original video path
        output_path: Output video path
        results: Analysis results from analyze()
        features: List of feature names to visualize (simple mode)
        visualizations: List of visualization configs (advanced mode)
        style: Visualization style ('text', 'dashboard', 'minimal', 'auto')
        fps: Output FPS (None = same as input)
        codec: Video codec
        frame_callback: Optional callback for custom processing
        
    Returns:
        True if successful
        
    Examples:
        # Simple text overlay
        vk.save_debug_video('input.mp4', 'debug.mp4', results, 
                           features=['scene_detection'])
        
        # Custom visualizations
        vk.save_debug_video('input.mp4', 'debug.mp4', results,
                           visualizations=[...])
    """
```

### Preset Styles

For convenience, provide preset visualization styles:

```python
# Minimal style - just essential info
vk.save_debug_video(..., style='minimal')
# Shows: frame number, main feature state

# Text style - all features as text
vk.save_debug_video(..., style='text')  
# Shows: all requested features as text overlays

# Dashboard style - comprehensive view
vk.save_debug_video(..., style='dashboard')
# Shows: dashboard panel with graphs and metrics

# Split style - side-by-side comparison
vk.save_debug_video(..., style='split')
# Shows: original + analysis visualizations

# Auto style - chooses based on features
vk.save_debug_video(..., style='auto')
# Automatically selects appropriate visualizations
```

### Visualization Registry

Allow registering custom visualizers:

```python
from videokurt.visualization import register_visualizer

@register_visualizer('custom_viz')
def my_custom_visualizer(frame, frame_idx, results, config):
    """Draw custom visualization on frame."""
    # Modify frame in place
    cv2.putText(frame, f"Custom: {frame_idx}", ...)
    return frame

# Use custom visualizer
vk.save_debug_video(
    ...,
    visualizations=[
        {'type': 'custom_viz', 'param1': 'value1'}
    ]
)
```

## Feature-Specific Visualizations

### Scene Detection
```python
{
    'type': 'scene_marker',
    'feature': 'scene_detection',
    'style': 'boundary',  # Show scene boundaries
    'transition_duration': 30,  # Frames to show transition
    'label_scenes': True,  # Label with scene numbers
    'color_per_scene': True  # Different color for each scene
}
```

### Scroll Detection
```python
{
    'type': 'scroll_indicator',
    'feature': 'scroll_detection',
    'show_direction': True,  # Up/down arrows
    'show_speed': True,  # Scroll speed indicator
    'position': 'right',
    'style': 'arrow'  # or 'bar', 'text'
}
```

### UI Change Detection
```python
{
    'type': 'change_highlights',
    'feature': 'ui_change_detection',
    'highlight_mode': 'box',  # or 'overlay', 'flash'
    'change_threshold': 0.1,
    'fade_duration': 15  # Frames to fade highlight
}
```

### Camera Movement
```python
{
    'type': 'camera_motion',
    'feature': 'camera_movement',
    'show_vector': True,  # Motion vector arrow
    'show_type': True,  # Pan/zoom/tilt label
    'trail_length': 30  # Frames of motion trail
}
```

## Performance Considerations

### Lazy Rendering
Only process frames that differ from original:

```python
def save_debug_video(..., lazy_rendering=True):
    """
    Only re-encode frames with overlays.
    Frames without visualizations are copied directly.
    """
```

### Multi-threaded Processing
Process visualizations in parallel:

```python
def save_debug_video(..., num_threads=4):
    """
    Use multiple threads for rendering visualizations.
    """
```

### Resolution Options
Render at different resolution than analysis:

```python
def save_debug_video(
    ...,
    render_scale=1.0,  # Render at full resolution
    analysis_scale=0.5  # But use half-res analysis results
):
```

## Usage Examples

### Example 1: Simple Activity Monitoring
```python
# Analyze screen recording for activity
vk = VideoKurt()
vk.add_analysis('frame_diff')
vk.add_feature('binary_activity')
results = vk.analyze('screen_recording.mp4')

# Save with activity indicator
vk.save_debug_video(
    'screen_recording.mp4',
    'activity_debug.mp4',
    results,
    features=['binary_activity'],
    style='minimal'
)
```

### Example 2: Comprehensive Scene Analysis
```python
# Full scene analysis with multiple visualizations
vk = VideoKurt()
vk.add_analysis('frame_diff')
vk.add_analysis('edge_canny')
vk.add_feature('scene_detection')
vk.add_feature('camera_movement')
results = vk.analyze('movie_clip.mp4')

# Save with detailed visualizations
vk.save_debug_video(
    'movie_clip.mp4',
    'scene_analysis.mp4',
    results,
    visualizations=[
        {
            'type': 'scene_marker',
            'feature': 'scene_detection',
            'style': 'boundary'
        },
        {
            'type': 'camera_motion',
            'feature': 'camera_movement',
            'show_vector': True
        },
        {
            'type': 'graph',
            'data': results.analyses['frame_diff'].data['pixel_diff'],
            'position': 'bottom-right',
            'size': (200, 80)
        }
    ]
)
```

### Example 3: UI Testing Debug
```python
# Debug UI automation test
vk = VideoKurt()
vk.add_feature('ui_change_detection')
vk.add_feature('scroll_detection')
results = vk.analyze('ui_test_recording.mp4')

# Save with UI change highlights
vk.save_debug_video(
    'ui_test_recording.mp4',
    'ui_test_debug.mp4',
    results,
    visualizations=[
        {
            'type': 'change_highlights',
            'feature': 'ui_change_detection',
            'highlight_mode': 'box'
        },
        {
            'type': 'scroll_indicator',
            'feature': 'scroll_detection',
            'show_direction': True
        },
        {
            'type': 'dashboard',
            'position': 'top',
            'metrics': [
                {'label': 'UI Change', 'feature': 'ui_change_detection'},
                {'label': 'Scrolling', 'feature': 'scroll_detection'}
            ]
        }
    ]
)
```

### Example 4: Motion Analysis Debug
```python
# Detailed motion analysis
vk = VideoKurt()
vk.add_analysis('optical_flow_dense')
vk.add_analysis('motion_heatmap')
vk.add_analysis('background_mog2')
results = vk.analyze('security_footage.mp4')

# Save with motion visualizations
vk.save_debug_video(
    'security_footage.mp4',
    'motion_debug.mp4',
    results,
    visualizations=[
        {
            'type': 'heatmap',
            'data': results.analyses['motion_heatmap'].data['cumulative'],
            'alpha': 0.4
        },
        {
            'type': 'split_view',
            'mode': 'horizontal',
            'views': [
                {'source': 'original', 'label': 'Original'},
                {'source': 'background_mog2', 'label': 'Foreground'}
            ]
        }
    ]
)
```

## Alternative Interface Ideas

### 1. Fluent Interface
```python
vk.visualize('input.mp4', results)
   .add_text('scene_detection', position='top-left')
   .add_graph('frame_diff', position='bottom-right')
   .add_heatmap('motion_heatmap', alpha=0.5)
   .save('debug_output.mp4')
```

### 2. Declarative Configuration
```python
debug_config = {
    'input': 'input.mp4',
    'output': 'debug.mp4',
    'visualizations': {
        'text': ['scene_detection', 'binary_activity'],
        'graphs': ['frame_diff'],
        'heatmaps': ['motion_heatmap']
    }
}
vk.save_debug_video(**debug_config)
```

### 3. Template-Based
```python
# Use predefined templates
vk.save_debug_video(
    'input.mp4', 'debug.mp4', results,
    template='motion_analysis'  # Predefined visualization set
)

# Or create custom template
vk.register_template('my_template', [
    {'type': 'text', 'feature': 'scene_detection'},
    {'type': 'graph', 'data': 'frame_diff'}
])
vk.save_debug_video(..., template='my_template')
```

## Implementation Priority

### Phase 1: Basic Text Overlay (MVP)
- Simple text overlay for features
- Basic positioning (9-point grid)
- Single color, fixed font

### Phase 2: Core Visualizations
- Binary activity bar
- Simple line graphs
- Basic heatmap overlay

### Phase 3: Advanced Features
- Dashboard view
- Split screen comparisons
- Custom visualizers
- Performance optimizations

### Phase 4: Polish
- Preset styles and templates
- Animation and transitions
- Export to different formats
- Real-time preview

## Open Questions

1. **Should we support real-time preview?**
   - Live preview while configuring visualizations
   - Would require GUI or web interface

2. **How to handle missing features?**
   - Skip silently?
   - Show "N/A" placeholder?
   - Raise warning/error?

3. **Should visualizations be frame-accurate?**
   - Match analysis frame_step or interpolate?
   - How to handle resolution differences?

4. **Export format options?**
   - Just video or also GIF, image sequence?
   - Separate visualization data export?

5. **Memory management for large videos?**
   - Stream processing vs full load?
   - Temporary file usage?

## Conclusion

The visual debug interface should be:
- **Simple by default**: `save_debug_video(input, output, results, features=['scene_detection'])`
- **Powerful when needed**: Full control over visualization types and styles
- **Extensible**: Easy to add new visualization types
- **Performant**: Shouldn't significantly slow down the workflow

This design provides both ease of use for common cases and flexibility for advanced debugging needs.