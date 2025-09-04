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

# Save with debug visualization - input path is in results
results.save_debug_video(
    output_path='debug_output.mp4',
    features=['scene_detection', 'binary_activity'],  # Which features to show
    style='text'  # Simple text overlay
)
```

### Advanced Usage - Custom Visualization (Current Capabilities)

```python
# Using visual_debugger's actual annotation types
from visual_debugger import AnnotationType

results.save_debug_video(
    output_path='debug_output.mp4',
    visualizations=[
        {
            'type': AnnotationType.POINT_AND_LABEL,
            'feature': 'scene_detection',
            'coordinates': (50, 50),
            'color': (0, 255, 0)
        },
        {
            'type': AnnotationType.CIRCLE_AND_LABEL,
            'feature': 'binary_activity',  # Shows as colored circle
            'coordinates': (50, 100),
            'radius': 20,
            'thickness': -1,  # Filled
            'color_active': (0, 255, 0),
            'color_inactive': (128, 128, 128)
        },
        {
            'type': AnnotationType.RECTANGLE,
            'coordinates': (10, 150, 200, 100),  # Motion region
            'color': (255, 255, 0)
        }
    ]
)
```

### Future Vision - With Enhanced visual_debugger

```python
# This would require implementing enhancements from enhancement_proposal.md
results.save_debug_video(
    output_path='debug_output.mp4',
    visualizations=[
        {
            'type': 'text',
            'feature': 'scene_detection',
            'position': 'top-left',
            'color': (0, 255, 0),
            'font_scale': 0.7
        },
        {
            'type': 'binary_bar',  # Would need to be added
            'feature': 'binary_activity',
            'position': 'bottom',
            'height': 30,
            'active_color': (0, 255, 0),
            'inactive_color': (128, 128, 128)
        },
        {
            'type': 'graph',  # Would need to be added
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

## Integration with visual_debugger Package

The `visual_debugger` package (already available at `/Users/ns/Desktop/projects/visual_debugger`) provides excellent annotation capabilities that can be leveraged for VideoKurt's debug visualization. Key benefits:

- **Pre-built annotation system** with points, labels, rectangles, circles, lines, masks
- **Image concatenation** for creating comparison views
- **Efficient OpenCV-based rendering**

### Using visual_debugger for VideoKurt

```python
from visual_debugger import VisualDebugger, Annotation, AnnotationType

class RawAnalysisResults:
    def save_debug_video(self, output_path, features=None, style='auto'):
        """Save video with debug annotations using visual_debugger."""
        
        # Initialize visual debugger (output='return' to get annotated frames)
        vd = VisualDebugger(tag="videokurt", output='return', active=True)
        
        # Process each frame
        cap = cv2.VideoCapture(self.filename)
        annotated_frames = []
        
        for frame_idx in range(self.frame_count):
            ret, frame = cap.read()
            if not ret:
                break
                
            # Create annotations based on features
            annotations = self._create_annotations(frame_idx, features)
            
            # Apply annotations
            annotated_frame = vd.visual_debug(frame, annotations)
            annotated_frames.append(annotated_frame)
        
        # Save annotated video
        VideoKurt.save_video(annotated_frames, output_path)
```

## Implementation Details

### Main Method: save_debug_video (on RawAnalysisResults)

```python
class RawAnalysisResults:
    # ... existing fields ...
    filename: str  # Already stored from analyze()
    
    def save_debug_video(
        self,
        output_path: Union[str, Path],
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
            output_path: Output video path
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
            results.save_debug_video('debug.mp4', 
                                    features=['scene_detection'])
            
            # Custom visualizations
            results.save_debug_video('debug.mp4',
                                    visualizations=[...])
        """
        # Use self.filename as input video path
        input_path = self.filename
```

### Preset Styles

For convenience, provide preset visualization styles:

```python
# Minimal style - just essential info
results.save_debug_video('output.mp4', style='minimal')
# Shows: frame number, main feature state

# Text style - all features as text
results.save_debug_video('output.mp4', style='text')  
# Shows: all requested features as text overlays

# Dashboard style - comprehensive view
results.save_debug_video('output.mp4', style='dashboard')
# Shows: dashboard panel with graphs and metrics

# Split style - side-by-side comparison
results.save_debug_video('output.mp4', style='split')
# Shows: original + analysis visualizations

# Auto style - chooses based on features
results.save_debug_video('output.mp4', style='auto')
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
results.save_debug_video(
    'output.mp4',
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
results.save_debug_video(
    'activity_debug.mp4',
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
results.save_debug_video(
    'scene_analysis.mp4',
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
results.save_debug_video(
    'ui_test_debug.mp4',
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
results.save_debug_video(
    'motion_debug.mp4',
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
    'output_path': 'debug.mp4',
    'visualizations': {
        'text': ['scene_detection', 'binary_activity'],
        'graphs': ['frame_diff'],
        'heatmaps': ['motion_heatmap']
    }
}
results.save_debug_video(**debug_config)
```

### 3. Template-Based
```python
# Use predefined templates
results.save_debug_video(
    'debug.mp4',
    template='motion_analysis'  # Predefined visualization set
)

# Or create custom template
VideoKurt.register_template('my_template', [
    {'type': 'text', 'feature': 'scene_detection'},
    {'type': 'graph', 'data': 'frame_diff'}
])
results.save_debug_video('output.mp4', template='my_template')
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

## Mapping VideoKurt Features to visual_debugger Annotations

### Example Implementation

```python
def _create_annotations_for_frame(self, frame_idx, features):
    """Create visual_debugger annotations for a specific frame."""
    from visual_debugger import Annotation, AnnotationType
    
    annotations = []
    
    # Scene detection - show scene boundaries
    if 'scene_detection' in features and 'scene_detection' in self.features:
        scene_data = self.features['scene_detection']
        if frame_idx in scene_data.scene_changes:
            annotations.append(
                Annotation(
                    type=AnnotationType.RECTANGLE,
                    coordinates=(10, 10, 200, 50),
                    color=(0, 255, 0),
                )
            )
            annotations.append(
                Annotation(
                    type=AnnotationType.POINT_AND_LABEL,
                    coordinates=(110, 35),
                    labels=f"Scene Change #{scene_data.scene_id[frame_idx]}",
                    color=(0, 255, 0)
                )
            )
    
    # Binary activity - show activity indicator
    if 'binary_activity' in features and 'binary_activity' in self.features:
        activity = self.features['binary_activity'].is_active[frame_idx]
        color = (0, 255, 0) if activity else (128, 128, 128)
        annotations.append(
            Annotation(
                type=AnnotationType.CIRCLE_AND_LABEL,
                coordinates=(50, 100),
                radius=20,
                thickness=-1,  # Filled circle
                labels="Active" if activity else "Inactive",
                color=color
            )
        )
    
    # Motion regions - draw rectangles around motion
    if 'motion_regions' in features and 'contour_detection' in self.analyses:
        contours = self.analyses['contour_detection'].data['contours'][frame_idx]
        for contour in contours[:5]:  # Show top 5 contours
            x, y, w, h = cv2.boundingRect(contour)
            annotations.append(
                Annotation(
                    type=AnnotationType.RECTANGLE,
                    coordinates=(x, y, w, h),
                    color=(255, 0, 0)
                )
            )
    
    # Optical flow - show motion vectors
    if 'motion_vectors' in features and 'optical_flow_sparse' in self.analyses:
        points = self.analyses['optical_flow_sparse'].data['tracked_points'][frame_idx]
        for pt in points[:20]:  # Show up to 20 points
            annotations.append(
                Annotation(
                    type=AnnotationType.POINT,
                    coordinates=tuple(pt.astype(int)),
                    color=(255, 255, 0)
                )
            )
    
    # Frame info - always show
    annotations.append(
        Annotation(
            type=AnnotationType.POINT_AND_LABEL,
            coordinates=(10, 30),
            labels=f"Frame {frame_idx}/{self.frame_count}",
            color=(255, 255, 255)
        )
    )
    
    return annotations
```

### Simplified Usage Example

```python
# Analyze video
vk = VideoKurt()
vk.add_analysis('frame_diff')
vk.add_analysis('contour_detection')
vk.add_feature('scene_detection')
vk.add_feature('binary_activity')
results = vk.analyze('video.mp4')

# Save with visual debugging using visual_debugger
results.save_debug_video(
    'debug.mp4',
    features=['scene_detection', 'binary_activity', 'motion_regions']
)
```

## Advantages of Using visual_debugger

1. **No need to reimplement** - Annotation drawing is already handled
2. **Consistent API** - Use Annotation dataclasses
3. **Proven code** - Already tested and working
4. **Easy to extend** - Just add new annotation mappings
5. **Performance** - Optimized OpenCV operations

## Implementation Priority with visual_debugger

### Phase 1: Basic Integration (Quick Win)
- Import visual_debugger
- Map 2-3 basic features to annotations (text labels, activity indicators)
- Test with simple videos

### Phase 2: Feature Mappings
- Map all VideoKurt features to appropriate annotations
- Add configuration for annotation styles
- Support feature-specific visualizations

### Phase 3: Advanced Visualizations
- Use visual_debugger's mask overlay for heatmaps
- Implement side-by-side comparisons
- Add concat_images for multi-analysis views

## Conclusion

By leveraging the existing `visual_debugger` package:
- **Faster implementation**: Reuse proven annotation code
- **Cleaner design**: Focus on feature-to-annotation mapping
- **Better maintainability**: Separate concerns between analysis and visualization
- **Immediate value**: Can start using it right away

The visual debug interface becomes:
- **Simple by default**: `results.save_debug_video('output.mp4', features=['scene_detection'])`
- **Powerful when needed**: Full control via custom annotations
- **Extensible**: Easy to add new feature mappings
- **Performant**: Leverages optimized visual_debugger code

This design provides both ease of use for common cases and flexibility for advanced debugging needs.