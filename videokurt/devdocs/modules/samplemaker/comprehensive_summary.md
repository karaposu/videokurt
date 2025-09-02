# SampleMaker Module - Comprehensive Summary

## Module Overview

The SampleMaker module (`videokurt/samplemaker/`) provides a comprehensive suite of utilities for creating synthetic video frames and sequences for testing VideoKurt components without requiring actual video files or external dependencies.

## Purpose and Rationale

### Why This Module Exists
1. **Dependency-Free Testing**: Enables testing without video files or large assets
2. **Controlled Conditions**: Creates precise, reproducible test scenarios
3. **Rapid Prototyping**: Quick generation of edge cases and specific patterns
4. **Performance Testing**: Lightweight frames (20x20 default) for fast iteration
5. **Event Simulation**: Mimics real video events (scrolls, popups, scene changes)

### Core Design Principles
- **Simplicity**: Pure NumPy arrays, no complex dependencies
- **Flexibility**: Parameterized generation for various scenarios
- **Realism**: Simulates actual video artifacts and patterns
- **Modularity**: Composable functions for complex scenes
- **Efficiency**: Small default sizes for rapid testing

## Module Structure

```
videokurt/samplemaker/
├── __init__.py           # Module exports and main API
├── frames.py             # Basic frame generation (blank, solid, gradient, checkerboard)
├── shapes.py             # Shape drawing utilities (rectangles, circles, text)
├── effects.py            # Visual effects (noise, compression artifacts)
├── motion.py             # Motion simulation (scroll, scene change, popup)
└── sequences.py          # Frame sequences and complete test videos
```

## Key Components

### 1. Basic Frame Generators

#### Foundation Functions
- `create_blank_frame()` - Black/empty frames
- `create_solid_frame()` - Uniform color frames
- `create_gradient_frame()` - Linear gradients (H/V/diagonal)
- `create_checkerboard()` - Alternating pattern blocks

#### Purpose
These provide base canvases for building more complex test scenarios. They establish known baselines for difference calculations.

### 2. Shape Drawing Functions

#### Drawing Primitives
- `add_rectangle()` - Filled or outline rectangles
- `add_circle()` - Filled or outline circles
- `add_text_region()` - Simulated text blocks with lines

#### Purpose
Enable creation of UI-like elements for testing event detection (popups, buttons, text areas).

### 3. Motion Simulators

#### Movement Functions
- `simulate_scroll()` - Pixel shifting in any direction
- `simulate_scene_change()` - Cut/fade/slide transitions
- `simulate_popup()` - Modal overlay with background dimming

#### Purpose
Test motion detection algorithms and event boundary identification without real video motion.

### 4. Noise and Artifact Generators

#### Distortion Functions
- `add_noise()` - Gaussian/salt-pepper/uniform noise
- `add_compression_artifacts()` - Block-based quality reduction

#### Purpose
Test robustness against real-world video quality issues and compression effects.

### 5. Sequence Generators

#### Timeline Functions
- `create_frame_sequence()` - Simple activity/idle/mixed sequences
- `create_test_video_frames()` - Complete test videos with comprehensive ground truth

#### Purpose
Create temporal sequences with precise ground truth for testing all VideoKurt components:
- Frame-by-frame annotations with timestamps
- Binary activity timeline with start/end frames
- Detailed event metadata including confidence scores
- Support for testing detection accuracy against known truth

## Implementation Details

### Frame Representation
- **Format**: NumPy arrays (H×W for grayscale, H×W×3 for BGR)
- **Data Type**: `np.uint8` (0-255 intensity range)
- **Default Size**: 20×20 pixels (configurable)
- **Color Order**: BGR (OpenCV convention)

### Coordinate System
- **Origin**: Top-left corner (0, 0)
- **X-axis**: Horizontal (left to right)
- **Y-axis**: Vertical (top to bottom)
- **Indexing**: `frame[y, x]` or `frame[y, x, channel]`

### Event Simulation Approach
1. **Scene Changes**: Complete frame replacement or gradual transition
2. **Scrolling**: Pixel shifting with wraparound
3. **Popups**: Overlay with darkened background
4. **Idle Periods**: Repeated identical frames
5. **Activity**: Moving objects or changing patterns

### Ground Truth Structure

#### Event Dictionary
```python
{
    'type': 'scroll',              # Event type identifier
    'start': 1.0,                  # Start time in seconds
    'end': 2.0,                    # End time in seconds
    'start_frame': 10,             # Starting frame index
    'end_frame': 20,               # Ending frame index
    'confidence': 1.0,             # Ground truth confidence (always 1.0)
    'metadata': {                  # Event-specific details
        'direction': 'down',
        'velocity': 100
    }
}
```

#### Frame Annotation
```python
{
    'frame_idx': 15,               # Frame number
    'timestamp': 1.5,              # Time in seconds
    'event_type': 'scroll',        # Current event type
    'active': True,                # Activity state
    'has_noise': False             # Optional noise flag
}
```

#### Activity Timeline Entry
```python
{
    'active': True,                # Binary activity state
    'start': 1.0,                  # Period start time
    'end': 2.0,                    # Period end time
    'start_frame': 10,             # Starting frame
    'end_frame': 20                # Ending frame
}
```

## Usage Examples

### Basic Testing
```python
from videokurt.samplemaker import create_blank_frame, add_circle
from videokurt.core import SimpleFrameDiff

# Create two frames with slight difference
frame1 = create_blank_frame((20, 20))
frame2 = add_circle(frame1, center=(10, 10), radius=3)

# Test differencing
diff = SimpleFrameDiff()
result = diff.compute_difference(frame1, frame2)
assert result.score > 0  # Change detected
```

### Event Detection Testing
```python
from videokurt.samplemaker import create_gradient_frame, simulate_scroll

# Test scroll detection
base = create_gradient_frame((30, 30), direction='vertical')
scrolled = simulate_scroll(base, pixels=5, direction='down')

# Frames should show vertical motion
# Use with optical flow or scroll detection algorithms
```

### Complete Sequence Testing with Ground Truth
```python
from videokurt.samplemaker import create_test_video_frames

# Generate comprehensive test video with full ground truth
test_data = create_test_video_frames(
    size=(50, 50),
    events={
        'scene_changes': True,
        'scrolls': True,
        'popups': True,
        'idle_periods': True,
        'noise': False  # Keep false for clean ground truth
    },
    fps=10.0  # Configurable frame rate
)

# Access complete data
frames = test_data['frames']  # List of numpy arrays
events = test_data['events']  # Detailed event information
ground_truth = test_data['ground_truth']  # Frame-by-frame annotations
activity_timeline = test_data['activity_timeline']  # Binary activity periods
total_frames = test_data['total_frames']
duration = test_data['duration']

# Test against known events with precise timing
for event in events:
    print(f"{event['type']}: frames {event['start_frame']}-{event['end_frame']}")
    print(f"  Time: {event['start']:.2f}s to {event['end']:.2f}s")
    print(f"  Confidence: {event['confidence']}")
    if 'metadata' in event:
        print(f"  Metadata: {event['metadata']}")

# Validate binary activity timeline
for period in activity_timeline:
    state = 'Active' if period['active'] else 'Inactive'
    print(f"{state}: frames {period['start_frame']}-{period['end_frame']}")

# Use frame-level ground truth for accuracy testing
for annotation in ground_truth:
    if annotation['event_type'] != 'idle':
        print(f"Frame {annotation['frame_idx']} @ {annotation['timestamp']:.2f}s: "
              f"{annotation['event_type']} (active={annotation['active']})")
```

## Integration with Core Modules

### Frame Differencing
- Provides controlled frame pairs with known differences
- Tests threshold sensitivity and noise handling
- Validates all differencing algorithms (Simple, Histogram, SSIM, Hybrid)
- Ground truth enables precise accuracy measurement

### Binary Activity Timeline
- Creates sequences with known active/inactive periods
- Tests state transition detection with exact frame boundaries
- Validates minimum duration filtering
- Activity timeline ground truth for comparison

### Event Detection
- Simulates specific event types for detector validation
- Provides ground truth with frame-accurate timing
- Tests confidence scoring calibration
- Includes event metadata for detailed validation

### Accuracy Testing
```python
# Compare detected events with ground truth
def calculate_accuracy(detected_events, ground_truth_events):
    for gt_event in ground_truth_events:
        # Find matching detected event
        detected = find_matching_event(
            detected_events, 
            gt_event['type'],
            gt_event['start_frame'],
            gt_event['end_frame']
        )
        if detected:
            # Calculate overlap, timing accuracy, etc.
            overlap = calculate_iou(detected, gt_event)
            print(f"Event {gt_event['type']}: {overlap:.2%} accuracy")
```

## Testing Strategy Benefits

### Advantages
1. **No External Files**: Tests run without video assets
2. **Deterministic**: Same inputs always produce same outputs
3. **Fast Execution**: Small frames process quickly
4. **Edge Case Coverage**: Easy to create specific scenarios
5. **Complete Ground Truth**: Frame-accurate timing and annotations
6. **Accuracy Measurement**: Precise validation against known truth
7. **Multi-level Testing**: Frame, event, and timeline validation

### Limitations
1. **Simplified Patterns**: Not as complex as real video
2. **No Audio**: Pure visual testing only
3. **Limited Realism**: Synthetic patterns may not capture all real-world complexity
4. **Small Scale**: Default sizes may not reveal scaling issues

## Future Enhancements

### Potential Additions
1. **Video Codec Simulation**: More realistic compression artifacts
2. **Camera Effects**: Blur, focus changes, exposure variations
3. **Complex Motion**: Non-linear movements, rotations
4. **Multi-Object Tracking**: Multiple moving elements
5. **Temporal Patterns**: Periodic events, rhythmic changes

### Performance Optimizations
1. **Batch Generation**: Create multiple frames simultaneously
2. **GPU Acceleration**: NumPy operations on CUDA
3. **Caching**: Reuse common base frames
4. **Lazy Generation**: Create frames on-demand

## Module Maintenance

### Testing the Test Module
- Unit tests for each generator function
- Visual inspection tools for generated frames
- Regression tests for consistent output

### Documentation Updates
- Keep examples current with API changes
- Document new parameters and options
- Maintain compatibility notes

## Conclusion

The SampleMaker module provides essential infrastructure for testing VideoKurt without external dependencies. Its modular structure, comprehensive ground truth generation, and precise timing information enable accurate validation of all detection algorithms. The enhanced `create_test_video_frames()` function returns complete ground truth data including:

- Frame-by-frame annotations with timestamps
- Binary activity timeline with exact boundaries
- Detailed event information with confidence scores
- Event metadata for thorough validation

This enables developers to:
- Measure detection accuracy precisely
- Validate timing accuracy to the frame level
- Test binary activity classification
- Ensure confidence scoring calibration
- Debug detection failures with known truth

The module embodies the principle of making testing both comprehensive and frictionless, enabling rapid development and validation of VideoKurt's detection algorithms with confidence in their accuracy.