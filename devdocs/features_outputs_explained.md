# Feature Outputs Explained

This document details the output format for each VideoKurt feature, helping you understand what data structures to expect and how to interpret them.

## Output Format Categories

VideoKurt features return data in three main categories:

1. **Per-frame Arrays**: Simple numpy arrays with one value per frame
2. **Event-based Dictionaries**: Dictionaries containing detected events/segments
3. **Spatial Dictionaries**: Dictionaries with spatial grids and statistics

## Per-Frame Features (Timeline Arrays)

These features return numpy arrays where each element corresponds to a video frame.

### binary_activity
- **Returns**: `np.ndarray` (dtype=uint8)
- **Shape**: `(n_frames,)`
- **Values**: 0 or 1
- **Example**:
```python
# [0, 0, 1, 1, 1, 0, 0, 1, 0, 0]
# Frame 0-1: inactive, Frame 2-4: active, etc.
```
- **Usage**:
```python
activity = results.features['binary_activity'].data
active_frames = np.sum(activity)  # Count active frames
activity_ratio = np.mean(activity)  # Percentage of active time
```

### motion_magnitude
- **Returns**: `np.ndarray` (dtype=float32)
- **Shape**: `(n_frames,)`
- **Values**: 0.0 to max_motion (unbounded)
- **Example**:
```python
# [0.1, 0.2, 5.3, 4.8, 0.3, 0.1, ...]
# Higher values = more motion
```
- **Usage**:
```python
motion = results.features['motion_magnitude'].data
avg_motion = np.mean(motion)
peak_motion = np.max(motion)
motion_periods = motion > threshold  # Binary mask of high motion
```

### stability_score
- **Returns**: `np.ndarray` (dtype=float32)
- **Shape**: `(n_frames,)`
- **Values**: 0.0 to 1.0
- **Example**:
```python
# [0.95, 0.98, 0.12, 0.15, 0.89, 0.91, ...]
# Values near 1.0 = stable/idle, near 0.0 = changing
```
- **Usage**:
```python
stability = results.features['stability_score'].data
idle_frames = stability > 0.9  # Frames with minimal change
reading_time = np.sum(idle_frames) / fps  # Seconds of idle time
```

### edge_density
- **Returns**: `np.ndarray` (dtype=float32)
- **Shape**: `(n_frames,)`
- **Values**: 0.0 to 1.0 (percentage)
- **Example**:
```python
# [0.45, 0.48, 0.12, 0.08, 0.52, ...]
# Higher values = more edges (text/detailed content)
```
- **Usage**:
```python
edges = results.features['edge_density'].data
text_frames = edges > 0.4  # Likely text content
avg_complexity = np.mean(edges)
```

### dominant_flow_vector
- **Returns**: `np.ndarray` (dtype=float32)
- **Shape**: `(n_frames, 2)` - [x_component, y_component]
- **Values**: Flow vectors (can be positive or negative)
- **Example**:
```python
# [[0.0, -5.2],   # Frame 0: upward motion
#  [0.1, -4.8],   # Frame 1: upward motion
#  [2.3, 0.1], ]  # Frame 2: rightward motion
```
- **Usage**:
```python
flow = results.features['dominant_flow_vector'].data
directions = np.arctan2(flow[:, 1], flow[:, 0])  # Angles
speeds = np.linalg.norm(flow, axis=1)  # Magnitudes
```

## Event-Based Features (Dictionaries)

These features detect and return specific events or segments rather than per-frame values.

### scene_detection
- **Returns**: `Dict[str, Any]`
- **Structure**:
```python
{
    'boundaries': [
        {'frame': 120, 'type': 'cut', 'confidence': 0.95},
        {'frame': 450, 'type': 'fade', 'confidence': 0.82},
        ...
    ],
    'scenes': [
        {'start': 0, 'end': 120, 'length': 120},
        {'start': 120, 'end': 450, 'length': 330},
        ...
    ],
    'num_scenes': 5,
    'avg_scene_length': 234.5
}
```
- **Usage**:
```python
scenes = results.features['scene_detection'].data
print(f"Found {scenes['num_scenes']} scenes")
for scene in scenes['scenes']:
    duration = scene['length'] / fps
    print(f"Scene from frame {scene['start']} ({duration:.1f}s)")
```

### scrolling_detection
- **Returns**: `Dict[str, Any]`
- **Structure**:
```python
{
    'scroll_events': [
        {
            'type': 'scroll',
            'direction': 'down',
            'start_frame': 100,
            'end_frame': 180,
            'avg_speed': 12.5  # pixels/frame
        },
        ...
    ],
    'num_scroll_events': 3,
    'total_scroll_frames': 245,
    'scroll_directions': {
        'up': 1, 'down': 2, 'left': 0, 'right': 0
    },
    'dominant_direction': 'down'
}
```
- **Usage**:
```python
scrolling = results.features['scrolling_detection'].data
if scrolling['num_scroll_events'] > 0:
    print(f"User scrolled {scrolling['dominant_direction']} mostly")
    for event in scrolling['scroll_events']:
        duration = (event['end_frame'] - event['start_frame']) / fps
        print(f"{event['direction']} scroll for {duration:.1f}s")
```

### activity_bursts
- **Returns**: `Dict[str, Any]`
- **Structure**:
```python
{
    'bursts': [
        {
            'start': 150,
            'end': 200,
            'duration': 50,
            'peak_intensity': 0.95,
            'avg_intensity': 0.78
        },
        ...
    ],
    'num_bursts': 4,
    'burst_ratio': 0.15,  # Percentage of time in bursts
    'avg_burst_intensity': 0.72,
    'burst_patterns': {
        'pattern': 'irregular',  # or 'periodic'
        'regularity': 0.3,
        'avg_interval': 120.5
    },
    'activity_timeline': np.ndarray  # Normalized 0-1 per frame
}
```
- **Usage**:
```python
bursts = results.features['activity_bursts'].data
print(f"Found {bursts['num_bursts']} activity bursts")
print(f"Pattern: {bursts['burst_patterns']['pattern']}")
intense_periods = [b for b in bursts['bursts'] if b['peak_intensity'] > 0.8]
```

### blob_tracking
- **Returns**: `Dict[str, Any]`
- **Structure**:
```python
{
    'blobs': [
        {
            'id': 1,
            'first_frame': 10,
            'last_frame': 150,
            'positions': [(x1,y1), (x2,y2), ...],
            'sizes': [area1, area2, ...],
            'lifetime': 140
        },
        ...
    ],
    'num_blobs': 12,
    'avg_lifetime': 45.5,
    'max_lifetime': 140,
    'timeline': {  # Blob count per frame
        'counts': [2, 2, 3, 3, 2, 1, ...],
        'total_frames': 500
    }
}
```

### connected_components
- **Returns**: `Dict[str, Any]`
- **Structure**:
```python
{
    'components_per_frame': [3, 4, 4, 2, ...],  # Count per frame
    'avg_components': 3.2,
    'max_components': 8,
    'stable_components': [  # Components lasting >min_frames
        {'frames': [10, 11, 12, ...], 'avg_area': 1250},
        ...
    ],
    'component_stats': {
        'total_unique': 45,
        'avg_lifetime': 12.3
    }
}
```

## Spatial Features (Grid/Map Dictionaries)

These features analyze spatial distribution of activity across the frame.

### spatial_occupancy_grid
- **Returns**: `Dict[str, Any]`
- **Structure**:
```python
{
    'occupancy_grid': np.ndarray,  # Shape: (grid_h, grid_w), e.g., (10, 10)
    'temporal_occupancy': np.ndarray,  # Shape: (n_frames, grid_h, grid_w)
    'max_occupancy': 0.85,
    'occupancy_distribution': np.ndarray  # Flattened grid
}
```
- **Usage**:
```python
occupancy = results.features['spatial_occupancy_grid'].data
grid = occupancy['occupancy_grid']  # 2D heat map
most_active = np.unravel_index(np.argmax(grid), grid.shape)
print(f"Most active region: row {most_active[0]}, col {most_active[1]}")

# Visualize as heatmap
import matplotlib.pyplot as plt
plt.imshow(grid, cmap='hot')
plt.colorbar(label='Activity Level')
```

### dwell_time_maps
- **Returns**: `Dict[str, Any]`
- **Structure**:
```python
{
    'dwell_map': np.ndarray,  # Shape: (height, width) - accumulated time
    'normalized_map': np.ndarray,  # Shape: (height, width) - 0-1 normalized
    'hotspots': [  # Regions with high dwell time
        {'center': (x, y), 'radius': r, 'total_time': t},
        ...
    ],
    'coverage': 0.65  # Percentage of frame with activity
}
```

### zone_based_activity
- **Returns**: `Dict[str, Any]`
- **Structure**:
```python
{
    'zones': [
        {
            'id': 0,
            'bounds': (x1, y1, x2, y2),
            'activity_level': 0.75,
            'active_frames': 234
        },
        ...
    ],
    'most_active_zone': 3,
    'activity_distribution': [0.2, 0.1, 0.4, 0.3]  # Per zone
}
```

## Motion Analysis Features

### motion_trajectories
- **Returns**: `Dict[str, Any]`
- **Structure**:
```python
{
    'trajectories': [
        {
            'id': 1,
            'points': [(x1,y1), (x2,y2), ...],
            'frames': [10, 11, 12, ...],
            'length': 125.5,  # Total path length in pixels
            'duration': 45    # Frames
        },
        ...
    ],
    'num_trajectories': 8,
    'avg_trajectory_length': 89.3,
    'longest_trajectory': {'id': 3, 'length': 234.5}
}
```

### motion_direction_histogram
- **Returns**: `Dict[str, Any]`
- **Structure**:
```python
{
    'histogram': np.ndarray,  # Shape: (n_bins,) default 8 directions
    'dominant_direction': 3,  # Bin index
    'direction_labels': ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'],
    'direction_percentages': [0.1, 0.05, 0.4, 0.2, ...],
    'entropy': 2.34  # Diversity of directions (higher = more varied)
}
```

## Temporal Pattern Features

### periodicity_strength
- **Returns**: `Dict[str, Any]`
- **Structure**:
```python
{
    'periodicity_score': 0.78,  # 0-1, higher = more periodic
    'dominant_frequency': 0.5,   # Hz
    'dominant_period': 2.0,      # Seconds
    'frequency_spectrum': np.ndarray,  # FFT magnitudes
    'periodic_segments': [
        {'start': 100, 'end': 300, 'frequency': 0.5},
        ...
    ]
}
```

### repetition_indicator
- **Returns**: `Dict[str, Any]`
- **Structure**:
```python
{
    'repetition_score': 0.65,  # 0-1, higher = more repetitive
    'repeated_segments': [
        {'frames': [10, 110, 210], 'similarity': 0.92},
        ...
    ],
    'num_repetitions': 3,
    'pattern_type': 'periodic'  # or 'sporadic', 'none'
}
```

### structural_similarity
- **Returns**: `np.ndarray` (dtype=float32)
- **Shape**: `(n_frames-1,)`  # One less than total frames
- **Values**: 0.0 to 1.0 (SSIM scores)
- **Note**: Compares each frame to the previous one
- **Usage**:
```python
ssim = results.features['structural_similarity'].data
changes = ssim < 0.8  # Significant visual changes
avg_similarity = np.mean(ssim)
```

## Comparison Features

### perceptual_hashes
- **Returns**: `Dict[str, Any]`
- **Structure**:
```python
{
    'hashes': ['a3f2b1c4...', 'b4c2a1f3...', ...],  # One per frame
    'duplicates': [
        {'frames': [10, 45, 102], 'hash': 'a3f2b1c4...'},
        ...
    ],
    'num_unique': 234,
    'uniqueness_ratio': 0.78  # Unique frames / total frames
}
```

## Working with Feature Outputs

### Type Checking
Always check the data type before accessing:

```python
data = results.features['some_feature'].data

if isinstance(data, np.ndarray):
    # Handle array data
    avg_value = np.mean(data)
    timeline = data
elif isinstance(data, dict):
    # Handle dictionary data
    if 'timeline' in data:
        timeline = data['timeline']
    if 'events' in data:
        events = data['events']
```

### Missing Features
Features may fail to compute. Always check existence:

```python
if 'motion_magnitude' in results.features:
    motion = results.features['motion_magnitude'].data
else:
    print("Motion magnitude not computed")
    motion = None
```

### Combining Features
Different features can be combined for richer analysis:

```python
# Combine stability and activity for idle detection
stability = results.features['stability_score'].data
activity = results.features['binary_activity'].data

# True idle: stable AND inactive
true_idle = (stability > 0.9) & (activity == 0)
idle_ratio = np.mean(true_idle)

# Combine scrolling and scene detection
scrolling = results.features['scrolling_detection'].data
scenes = results.features['scene_detection'].data

# Check if scrolling happened during scene
for scroll in scrolling['scroll_events']:
    for scene in scenes['scenes']:
        if scene['start'] <= scroll['start_frame'] <= scene['end']:
            print(f"Scrolling in scene {scene}")
```

## Summary

- **Arrays** are best for timeline analysis and statistics
- **Event dictionaries** are best for detecting specific behaviors
- **Spatial dictionaries** are best for understanding where activity occurs
- Always check data types and handle missing features gracefully
- Combine multiple features for more sophisticated analysis