# New VideoKurt Architecture

## Overview

The new VideoKurt system is designed for raw visual analysis without semantic interpretation. It provides 11 different computer vision analyses that can be configured independently and run on video files to extract raw visual patterns and motion data. 

## Core Principles

1. **Raw Analysis Output** - No semantic interpretation, just visual patterns
2. **Configurable Analyses** - Each analysis can be configured independently
3. **Modular Architecture** - Pick and choose which analyses to run
4. **Performance Focused** - Built-in downsampling and optimization options

## Available Analyses

### Level 1: Basic Analyses
- `frame_diff` - Simple pixel differencing between frames
- `edge_canny` - Edge detection using Canny algorithm
- `frame_diff_advanced` - Triple differencing and running average
- `contour_detection` - Shape boundary detection from motion

### Level 2: Intermediate Analyses  
- `background_mog2` - MOG2 background subtraction
- `background_knn` - KNN background subtraction
- `optical_flow_sparse` - Lucas-Kanade sparse optical flow
- `optical_flow_dense` - Farneback dense optical flow

### Level 3: Advanced Analyses
- `motion_heatmap` - Cumulative motion accumulation
- `frequency_fft` - Temporal frequency analysis
- `flow_hsv_viz` - HSV visualization of optical flow

## Usage Examples

### Basic Usage - Run All Analyses

```python
from videokurt import VideoKurt

vk = VideoKurt()
results = vk.analyze("path/to/video.mp4")

# Access results
for analysis_name, analysis_result in results.analyses.items():
    print(f"{analysis_name}: {analysis_result.data.keys()}")
```

### Selective Analysis

```python
from videokurt import VideoKurt

vk = VideoKurt()

# Run only specific analyses
results = vk.analyze(
    "path/to/video.mp4",
    analyses=['frame_diff', 'optical_flow_dense', 'motion_heatmap']
)

# Access specific analysis data
frame_diffs = results.analyses['frame_diff'].data['pixel_diff']
flow_fields = results.analyses['optical_flow_dense'].data['flow_field']
heatmap = results.analyses['motion_heatmap'].data['cumulative']
```

### Custom Configuration Per Analysis

```python
from videokurt import VideoKurt
from videokurt.analysis_models import (
    FrameDiff, 
    OpticalFlowDense, 
    MotionHeatmap
)

vk = VideoKurt()

# Create configured analysis instances
analyses = {
    'frame_diff': FrameDiff(
        downsample=0.5,
        threshold=0.1
    ),
    'optical_flow_dense': OpticalFlowDense(
        downsample=0.25,
        levels=3,
        winsize=15,
        iterations=5
    ),
    'motion_heatmap': MotionHeatmap(
        downsample=0.25,
        decay_factor=0.95,
        snapshot_interval=30
    )
}

# Run with custom configurations
results = vk.analyze(
    "path/to/video.mp4",
    analyses=analyses  # Pass configured instances
)
```

### Global Configuration

```python
from videokurt import VideoKurt

vk = VideoKurt(
    # Global settings
    max_frames=1000,      # Process max 1000 frames
    max_seconds=30,       # Or max 30 seconds
    frame_step=2,         # Process every 2nd frame
    downsample=0.5        # Global downsample (can be overridden per analysis)
)

# These global settings apply to all analyses
results = vk.analyze("path/to/video.mp4")
```

### Mixed Configuration

```python
from videokurt import VideoKurt
from videokurt.analysis_models import OpticalFlowDense

vk = VideoKurt(
    max_seconds=60,       # Global limit
    downsample=0.5        # Default downsample
)

# Mix: some with default config, some custom
results = vk.analyze(
    "path/to/video.mp4",
    analyses={
        'frame_diff': None,  # Use defaults
        'edge_canny': None,  # Use defaults
        'optical_flow_dense': OpticalFlowDense(
            downsample=0.1,  # Override global downsample
            levels=5         # Custom parameter
        )
    }
)
```

## Configuration Hierarchy

1. **Analysis-specific config** (highest priority)
   - Parameters passed to analysis class constructor
   
2. **Global VideoKurt config** 
   - Applied to all analyses unless overridden
   - Includes: downsample, max_frames, max_seconds, frame_step
   
3. **Analysis defaults** (lowest priority)
   - Built-in defaults for each analysis class

## Output Structure

```python
results = vk.analyze("video.mp4")

# RawAnalysisResults object contains:
results.dimensions       # (width, height) of original video
results.fps             # Frames per second
results.duration        # Total duration in seconds
results.frame_count     # Number of frames processed
results.elapsed_time    # Processing time
results.analyses        # Dict[str, RawAnalysis]

# Each RawAnalysis contains:
analysis = results.analyses['frame_diff']
analysis.method         # 'frame_diff'
analysis.data          # Dict with actual data arrays
analysis.parameters    # Configuration used
analysis.processing_time # Time for this analysis
analysis.output_shapes # Shapes of output arrays
analysis.dtype_info    # Data types of outputs
```

## Performance Considerations

### Downsampling Strategy

Different analyses benefit from different downsampling:

```python
# Recommended downsampling per analysis type
config = {
    'frame_diff': 0.5,           # Moderate downsampling OK
    'edge_canny': 0.5,           # Moderate downsampling OK
    'frame_diff_advanced': 0.5,  # Moderate downsampling OK
    'contour_detection': 0.5,    # Often needs downsampling
    'background_mog2': 0.5,      # Benefits from downsampling
    'background_knn': 0.5,       # Benefits from downsampling
    'optical_flow_sparse': 0.5,  # Moderate downsampling OK
    'optical_flow_dense': 0.25,  # Heavy downsampling recommended
    'motion_heatmap': 0.25,      # Heavy downsampling for memory
    'frequency_fft': 0.1,        # Very heavy downsampling
    'flow_hsv_viz': 0.5          # Moderate downsampling OK
}
```

### Parallel Processing

```python
from videokurt import VideoKurt

vk = VideoKurt(
    parallel=True,        # Run analyses in parallel
    max_workers=4        # Number of parallel workers
)

# Analyses will run concurrently
results = vk.analyze("video.mp4")
```

### Memory Management

```python
from videokurt import VideoKurt

vk = VideoKurt(
    chunk_size=100,      # Process video in 100-frame chunks
    clear_frames=True    # Clear frame buffer after analysis
)

# For very large videos
results = vk.analyze("large_video.mp4")
```

## Advanced Usage

### Custom Analysis Pipeline

```python
from videokurt import VideoKurt
from videokurt.analysis_models import ANALYSIS_REGISTRY

# Create custom pipeline
class MyVideoAnalyzer:
    def __init__(self):
        self.vk = VideoKurt()
        
    def analyze_for_motion(self, video_path):
        """Run only motion-related analyses"""
        motion_analyses = [
            'frame_diff',
            'optical_flow_dense', 
            'motion_heatmap',
            'background_mog2'
        ]
        return self.vk.analyze(video_path, analyses=motion_analyses)
    
    def analyze_for_structure(self, video_path):
        """Run only structure-related analyses"""
        structure_analyses = [
            'edge_canny',
            'contour_detection',
            'frequency_fft'
        ]
        return self.vk.analyze(video_path, analyses=structure_analyses)
```

### Accessing Raw Data

```python
results = vk.analyze("video.mp4", analyses=['optical_flow_dense'])

# Get raw flow field data
flow_data = results.analyses['optical_flow_dense'].data['flow_field']
# Shape: (num_frames-1, height, width, 2)
# Last dimension is (dx, dy) components

# Process the raw data
import numpy as np
magnitudes = np.sqrt(flow_data[..., 0]**2 + flow_data[..., 1]**2)
angles = np.arctan2(flow_data[..., 1], flow_data[..., 0])
```

### Saving Results

```python
import pickle
import json

results = vk.analyze("video.mp4")

# Save complete results
with open("results.pkl", "wb") as f:
    pickle.dump(results, f)

# Save metadata only
metadata = {
    'video': "video.mp4",
    'dimensions': results.dimensions,
    'fps': results.fps,
    'duration': results.duration,
    'analyses_run': list(results.analyses.keys()),
    'processing_time': results.elapsed_time
}
with open("metadata.json", "w") as f:
    json.dump(metadata, f)

# Save specific analysis data
np.save("frame_diffs.npy", results.analyses['frame_diff'].data['pixel_diff'])
```

## Error Handling

```python
from videokurt import VideoKurt
from videokurt.exceptions import AnalysisError, VideoLoadError

vk = VideoKurt()

try:
    results = vk.analyze("video.mp4")
except VideoLoadError as e:
    print(f"Could not load video: {e}")
except AnalysisError as e:
    print(f"Analysis failed: {e}")
    # Partial results may still be available
    if e.partial_results:
        print(f"Completed analyses: {e.partial_results.analyses.keys()}")
```

## Best Practices

1. **Start with heavy downsampling** (0.25 or less) to test quickly
2. **Run selective analyses** based on your needs, not all 11 every time
3. **Use frame_step** to skip frames for long videos
4. **Configure each analysis** based on your specific use case
5. **Monitor memory usage** with large videos or multiple analyses
6. **Save intermediate results** for long processing jobs

## Migration from Old VideoKurt

Old system:
```python
# Old way - semantic interpretation
results = videokurt.analyze("video.mp4")
timeline = results.timeline  # Binary activity timeline
segments = results.segments  # Semantic segments like "SCROLLING"
```

New system:
```python
# New way - raw analysis data
results = vk.analyze("video.mp4")
frame_diffs = results.analyses['frame_diff'].data['pixel_diff']
# Create your own interpretation from raw data
```

The new system provides raw data that you can interpret based on your specific needs, rather than forcing a particular semantic interpretation.