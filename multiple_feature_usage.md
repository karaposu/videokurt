# Multiple Feature Usage with VideoKurt

This document explains how to use multiple features together in VideoKurt for comprehensive video analysis.

## Table of Contents
1. [Basic Concepts](#basic-concepts)
2. [Adding Multiple Features](#adding-multiple-features)
3. [Dependency Management](#dependency-management)
4. [Common Feature Combinations](#common-feature-combinations)
5. [Performance Considerations](#performance-considerations)
6. [Best Practices](#best-practices)
7. [Examples](#examples)

## Basic Concepts

VideoKurt uses a builder pattern where you:
1. Create a VideoKurt instance
2. Add multiple features
3. Configure processing options
4. Run analysis once

All features share the same preprocessed frames and base analyses, making multi-feature analysis efficient.

## Adding Multiple Features

### Basic Syntax

```python
from videokurt import VideoKurt

vk = VideoKurt()

# Add multiple features
vk.add_feature('motion_magnitude')
vk.add_feature('stability_score')
vk.add_feature('scrolling_detection')
vk.add_feature('edge_density')

# Configure once
vk.configure(frame_step=2, resolution_scale=0.5)

# Analyze once - all features computed
results = vk.analyze('video.mp4')

# Access all feature results
motion = results.features['motion_magnitude'].data
stability = results.features['stability_score'].data
scrolling = results.features['scrolling_detection'].data
edges = results.features['edge_density'].data
```

### With Feature Parameters

Each feature can have its own parameters:

```python
vk = VideoKurt()

# Add features with specific configurations
vk.add_feature('motion_magnitude', normalize=True)
vk.add_feature('stability_score', window_size=5)
vk.add_feature('edge_density', threshold=50)
vk.add_feature('dwell_time_maps', decay_rate=0.9)
```

## Dependency Management

### Automatic Dependency Resolution

VideoKurt automatically adds required analyses when you add features:

```python
vk = VideoKurt()

# This feature requires 'optical_flow_dense'
vk.add_feature('motion_magnitude')

# This feature requires 'frame_diff'
vk.add_feature('stability_score')

# VideoKurt automatically adds both analyses
# No need to manually add them!
```

### Shared Dependencies

When multiple features need the same analysis, it's computed only once:

```python
vk = VideoKurt()

# Both need 'frame_diff'
vk.add_feature('stability_score')
vk.add_feature('binary_activity')
vk.add_feature('activity_bursts')

# frame_diff is computed once and shared
```

### Checking Dependencies

You can see what analyses will be run:

```python
vk = VideoKurt()
vk.add_feature('motion_trajectories')
vk.add_feature('scrolling_detection')

# Check what analyses are configured
print("Analyses to run:", list(vk._analyses.keys()))
# Output: ['optical_flow_sparse', 'optical_flow_dense', ...]
```

## Common Feature Combinations

### 1. Screen Recording Analysis

For comprehensive screen recording analysis:

```python
vk = VideoKurt()

# Core features for UI analysis
vk.add_feature('scrolling_detection')     # Detect scrolling
vk.add_feature('stability_score')         # Find idle periods
vk.add_feature('scene_detection')         # App/page switches
vk.add_feature('edge_density')           # Text vs media
vk.add_feature('motion_magnitude')       # Overall activity

# Spatial analysis
vk.add_feature('dwell_time_maps')        # Static regions
vk.add_feature('spatial_occupancy_grid') # Activity distribution

# Temporal patterns
vk.add_feature('activity_bursts')        # Intense periods
vk.add_feature('repetition_indicator')   # Loading/animations

vk.configure(frame_step=2, resolution_scale=0.5)
results = vk.analyze('screen_recording.mp4')
```

### 2. Motion Analysis Suite

For detailed motion analysis:

```python
vk = VideoKurt()

# Motion features
vk.add_feature('motion_magnitude')       # How much motion
vk.add_feature('dominant_flow_vector')   # Main direction
vk.add_feature('motion_trajectories')    # Path tracking
vk.add_feature('motion_direction_histogram') # Direction distribution

vk.configure(frame_step=1, resolution_scale=0.6)
results = vk.analyze('video.mp4')
```

### 3. Content Type Detection

To understand what type of content is in the video:

```python
vk = VideoKurt()

# Content classification features
vk.add_feature('edge_density')          # Text detection
vk.add_feature('texture_uniformity')    # Smooth vs detailed
vk.add_feature('histogram_statistics')  # Color distribution
vk.add_feature('dct_energy')           # Frequency content

vk.configure(frame_step=5, resolution_scale=0.5)
results = vk.analyze('video.mp4')
```

### 4. Temporal Analysis

For understanding patterns over time:

```python
vk = VideoKurt()

# Temporal features
vk.add_feature('temporal_activity_patterns')
vk.add_feature('periodicity_strength')
vk.add_feature('repetition_indicator')
vk.add_feature('activity_bursts')
vk.add_feature('structural_similarity')

vk.configure(frame_step=1, resolution_scale=0.5)
results = vk.analyze('video.mp4')
```

## Performance Considerations

### Processing Order

Features are computed in the order they were added, but all base analyses run first:

```python
# Execution order:
# 1. Load video
# 2. Preprocess frames
# 3. Run all analyses (frame_diff, optical_flow, etc.)
# 4. Compute features in order added
```

### Memory Usage

More features = more memory. Strategies to reduce memory:

```python
# Option 1: Lower resolution
vk.configure(resolution_scale=0.3)  # 30% of original

# Option 2: Skip frames
vk.configure(frame_step=5)  # Every 5th frame

# Option 3: Process in chunks (future feature)
vk.configure(process_chunks=4, chunk_overlap=30)
```

### Feature Compatibility

Some features work better together:

```python
# Good combinations (complementary information)
vk.add_feature('motion_magnitude')  # Amount of motion
vk.add_feature('stability_score')   # Inverse - stability

# Redundant combinations (similar information)
vk.add_feature('binary_activity')   # Simple activity
vk.add_feature('motion_magnitude')  # Detailed activity (includes above)
```

## Best Practices

### 1. Start Simple, Add Gradually

```python
# Start with core features
vk = VideoKurt()
vk.add_feature('motion_magnitude')
vk.add_feature('stability_score')
results = vk.analyze('video.mp4')

# Examine results, then add more as needed
```

### 2. Group Related Features

```python
def add_scrolling_analysis(vk):
    """Add features for scrolling detection."""
    vk.add_feature('scrolling_detection')
    vk.add_feature('dominant_flow_vector')
    vk.add_feature('motion_direction_histogram')

def add_stability_analysis(vk):
    """Add features for stability/idle detection."""
    vk.add_feature('stability_score')
    vk.add_feature('binary_activity')
    vk.add_feature('dwell_time_maps')

vk = VideoKurt()
add_scrolling_analysis(vk)
add_stability_analysis(vk)
```

### 3. Handle Missing Features Gracefully

```python
results = vk.analyze('video.mp4')

# Check if feature computed successfully
if 'motion_magnitude' in results.features:
    motion = results.features['motion_magnitude'].data
    print(f"Average motion: {motion.get('average', 0):.3f}")
else:
    print("Motion magnitude not computed")
```

### 4. Use Appropriate Configuration

Different videos need different settings:

```python
# For screen recordings (stable, high detail)
vk.configure(frame_step=2, resolution_scale=0.5)

# For action videos (fast motion)
vk.configure(frame_step=1, resolution_scale=0.7)

# For long videos (reduce processing)
vk.configure(frame_step=10, resolution_scale=0.3)
```

## Examples

### Example 1: Complete UI Analysis

```python
from videokurt import VideoKurt
import numpy as np

def analyze_screen_recording(video_path):
    """Comprehensive screen recording analysis."""
    
    vk = VideoKurt()
    
    # Add all relevant features for UI analysis
    features = [
        'scrolling_detection',
        'stability_score',
        'scene_detection',
        'edge_density',
        'motion_magnitude',
        'dwell_time_maps',
        'spatial_occupancy_grid',
        'activity_bursts',
        'repetition_indicator'
    ]
    
    for feature in features:
        vk.add_feature(feature)
    
    # Optimize for screen recordings
    vk.configure(frame_step=2, resolution_scale=0.5)
    
    # Run analysis
    print(f"Analyzing {video_path}...")
    results = vk.analyze(video_path)
    
    # Process results
    report = {}
    
    # Check scrolling
    if 'scrolling_detection' in results.features:
        scroll_data = results.features['scrolling_detection'].data
        report['has_scrolling'] = scroll_data.get('is_scrolling', False)
        report['scroll_direction'] = scroll_data.get('direction', 'none')
    
    # Check stability
    if 'stability_score' in results.features:
        stability = results.features['stability_score'].data
        report['avg_stability'] = np.mean(stability.get('timeline', []))
        report['idle_ratio'] = sum(s > 0.9 for s in stability.get('timeline', [])) / len(stability.get('timeline', [1]))
    
    # Check scene changes
    if 'scene_detection' in results.features:
        scenes = results.features['scene_detection'].data
        report['num_scenes'] = scenes.get('num_scenes', 1)
    
    # Activity level
    if 'motion_magnitude' in results.features:
        motion = results.features['motion_magnitude'].data
        report['activity_level'] = motion.get('average', 0)
    
    return report

# Use it
report = analyze_screen_recording('recording.mp4')
print("Analysis Report:")
for key, value in report.items():
    print(f"  {key}: {value}")
```

### Example 2: Comparative Analysis

```python
def compare_videos(video1_path, video2_path):
    """Compare two videos using multiple features."""
    
    results = []
    
    for video_path in [video1_path, video2_path]:
        vk = VideoKurt()
        
        # Add comparison features
        vk.add_feature('motion_magnitude')
        vk.add_feature('edge_density')
        vk.add_feature('stability_score')
        vk.add_feature('repetition_indicator')
        
        vk.configure(frame_step=5, resolution_scale=0.5)
        result = vk.analyze(video_path)
        results.append(result)
    
    # Compare results
    comparison = {}
    
    for feature in ['motion_magnitude', 'edge_density', 'stability_score']:
        if feature in results[0].features and feature in results[1].features:
            val1 = results[0].features[feature].data.get('average', 0)
            val2 = results[1].features[feature].data.get('average', 0)
            comparison[feature] = {
                'video1': val1,
                'video2': val2,
                'difference': val2 - val1
            }
    
    return comparison
```

### Example 3: Feature Correlation

```python
def analyze_feature_correlation(video_path):
    """Check how different features correlate."""
    
    vk = VideoKurt()
    
    # Add potentially correlated features
    vk.add_feature('motion_magnitude')
    vk.add_feature('stability_score')
    vk.add_feature('binary_activity')
    vk.add_feature('edge_density')
    
    vk.configure(frame_step=1, resolution_scale=0.5)
    results = vk.analyze(video_path)
    
    # Extract timelines
    timelines = {}
    
    if 'motion_magnitude' in results.features:
        timelines['motion'] = results.features['motion_magnitude'].data.get('timeline', [])
    
    if 'stability_score' in results.features:
        timelines['stability'] = results.features['stability_score'].data.get('timeline', [])
    
    if 'binary_activity' in results.features:
        timelines['activity'] = results.features['binary_activity'].data.get('timeline', [])
    
    # Check correlation
    if 'motion' in timelines and 'stability' in timelines:
        # Should be inversely correlated
        correlation = np.corrcoef(timelines['motion'], timelines['stability'])[0, 1]
        print(f"Motion-Stability correlation: {correlation:.3f}")
        
    return timelines
```

## Feature Dependencies Reference

Quick reference for what each feature needs:

| Feature | Required Analyses |
|---------|------------------|
| motion_magnitude | optical_flow_dense |
| scrolling_detection | optical_flow_dense |
| stability_score | frame_diff |
| binary_activity | frame_diff |
| edge_density | edge_canny |
| dwell_time_maps | frame_diff |
| scene_detection | color_histogram |
| motion_trajectories | optical_flow_sparse |
| activity_bursts | frame_diff |
| structural_similarity | frame_diff |

## Troubleshooting

### Features Not Computing

If a feature doesn't appear in results:
1. Check if required analyses ran successfully
2. Look for error messages during analysis
3. Verify video has appropriate content for the feature

### Memory Issues

If running out of memory:
1. Reduce resolution_scale (0.3 or lower)
2. Increase frame_step (process fewer frames)
3. Use fewer features
4. Process shorter video segments

### Slow Processing

To speed up analysis:
1. Use frame_step > 1
2. Lower resolution_scale
3. Remove unnecessary features
4. Avoid redundant features

## Summary

Using multiple features in VideoKurt is straightforward:
- Add all desired features with `add_feature()`
- Configure once with `configure()`
- Analyze once with `analyze()`
- Access all results from the returned object

The system handles dependencies automatically and shares computations efficiently, making multi-feature analysis both powerful and performant.