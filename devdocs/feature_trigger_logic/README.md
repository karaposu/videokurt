# Feature Trigger Logic Documentation

This directory contains detailed documentation about what triggers detection for each VideoKurt feature, particularly focused on screen recording scenarios.

## Purpose

Understanding what triggers each feature helps:
- Select the right features for your analysis
- Interpret results correctly
- Avoid features that don't work well with screen recordings
- Optimize parameters for your use case

## Feature Categories

### 🎯 High-Value for Screen Recordings

These features work excellently with screen recordings:

- [**scrolling_detection.md**](scrolling_detection.md) - Purpose-built for UI scrolling patterns
- [**stability_score.md**](stability_score.md) - Detects idle periods and reading time
- [**scene_detection.md**](scene_detection.md) - Identifies application/page switches
- [**dwell_time_maps.md**](dwell_time_maps.md) - Shows static vs dynamic regions
- [**edge_density.md**](edge_density.md) - Distinguishes text from media content
- [**repetition_indicator.md**](repetition_indicator.md) - Finds loading states and loops
- [**temporal_activity_patterns.md**](temporal_activity_patterns.md) - Reveals work rhythms

### 📊 General Activity Metrics

Universal features that work across all video types:

- [**motion_magnitude.md**](motion_magnitude.md) - Overall activity level
- [**binary_activity.md**](binary_activity.md) - Simple active/idle detection
- [**activity_bursts.md**](activity_bursts.md) - Intense activity periods
- [**spatial_occupancy_grid.md**](spatial_occupancy_grid.md) - Where activity occurs

### 🎥 Motion Analysis

Features that track movement patterns:

- [**motion_trajectories.md**](motion_trajectories.md) - Path tracking
- [**boundary_crossing.md**](boundary_crossing.md) - Objects crossing lines

### ⚠️ Limited Use for Screen Recordings

These features are designed for physical video and have limited utility for screen recordings:

- [**blob_tracking.md**](blob_tracking.md) - Tracks "objects" (poor for UI)
- [**blob_stability.md**](blob_stability.md) - Object persistence (noisy for UI)

## Quick Reference Table

| Feature | Screen Recording Value | Primary Use Case |
|---------|----------------------|------------------|
| Scrolling Detection | ⭐⭐⭐⭐⭐ | Detecting scroll patterns |
| Stability Score | ⭐⭐⭐⭐⭐ | Finding idle/reading periods |
| Scene Detection | ⭐⭐⭐⭐⭐ | App/page switches |
| Dwell Time Maps | ⭐⭐⭐⭐ | UI stability analysis |
| Edge Density | ⭐⭐⭐⭐ | Text vs media detection |
| Motion Magnitude | ⭐⭐⭐⭐ | General activity level |
| Temporal Patterns | ⭐⭐⭐⭐ | Work rhythm analysis |
| Repetition Indicator | ⭐⭐⭐ | Loading/loop detection |
| Activity Bursts | ⭐⭐⭐ | Intense activity periods |
| Spatial Occupancy | ⭐⭐⭐ | Activity distribution |
| Binary Activity | ⭐⭐⭐ | Simple idle detection |
| Motion Trajectories | ⭐⭐ | Drag operations |
| Boundary Crossing | ⭐⭐ | Limited UI scenarios |
| Blob Tracking | ⭐ | Poor for UI |
| Blob Stability | ⭐ | Very noisy for UI |

## How to Use This Documentation

1. **Start with your goal** - What do you want to detect?
2. **Check high-value features first** - These are optimized for screen recordings
3. **Read trigger conditions** - Understand what activates detection
4. **Review limitations** - Know what won't work
5. **Check output format** - Understand what data you'll get

## Common Screen Recording Patterns

### Reading/Reviewing Content
- **High**: Stability Score, Dwell Time
- **Low**: Motion Magnitude, Binary Activity
- **Use**: Detect reading time and focus areas

### Active Navigation
- **High**: Scene Detection, Activity Bursts
- **Variable**: Scrolling Detection
- **Use**: Track user journey and context switches

### Content Creation
- **High**: Edge Density (for text), Temporal Patterns
- **Medium**: Spatial Occupancy
- **Use**: Understand work patterns and productivity

### Video/Media Consumption
- **Low**: Edge Density, Stability Score
- **High**: Motion Magnitude (continuous)
- **Use**: Detect media playback periods

## Feature Selection Guide

### For User Behavior Analysis
1. Temporal Activity Patterns
2. Stability Score
3. Scene Detection
4. Scrolling Detection

### For UI/UX Analysis
1. Dwell Time Maps
2. Spatial Occupancy Grid
3. Edge Density
4. Scrolling Detection

### For Performance Analysis
1. Motion Magnitude
2. Activity Bursts
3. Repetition Indicator
4. Stability Score

### For Content Classification
1. Edge Density
2. Scene Detection
3. Stability Score
4. Motion Magnitude

## Notes

- Most features work better with screen recordings when combined
- Single features rarely tell the complete story
- Thresholds often need adjustment for screen content
- Resolution and frame rate affect many features
- Compression artifacts can trigger false positives