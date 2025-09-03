# List of All Analysis Methods for VideoKurt

## Overview

This document lists all available analysis methods ordered from simplest to most advanced. These analyses are the foundation of VideoKurt's feature extraction pipeline.

---

## 1. Frame Differencing (Simple)

**Method Name**: `frame_diff`

**What it does**: Computes pixel-wise differences between consecutive frames

**Implementation**: Currently in VideoKurt core

```python
Raw Output:
    pixel_diff: np.ndarray  # [T, H, W] - |frame[t] - frame[t-1]|
                           # Absolute difference per pixel
```

**Use when**:
- Quick activity detection
- Binary active/idle classification
- Detecting any change

**Pros**: Very fast, simple, no parameters
**Cons**: Sensitive to noise, no direction information

---

## 2. Edge Detection (Canny)

**Method Name**: `edge_canny`

**What it does**: Detects edges/boundaries in images

**Implementation**: Not yet implemented

```python
Raw Output:
    edge_map: np.ndarray  # [T, H, W] - binary edge pixels (0 or 255)
                         # Detected using Canny edge detector
    gradient_magnitude: np.ndarray  # [T, H, W] - edge strength per pixel
    gradient_direction: np.ndarray  # [T, H, W] - edge orientation per pixel
```

**Use when**:
- Detecting text regions (high edge density)
- Finding UI boundaries
- Measuring visual complexity

**Pros**: Fast, well-understood, structural information
**Cons**: Sensitive to parameters, only structural info

---

## 3. Advanced Frame Differencing

**Method Name**: `frame_diff_advanced`

**What it does**: Multiple differencing techniques for richer change detection

**Implementation**: `explorations/05_advanced_frame_differencing.py`

```python
Raw Output:
    triple_diff: np.ndarray        # [T, H, W] - (f[t]-f[t-1])-(f[t-1]-f[t-2])
                                   # Second derivative, shows acceleration
    running_avg_diff: np.ndarray   # [T, H, W] - frame vs running average
                                   # Difference from temporal average
    accumulated_diff: np.ndarray   # [H, W] - cumulative differences over time
                                   # Shows persistent activity regions
```

**Use when**:
- Detecting motion acceleration/deceleration
- Finding start/stop of scrolling
- Distinguishing sustained activity from brief changes

**Pros**: Detects motion patterns, not just motion
**Cons**: More complex, requires multiple frame buffers

---

## 4. Contour Detection

**Method Name**: `contour_detection`

**What it does**: Finds and tracks shape boundaries in each frame

**Implementation**: `explorations/04_contour_detection.py`

```python
Raw Output:
    contours: List[List[np.ndarray]]  # [T][n] - list of contour points
                                      # Each contour is array of (x,y) points
    hierarchy: List[np.ndarray]       # [T][n][4] - contour relationships
                                      # [Next, Previous, First_Child, Parent]
```

**Use when**:
- Counting distinct UI elements
- Detecting shape changes
- Finding rectangular regions (UI elements)

**Pros**: Good for structured content, shape analysis
**Cons**: Sensitive to threshold, no temporal coherence

---

## 5. Background Subtraction (MOG2)

**Method Name**: `background_mog2`

**What it does**: Learns background model and detects foreground objects

**Implementation**: `explorations/03_background_subtraction.py`

```python
Raw Output:
    foreground_mask: np.ndarray    # [T, H, W] - binary mask (0=bg, 255=fg)
                                   # Learned background model separates fg/bg
```

**Use when**:
- Detecting new elements appearing
- Separating moving objects from static background
- Finding UI popups/overlays

**Pros**: Adapts to gradual changes, good segmentation
**Cons**: Needs learning period (~30 frames), can be fooled by slow motion

---

## 6. Background Subtraction (KNN)

**Method Name**: `background_knn`

**What it does**: K-nearest neighbors background model (alternative to MOG2)

**Implementation**: `explorations/03_background_subtraction.py`

```python
Raw Output:
    foreground_mask: np.ndarray    # [T, H, W] - binary mask (0=bg, 255=fg)
                                   # K-NN based, slower adaptation than MOG2
```

**Use when**:
- Need slower adaptation than MOG2
- Want to detect changes that persist longer

**Pros**: More stable, less prone to rapid adaptation
**Cons**: Slower to adapt to scene changes

---

## 7. Optical Flow - Lucas-Kanade (Sparse)

**Method Name**: `optical_flow_sparse`

**What it does**: Tracks specific feature points (corners, edges) across frames

**Implementation**: `explorations/02_optical_flow_lucas_kanade.py`

```python
Raw Output:
    tracked_points: List[TrackedPoint]  # [(id, x, y, dx, dy), ...]
                                        # Per frame list of tracked features
    point_status: np.ndarray            # [n_points] - tracking success/failure
```

**Use when**:
- Tracking specific UI elements
- Need to distinguish objects moving differently
- Detecting if elements move uniformly or independently

**Pros**: Efficient, tracks specific features, maintains point identity
**Cons**: Can lose tracking, sparse information only

---

## 8. Optical Flow - Farneback (Dense)

**Method Name**: `optical_flow_dense`

**What it does**: Calculates motion vectors for EVERY pixel between frames

**Implementation**: `explorations/01_optical_flow_farneback.py`

```python
Raw Output:
    flow_field: np.ndarray  # [T, H, W, 2] - (dx, dy) for every pixel
                           # Contains complete motion information:
                           # - dx: horizontal displacement per pixel
                           # - dy: vertical displacement per pixel
                           # - magnitude = sqrt(dx² + dy²)
                           # - direction = arctan2(dy, dx)
```

**Use when**: 
- Need complete motion information
- Detecting scrolling/panning patterns
- Measuring motion intensity across entire frame

**Pros**: Complete motion information, good for scrolling detection
**Cons**: Computationally expensive, noisy on texture-less regions

---

## 9. Motion Heatmap

**Method Name**: `motion_heatmap`

**What it does**: Accumulates motion over time to find activity zones

**Implementation**: `explorations/07_motion_heatmap.py`

```python
Raw Output:
    cumulative_heatmap: np.ndarray    # [H, W] - sum of motion over all frames
    weighted_heatmap: np.ndarray      # [H, W] - recent motion weighted by decay
    heatmap_snapshots: List[Tuple[float, np.ndarray]]  # [(timestamp, heatmap), ...]
                                     # Periodic snapshots of activity
```

**Use when**:
- Finding which screen areas are most active
- Identifying scrollable regions vs static regions
- Creating video summaries

**Pros**: Spatial understanding of activity patterns
**Cons**: Loses temporal detail, memory intensive

---

## 10. Frequency Analysis (FFT)

**Method Name**: `frequency_fft`

**What it does**: Analyzes temporal frequency of pixel changes

**Implementation**: Partially in explorations

```python
Raw Output:
    frequency_spectrum: np.ndarray    # [T, n_freq] or [H, W, n_freq]
                                     # FFT magnitude at each frequency
    phase_spectrum: np.ndarray        # [T, n_freq] or [H, W, n_freq]
                                     # FFT phase at each frequency
```

**Use when**:
- Detecting animations/videos
- Finding refresh rates
- Identifying periodic UI updates

**Pros**: Detects patterns invisible to other methods
**Cons**: Requires temporal window, complex interpretation

---

## 11. HSV Flow Visualization

**Method Name**: `flow_hsv_viz`

**What it does**: Converts optical flow to color representation for visualization

**Implementation**: `explorations/06_flow_visualization_hsv.py`

```python
Raw Output:
    hsv_flow: np.ndarray  # [T, H, W, 3] - HSV color representation
                         # H (Hue): motion direction (0-360°)
                         # S (Saturation): confidence/consistency  
                         # V (Value): motion magnitude/speed
```

**Use when**:
- Debugging motion detection
- Creating visual summaries
- Understanding flow patterns

**Pros**: Intuitive visualization
**Cons**: Mainly diagnostic, not for feature extraction

---

## Complexity Progression

### Level 1: Basic (Real-time capable)
1. Frame Differencing
2. Edge Detection

### Level 2: Intermediate 
3. Advanced Frame Differencing
4. Contour Detection

### Level 3: Advanced
5. Background Subtraction (MOG2)
6. Background Subtraction (KNN)
7. Lucas-Kanade Optical Flow

### Level 4: Complex (Computationally intensive)
8. Farneback Optical Flow
9. Motion Heatmap
10. Frequency Analysis
11. HSV Flow Visualization

---

## Recommended Default Pipeline

For most use cases, combine analyses from different levels:

1. **Frame Differencing** (Level 1) - Quick activity detection
2. **Contour Detection** (Level 2) - Structure analysis
3. **Background Subtraction MOG2** (Level 3) - New element detection  
4. **Farneback Optical Flow** (Level 4) - Motion patterns

This provides a good balance of speed, accuracy, and feature richness.

---

## Performance Considerations

### Processing Speed (on 1920x1080 @ 30fps):
- **Frame Differencing**: ~2ms per frame
- **Edge Detection**: ~5ms per frame
- **Contour Detection**: ~10ms per frame
- **Background Subtraction**: ~15ms per frame
- **Lucas-Kanade**: ~20ms per frame (100 points)
- **Farneback Flow**: ~100ms per frame
- **Motion Heatmap**: ~50ms per frame
- **Frequency Analysis**: ~200ms per window

### Memory Requirements:
- **Basic** (Diff, Edge): ~10MB per second
- **Intermediate** (Contours, BG): ~50MB per second  
- **Advanced** (Optical Flow): ~200MB per second
- **Complex** (Heatmaps, FFT): ~500MB per second

---

## Method Name Reference

Quick reference for all analysis method names used as keys in the results dictionary:

| Method Name | Analysis Type | Complexity Level |
|------------|---------------|------------------|
| `frame_diff` | Frame Differencing (Simple) | Level 1 |
| `edge_canny` | Edge Detection (Canny) | Level 1 |
| `frame_diff_advanced` | Advanced Frame Differencing | Level 2 |
| `contour_detection` | Contour Detection | Level 2 |
| `background_mog2` | Background Subtraction (MOG2) | Level 3 |
| `background_knn` | Background Subtraction (KNN) | Level 3 |
| `optical_flow_sparse` | Optical Flow Lucas-Kanade | Level 3 |
| `optical_flow_dense` | Optical Flow Farneback | Level 4 |
| `motion_heatmap` | Motion Heatmap | Level 4 |
| `frequency_fft` | Frequency Analysis (FFT) | Level 4 |
| `flow_hsv_viz` | HSV Flow Visualization | Level 4 |

### Usage Example:
```python
# Run specific analyses
results = vk.analyze_video("video.mp4", 
    analyses=['frame_diff', 'optical_flow_dense', 'motion_heatmap'])

# Access results
if results.has_analysis('optical_flow_dense'):
    flow = results.get_analysis('optical_flow_dense')
```