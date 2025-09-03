# Basic vs Middle vs Advanced Features in VideoKurt

## Architecture Overview
VideoKurt's architecture separates visual analysis into four layers:
```
Raw Analysis → Basic Features → Middle Features → Advanced Features
```

Each layer builds upon the previous, increasing in complexity and pattern sophistication.

### Raw Analyses Required
For the features described in this document, VideoKurt needs these raw analyses:
- **Currently Implemented**: frame_diff, edge_canny, optical_flow_dense/sparse, background_mog2/knn, motion_heatmap, frequency_fft, contour_detection
- **Proposed Additions**: color_histogram, dct_transform, texture_descriptors (see raw_analyses_explained.md)

## Classification Parameters

### Key Distinctions

| Parameter | Basic Features | Middle Features | Advanced Features |
|-----------|---------------|-----------------|-------------------|
| **Input Complexity** | Single array/frame | Multiple frames or masks | Multiple features + context |
| **Temporal Scope** | Per-frame or simple window | Multi-frame tracking | Extended sequences |
| **Spatial Awareness** | Pixel-level | Region/blob-level | Multi-region patterns |
| **Computation Type** | Statistics & thresholds | Segmentation & correspondence | Multi-feature pattern detection |
| **Domain Knowledge** | None | Minimal (physics) | Visual patterns (not semantic) |
| **Output Type** | Scalars & arrays | Structured data (blobs, tracks) | Pattern classifications |
| **Statefulness** | Stateless | Simple state (tracking) | Complex state (patterns) |
| **Interpretability** | "How much?" | "What and where?" | "What visual pattern?" |

## Basic Features
**Definition**: Direct mathematical operations on raw analysis outputs

### Characteristics
- Single-step calculations
- No object awareness
- Frame-independent (or simple temporal window)
- Pure numerical output
- Millisecond computation

### Examples
```python
# Binary Activity
pixel_diff = frame_diff_data['pixel_diff']
activity = np.mean(pixel_diff > threshold) > 0.1
# Output: 0 or 1

# Motion Magnitude  
flow = optical_flow_data['flow_field']
magnitude = np.sqrt(flow[...,0]**2 + flow[...,1]**2).mean()
# Output: 15.7 (scalar)

# Edge Density
edges = edge_data['edge_map']
density = np.sum(edges > 0) / edges.size
# Output: 0.23 (percentage)
```

### Complete List
1. Binary activity
2. Motion magnitude
3. Motion direction histogram
4. Edge density
5. Change regions (bounding boxes)
6. Stability score
7. Repetition indicator (FFT-based)
8. Foreground ratio
9. Frame difference percentile
10. Dominant flow vector
11. Histogram statistics (mean, peak, spread)
12. DCT energy (from coefficients)
13. Texture uniformity score

### Raw Analysis Dependencies
- Features 1-3, 9: Require `frame_diff`
- Features 4, 13: Require `edge_canny`
- Features 2, 10: Require `optical_flow_dense`
- Features 7: Requires `frequency_fft`
- Features 8: Requires `background_mog2/knn`
- Features 11: Requires `color_histogram` (proposed)
- Features 12: Requires `dct_transform` (proposed)
- Features 13: Requires `texture_descriptors` (proposed)

## Middle Features
**Definition**: Pattern extraction through segmentation, tracking, and spatial-temporal analysis

### Characteristics
- Multi-step processing with intermediate representations
- Region/blob awareness (but not semantic objects)
- Cross-frame correspondence
- Structured output (not just numbers)
- 10s of milliseconds computation

### Examples
```python
# Blob Count & Properties
foreground = background_subtraction_data['foreground_mask']
blobs = connected_components(foreground)
blob_count = len(blobs)
blob_sizes = [blob.area for blob in blobs]
# Output: {"count": 3, "sizes": [450, 230, 120]}

# Simple Trajectory
blob_centers = track_blobs_across_frames(blobs_over_time)
trajectory = smooth_path(blob_centers)
# Output: [(x1,y1,t1), (x2,y2,t2), ...]

# Zone-Based Activity
zones = define_spatial_zones(frame_shape)  # Predefined regions
zone_activity = {
    zone_id: measure_activity_in_region(activity_map, zone)
    for zone_id, zone in zones.items()
}
# Output: {"entrance": 0.8, "counter": 0.2, "exit": 0.5}
```

### Complete List
1. Blob count & properties
2. Blob stability (persistence)
3. Dwell time maps
4. Cross-frame blob tracking
5. Zone-based activity
6. Motion trajectories
7. Interaction zones (blob overlap)
8. Activity bursts
9. Periodicity strength
10. Boundary crossings
11. Spatial occupancy grid
12. Temporal activity patterns
13. Structural similarity (SSIM)
14. Perceptual hashes (from DCT)
15. Connected components analysis

## Advanced Features
**Definition**: Complex visual pattern detection through multi-feature analysis without semantic interpretation

### Characteristics
- Combines multiple basic/middle features
- Visual pattern classification (not object identification)
- Extended temporal pattern analysis
- Technical pattern labels (cut, fade, scroll)
- 100s of milliseconds+ computation
- Statistical or rule-based pattern detection

### Examples
```python
# Scene Boundary Detection
def detect_scene_cut(features):
    if (features['frame_diff'] > 0.7 and 
        features['edge_change'] > 0.5 and
        features['histogram_correlation'] < 0.3):
        return "hard_cut"
    elif features['motion_divergence'] > 0.8:
        return "zoom_transition"
    # Output: "hard_cut at frame 142"

# Scrolling Detection (Screen Recording)
def detect_scrolling(motion_histogram, flow_field):
    vertical_dominant = motion_histogram[90] > 0.7  # 90° = vertical
    consistent_direction = flow_variance < threshold
    if vertical_dominant and consistent_direction:
        return "vertical_scroll"
    # Output: "vertical_scroll from frame 50-120"

# UI Change Detection
def detect_ui_change(edge_density, structural_similarity):
    if (edge_density_change > 0.3 and
        structural_similarity < 0.6 and
        change_region.area > 0.4 * frame.area):
        return "major_ui_transition"
    # Output: "major_ui_transition at frame 200"
```

### Complete List
1. Scene boundaries (cuts, fades, wipes)
2. Camera movement classification (pan, zoom, tilt)
3. Scrolling detection (vertical, horizontal)
4. UI change detection (transitions, popups)
5. App/window switching detection
6. Motion pattern classification (linear, circular, chaotic)
7. Shot type detection (static, handheld, tracking)
8. Transition type detection (dissolve, swipe, fade)
9. Visual anomaly detection (statistical outliers)
10. Repetitive pattern classification
11. Motion coherence patterns (uniform vs scattered)
12. Structural change patterns

## Decision Tree for Feature Classification

```
Is it a direct calculation on pixels/arrays?
├─ YES → BASIC FEATURE
└─ NO → Does it track/segment regions?
    ├─ YES → Does it detect complex visual patterns?
    │   ├─ YES → ADVANCED FEATURE
    │   └─ NO → MIDDLE FEATURE
    └─ NO → Does it classify visual patterns?
        ├─ YES → ADVANCED FEATURE
        └─ NO → BASIC FEATURE (complex math)
```

## Practical Examples

### Example 1: Motion Analysis Progression
- **Raw**: Optical flow field (vectors)
- **Basic**: Motion magnitude (average vector length)
- **Middle**: Motion trajectories (tracked paths)
- **Advanced**: "Linear motion pattern with consistent velocity"

### Example 2: Change Detection Progression
- **Raw**: Frame differences (pixel arrays)
- **Basic**: Binary activity (0/1 per frame)
- **Middle**: Activity bursts (grouped active periods)
- **Advanced**: "Intermittent activity pattern with spatial clustering"

### Example 3: Spatial Analysis Progression
- **Raw**: Background subtraction mask
- **Basic**: Foreground ratio (percentage)
- **Middle**: Blob count and zones
- **Advanced**: "3 persistent blobs with distinct motion patterns"

## Implementation Guidelines

### Basic Features Should:
- Complete in <5ms per frame
- Use numpy operations directly
- Have no temporal state
- Return numbers or simple arrays

### Middle Features Should:
- Complete in <50ms per frame
- May maintain tracking state
- Use OpenCV morphology/contours
- Return structured data

### Advanced Features Should:
- May take 100ms+ per frame
- Maintain complex pattern detection state
- Apply visual pattern rules (not business rules)
- Return pattern classifications (not semantic meanings)

## Choosing the Right Level

| Use Case | Required Levels |
|----------|----------------|
| Simple motion detection | Raw + Basic |
| Activity monitoring | Raw + Basic + Middle |
| Scene detection | Raw + Basic + Advanced |
| UI automation testing | All four levels |
| Video summarization | Raw + Basic + Advanced |
| Blob tracking | Raw + Basic + Middle |

## Key Insights

1. **Basic features** answer "How much?" with numbers
2. **Middle features** answer "What structures?" with blobs and tracks
3. **Advanced features** answer "What visual pattern?" with classifications

The progression from Basic → Middle → Advanced represents increasing:
- Computational complexity
- Temporal awareness  
- Spatial pattern recognition
- Multi-feature integration
- Pattern sophistication

This separation allows users to:
- Stop at the appropriate level for their needs
- Debug issues at each layer
- Optimize performance by selective processing
- Build custom advanced features from middle features

## Important Distinction

**VideoKurt Advanced Features ARE:**
- Scene cut detection
- Scrolling pattern detection  
- Camera movement classification
- Visual anomaly detection

**VideoKurt Advanced Features ARE NOT:**
- Person identification
- Behavior interpretation (loitering, shopping)
- Business rule validation (hygiene, theft)
- Semantic understanding (customer needs help)

Advanced features detect **complex visual patterns**, not **semantic meanings**. The interpretation of what these patterns mean for a specific domain (retail, security, etc.) belongs to the application layer built on top of VideoKurt.