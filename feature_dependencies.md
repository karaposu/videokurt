# Feature Dependencies

This document defines the hardcoded analysis requirements for each VideoKurt feature. These dependencies have been chosen to provide the best balance of accuracy and performance for general use cases.

## Quick Reference Table

| Feature | Required Analyses | Speed | Accuracy |
|---------|------------------|-------|----------|
| binary_activity | frame_diff | Fast (2ms) | High |
| motion_magnitude | optical_flow_sparse | Medium (10ms) | High |
| edge_density | edge_canny | Medium (8ms) | High |
| change_regions | frame_diff, contour_detection | Medium (12ms) | Medium |
| stability_score | frame_diff, optical_flow_sparse | Medium (15ms) | High |
| foreground_ratio | background_mog2 | Fast (5ms) | Medium |
| scene_detection | frame_diff, edge_canny, color_histogram | Medium (20ms) | High |
| scrolling_detection | optical_flow_dense, edge_canny | Slow (50ms) | High |
| ui_change_detection | frame_diff, edge_canny, contour_detection | Medium (25ms) | High |
| camera_movement | optical_flow_sparse, frame_diff | Medium (15ms) | High |
| activity_bursts | frame_diff | Fast (3ms) | High |
| motion_trajectories | optical_flow_sparse | Medium (12ms) | High |

---

## Basic Features

### binary_activity
**Required:** `frame_diff`  
**Strategy:** Threshold-based pixel change detection  
**Performance:** ~2ms per frame  
**Use Case:** Activity timeline, idle detection

### motion_magnitude
**Required:** `optical_flow_sparse`  
**Strategy:** Average magnitude of tracked feature points  
**Performance:** ~10ms per frame  
**Use Case:** Motion intensity measurement

### motion_direction_histogram
**Required:** `optical_flow_dense`  
**Strategy:** Histogram of flow field directions  
**Performance:** ~30ms per frame  
**Use Case:** Dominant motion direction analysis

### edge_density
**Required:** `edge_canny`  
**Strategy:** Ratio of edge pixels to total pixels  
**Performance:** ~8ms per frame  
**Use Case:** Scene complexity, text detection

### change_regions
**Required:** `frame_diff`, `contour_detection`  
**Strategy:** Find contours in difference maps  
**Performance:** ~12ms per frame  
**Use Case:** Localized change detection

### stability_score
**Required:** `frame_diff`, `optical_flow_sparse`  
**Strategy:** Combined motion and change metrics  
**Performance:** ~15ms per frame  
**Use Case:** Camera shake detection

### repetition_indicator
**Required:** `frame_diff`, `frequency_fft`  
**Strategy:** FFT analysis of temporal changes  
**Performance:** ~40ms per window  
**Use Case:** Loop detection, periodic motion

### foreground_ratio
**Required:** `background_mog2` OR `background_knn`  
**Strategy:** Ratio of foreground pixels  
**Performance:** ~5ms per frame  
**Use Case:** Object presence detection

### frame_difference_percentile
**Required:** `frame_diff`  
**Strategy:** Statistical analysis of differences  
**Performance:** ~3ms per frame  
**Use Case:** Adaptive thresholding

### dominant_flow_vector
**Required:** `optical_flow_dense`  
**Strategy:** Weighted average of flow field  
**Performance:** ~35ms per frame  
**Use Case:** Global motion estimation

### histogram_statistics
**Required:** `color_histogram`  
**Strategy:** Statistical moments of color distribution  
**Performance:** ~5ms per frame  
**Use Case:** Color consistency analysis

### dct_energy
**Required:** `dct_transform`  
**Strategy:** Sum of DCT coefficient magnitudes  
**Performance:** ~8ms per frame  
**Use Case:** Compression quality estimation

### texture_uniformity
**Required:** `texture_descriptors`  
**Strategy:** LBP histogram entropy  
**Performance:** ~10ms per frame  
**Use Case:** Texture classification

---

## Middle Features

### activity_bursts
**Required:** `frame_diff`  
**Strategy:** Temporal clustering of high activity periods  
**Performance:** ~3ms per frame  
**Use Case:** Event detection, highlight extraction

### blob_tracking
**Required:** `contour_detection`, `background_mog2`  
**Strategy:** Track contours across frames  
**Performance:** ~20ms per frame  
**Use Case:** Object tracking

### blob_stability
**Required:** `contour_detection`  
**Strategy:** Consistency of contour properties  
**Performance:** ~15ms per frame  
**Use Case:** Static object detection

### dwell_time_maps
**Required:** `optical_flow_sparse`, `motion_heatmap`  
**Strategy:** Accumulate presence over time  
**Performance:** ~8ms per frame  
**Use Case:** Attention analysis

### zone_based_activity
**Required:** `frame_diff`, `motion_heatmap`  
**Strategy:** Divide frame into zones, measure activity  
**Performance:** ~5ms per frame  
**Use Case:** Screen region analysis

### motion_trajectories
**Required:** `optical_flow_sparse`  
**Strategy:** Link tracked points across frames  
**Performance:** ~12ms per frame  
**Use Case:** Path analysis

### interaction_zones
**Required:** `motion_heatmap`, `contour_detection`  
**Strategy:** Identify frequently active regions  
**Performance:** ~10ms per frame  
**Use Case:** UI hotspot detection

### periodicity_strength
**Required:** `frequency_fft`, `frame_diff`  
**Strategy:** FFT peak detection  
**Performance:** ~45ms per window  
**Use Case:** Repetitive motion detection

### boundary_crossings
**Required:** `optical_flow_sparse`, `contour_detection`  
**Strategy:** Track points crossing defined boundaries  
**Performance:** ~15ms per frame  
**Use Case:** Entry/exit detection

### spatial_occupancy_grid
**Required:** `background_mog2`, `contour_detection`  
**Strategy:** Grid-based presence tracking  
**Performance:** ~12ms per frame  
**Use Case:** Space utilization

### temporal_activity_patterns
**Required:** `frame_diff`, `frequency_fft`  
**Strategy:** Time-series analysis of activity  
**Performance:** ~20ms per window  
**Use Case:** Behavioral patterns

### structural_similarity
**Required:** `edge_canny`, `texture_descriptors`  
**Strategy:** SSIM with edge and texture features  
**Performance:** ~15ms per frame  
**Use Case:** Quality assessment

### perceptual_hashes
**Required:** `dct_transform`  
**Strategy:** Binary hash from DCT coefficients  
**Performance:** ~8ms per frame  
**Use Case:** Duplicate detection

### connected_components
**Required:** `frame_diff`, `contour_detection`  
**Strategy:** Label and track connected regions  
**Performance:** ~10ms per frame  
**Use Case:** Object segmentation

---

## Advanced Features

### scene_detection
**Required:** `frame_diff`, `edge_canny`, `color_histogram`  
**Strategy:** Multi-cue boundary detection with adaptive thresholds  
**Performance:** ~20ms per frame  
**Use Case:** Video segmentation, chapter detection

### camera_movement
**Required:** `optical_flow_sparse`, `frame_diff`  
**Strategy:** Global motion estimation from feature tracks  
**Performance:** ~15ms per frame  
**Use Case:** Stabilization, cinematography analysis

### scrolling_detection
**Required:** `optical_flow_dense`, `edge_canny`  
**Strategy:** Detect uniform vertical/horizontal motion patterns  
**Performance:** ~50ms per frame  
**Use Case:** UI automation, reading behavior

### ui_change_detection
**Required:** `frame_diff`, `edge_canny`, `contour_detection`  
**Strategy:** Structural change detection with region analysis  
**Performance:** ~25ms per frame  
**Use Case:** Automated UI testing

### app_window_switching
**Required:** `frame_diff`, `edge_canny`, `color_histogram`  
**Strategy:** Detect large-scale layout changes  
**Performance:** ~18ms per frame  
**Use Case:** Workflow analysis

### motion_pattern_classification
**Required:** `optical_flow_dense`, `motion_trajectories`  
**Strategy:** Classify motion into categories (linear/circular/chaotic)  
**Performance:** ~40ms per frame  
**Use Case:** Behavior classification

### shot_type_detection
**Required:** `edge_canny`, `contour_detection`, `texture_descriptors`  
**Strategy:** Classify shot scale based on visual features  
**Performance:** ~22ms per frame  
**Use Case:** Film analysis

### transition_type_detection
**Required:** `frame_diff`, `color_histogram`, `edge_canny`  
**Strategy:** Identify transition effects (cut/fade/wipe)  
**Performance:** ~15ms per frame  
**Use Case:** Video editing analysis

### visual_anomaly_detection
**Required:** `frame_diff`, `dct_transform`, `texture_descriptors`  
**Strategy:** Statistical outlier detection in feature space  
**Performance:** ~30ms per frame  
**Use Case:** Quality control

### repetitive_pattern_classification
**Required:** `frequency_fft`, `motion_heatmap`  
**Strategy:** Identify and classify periodic patterns  
**Performance:** ~35ms per window  
**Use Case:** Animation detection

### motion_coherence_patterns
**Required:** `optical_flow_dense`, `contour_detection`  
**Strategy:** Group pixels with similar motion  
**Performance:** ~45ms per frame  
**Use Case:** Object segmentation

### structural_change_patterns
**Required:** `edge_canny`, `contour_detection`, `texture_descriptors`  
**Strategy:** Track structural element changes  
**Performance:** ~28ms per frame  
**Use Case:** Layout change detection

---

## Performance Notes

### Speed Categories
- **Fast**: < 10ms per frame
- **Medium**: 10-30ms per frame  
- **Slow**: > 30ms per frame

### Memory Impact
- **Low**: frame_diff, edge_canny, color_histogram
- **Medium**: optical_flow_sparse, contour_detection, background_mog2
- **High**: optical_flow_dense, frequency_fft, motion_heatmap

### Optimization Tips
1. Features sharing analyses are computed together efficiently
2. Heavy analyses (optical_flow_dense) are cached and reused
3. Use frame_step and resolution_scale to reduce computation
4. Group features by their analysis requirements

---

## Dependency Graph

### Most Common Dependencies
1. `frame_diff` - Used by 15+ features
2. `edge_canny` - Used by 10+ features
3. `contour_detection` - Used by 10+ features
4. `optical_flow_sparse` - Used by 8+ features
5. `color_histogram` - Used by 5+ features

### Standalone Analyses
These are rarely used alone:
- `flow_hsv_viz` - Only for visualization
- `frame_diff_advanced` - Specialized use cases

### Heavy Combinations
These feature combinations are expensive:
- `scrolling_detection` + `motion_pattern_classification` (both need optical_flow_dense)
- `repetition_indicator` + `periodicity_strength` (both need frequency_fft)

---

## Implementation Code Reference

```python
# This is hardcoded in the feature implementations
FEATURE_DEPENDENCIES = {
    # Basic features
    'binary_activity': ['frame_diff'],
    'motion_magnitude': ['optical_flow_sparse'],
    'motion_direction_histogram': ['optical_flow_dense'],
    'edge_density': ['edge_canny'],
    'change_regions': ['frame_diff', 'contour_detection'],
    'stability_score': ['frame_diff', 'optical_flow_sparse'],
    'repetition_indicator': ['frame_diff', 'frequency_fft'],
    'foreground_ratio': ['background_mog2'],
    'frame_difference_percentile': ['frame_diff'],
    'dominant_flow_vector': ['optical_flow_dense'],
    'histogram_statistics': ['color_histogram'],
    'dct_energy': ['dct_transform'],
    'texture_uniformity': ['texture_descriptors'],
    
    # Middle features
    'activity_bursts': ['frame_diff'],
    'blob_tracking': ['contour_detection', 'background_mog2'],
    'blob_stability': ['contour_detection'],
    'dwell_time_maps': ['optical_flow_sparse', 'motion_heatmap'],
    'zone_based_activity': ['frame_diff', 'motion_heatmap'],
    'motion_trajectories': ['optical_flow_sparse'],
    'interaction_zones': ['motion_heatmap', 'contour_detection'],
    'periodicity_strength': ['frequency_fft', 'frame_diff'],
    'boundary_crossings': ['optical_flow_sparse', 'contour_detection'],
    'spatial_occupancy_grid': ['background_mog2', 'contour_detection'],
    'temporal_activity_patterns': ['frame_diff', 'frequency_fft'],
    'structural_similarity': ['edge_canny', 'texture_descriptors'],
    'perceptual_hashes': ['dct_transform'],
    'connected_components': ['frame_diff', 'contour_detection'],
    
    # Advanced features
    'scene_detection': ['frame_diff', 'edge_canny', 'color_histogram'],
    'camera_movement': ['optical_flow_sparse', 'frame_diff'],
    'scrolling_detection': ['optical_flow_dense', 'edge_canny'],
    'ui_change_detection': ['frame_diff', 'edge_canny', 'contour_detection'],
    'app_window_switching': ['frame_diff', 'edge_canny', 'color_histogram'],
    'motion_pattern_classification': ['optical_flow_dense', 'motion_trajectories'],
    'shot_type_detection': ['edge_canny', 'contour_detection', 'texture_descriptors'],
    'transition_type_detection': ['frame_diff', 'color_histogram', 'edge_canny'],
    'visual_anomaly_detection': ['frame_diff', 'dct_transform', 'texture_descriptors'],
    'repetitive_pattern_classification': ['frequency_fft', 'motion_heatmap'],
    'motion_coherence_patterns': ['optical_flow_dense', 'contour_detection'],
    'structural_change_patterns': ['edge_canny', 'contour_detection', 'texture_descriptors'],
}
```

## Usage Guidelines

1. **Start with basic features** - They're fast and often sufficient
2. **Add advanced features selectively** - They're expensive
3. **Group related features** - They share analyses
4. **Monitor performance** - Use timing info to optimize
5. **Consider frame_step** - Reduce computation by sampling