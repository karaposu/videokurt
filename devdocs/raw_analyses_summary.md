# VideoKurt Raw Analyses Summary

This document provides a comprehensive overview of all 14 raw analysis implementations in VideoKurt. Raw analyses extract pixel-level information directly from video frames, serving as the foundation for higher-level feature extraction.

## Quick Reference Table

| Analysis | Category | Speed | Output Type | Frames Required |
|----------|----------|--------|-------------|-----------------|
| `frame_diff` | Change Detection | Fast | Pixel differences | 2+ |
| `frame_diff_advanced` | Change Detection | Fast | Triple diff, running avg | 3+ |
| `background_mog2` | Background Subtraction | Medium | Foreground masks | 2+ |
| `background_knn` | Background Subtraction | Medium | Foreground masks | 2+ |
| `edge_canny` | Structure Detection | Medium | Edge maps, gradients | 1+ |
| `contour_detection` | Structure Detection | Medium | Contour points | 2+ |
| `optical_flow_sparse` | Motion Analysis | Medium | Tracked points | 2+ |
| `optical_flow_dense` | Motion Analysis | Slow | Flow fields | 2+ |
| `motion_heatmap` | Motion Analysis | Fast | Activity maps | 2+ |
| `flow_hsv_viz` | Motion Visualization | Medium | HSV flow images | 2+ |
| `frequency_fft` | Frequency Domain | Slow | Spectrum analysis | 64+ |
| `color_histogram` | Color Analysis | Fast | Color distributions | 1+ |
| `dct_transform` | Frequency Domain | Medium | DCT coefficients | 1+ |
| `texture_descriptors` | Texture Analysis | Medium | Texture features | 1+ |

---

## Detailed Analysis Descriptions

### Change Detection Analyses

#### 1. **frame_diff**
**Purpose:** Detect pixel-level changes between consecutive frames  
**Output:** 
- `pixel_diff`: `np.ndarray[uint8, shape=(n-1, h, w)]` - Absolute differences
**Parameters:**
- `threshold`: float = 10.0 - Minimum change to consider
- `downsample`: float = 1.0 - Resolution scaling
**Use Cases:** Motion detection, activity monitoring, scene changes

#### 2. **frame_diff_advanced**
**Purpose:** Sophisticated change detection using multiple techniques  
**Output:**
- `triple_diff`: `np.ndarray[uint8, shape=(n-2, h, w)]` - Three-frame differencing
- `running_avg_diff`: `np.ndarray[uint8, shape=(n-2, h, w)]` - Running average comparison
- `accumulated_diff`: `np.ndarray[uint8, shape=(h, w)]` - Accumulated motion
**Parameters:**
- `threshold`: float = 30.0 - Change threshold
- `alpha`: float = 0.02 - Running average weight
- `downsample`: float = 1.0
**Use Cases:** Robust motion detection, noise filtering, ghost removal

### Background Subtraction Analyses

#### 3. **background_mog2**
**Purpose:** Separate foreground objects from background using Mixture of Gaussians  
**Output:**
- `foreground_mask`: `np.ndarray[uint8, shape=(n, h, w)]` - Binary masks
**Parameters:**
- `history`: int = 500 - Frames for background model
- `var_threshold`: float = 16 - Variance threshold
- `detect_shadows`: bool = True - Shadow detection
- `downsample`: float = 1.0
**Use Cases:** Object tracking, surveillance, activity detection

#### 4. **background_knn**
**Purpose:** K-Nearest Neighbors background subtraction  
**Output:**
- `foreground_mask`: `np.ndarray[uint8, shape=(n, h, w)]` - Binary masks
**Parameters:**
- `history`: int = 500 - Background model frames
- `dist_threshold`: float = 400.0 - Distance threshold
- `detect_shadows`: bool = True
- `downsample`: float = 1.0
**Use Cases:** Robust foreground detection, shadow handling

### Structure Detection Analyses

#### 5. **edge_canny**
**Purpose:** Detect edges and boundaries in frames  
**Output:**
- `edge_map`: `np.ndarray[uint8, shape=(n, h, w)]` - Binary edge pixels
- `gradient_magnitude`: `np.ndarray[float32, shape=(n, h, w)]` - Edge strength
- `gradient_direction`: `np.ndarray[float32, shape=(n, h, w)]` - Edge orientation
**Parameters:**
- `low_threshold`: float = 50 - Lower hysteresis threshold
- `high_threshold`: float = 150 - Upper hysteresis threshold
- `blur_kernel`: int = 5 - Gaussian blur size
- `downsample`: float = 1.0
**Use Cases:** Object detection, scene structure, text detection

#### 6. **contour_detection**
**Purpose:** Find and trace object boundaries  
**Output:**
- `contours`: `List[List[np.ndarray]]` - Contour points per frame
- `hierarchy`: `List[np.ndarray]` - Contour relationships
- `areas`: `List[List[float]]` - Contour areas
- `centroids`: `List[List[tuple]]` - Contour centers
**Parameters:**
- `threshold`: float = 30 - Binary threshold
- `min_area`: float = 100 - Minimum contour area
- `max_contours`: int = 100 - Maximum contours to keep
- `downsample`: float = 1.0
**Use Cases:** Object counting, shape analysis, region detection

### Motion Analysis

#### 7. **optical_flow_sparse**
**Purpose:** Track specific feature points across frames  
**Output:**
- `tracked_points`: `List[np.ndarray]` - Point positions per frame
- `point_status`: `List[np.ndarray]` - Tracking success flags
**Parameters:**
- `max_corners`: int = 100 - Maximum points to track
- `quality_level`: float = 0.3 - Corner quality threshold
- `min_distance`: int = 7 - Minimum distance between points
- `downsample`: float = 1.0
**Use Cases:** Object tracking, camera motion, trajectory analysis

#### 8. **optical_flow_dense**
**Purpose:** Compute motion vectors for every pixel  
**Output:**
- `flow_field`: `np.ndarray[float32, shape=(n-1, h, w, 2)]` - (dx, dy) vectors
**Parameters:**
- `pyr_scale`: float = 0.5 - Pyramid scale
- `levels`: int = 3 - Pyramid levels
- `winsize`: int = 15 - Averaging window
- `iterations`: int = 3 - Algorithm iterations
- `downsample`: float = 0.25 - Heavy downsampling default
**Use Cases:** Motion fields, video stabilization, motion segmentation

#### 9. **motion_heatmap**
**Purpose:** Accumulate motion activity over time  
**Output:**
- `cumulative`: `np.ndarray[uint8, shape=(h, w)]` - Total accumulation
- `weighted`: `np.ndarray[uint8, shape=(h, w)]` - Weighted by recency
- `snapshots`: `List[np.ndarray]` - Periodic snapshots
**Parameters:**
- `decay_factor`: float = 0.95 - Temporal decay
- `threshold`: float = 20 - Motion threshold
- `snapshot_interval`: int = 30 - Frames between snapshots
- `downsample`: float = 0.25 - Heavy downsampling
**Use Cases:** Activity zones, dwell time, hot spots

#### 10. **flow_hsv_viz**
**Purpose:** Convert optical flow to HSV color visualization  
**Output:**
- `hsv_flow`: `np.ndarray[uint8, shape=(n-1, h, w, 3)]` - HSV images
**Parameters:**
- `max_magnitude`: float = 10.0 - Maximum flow for saturation
- `saturation_boost`: float = 3.0 - Saturation multiplier
- `downsample`: float = 0.5
**Use Cases:** Flow visualization, motion debugging, presentation

### Frequency Domain Analyses

#### 11. **frequency_fft**
**Purpose:** Detect periodic patterns using Fast Fourier Transform  
**Output:**
- `frequency_spectrum`: `np.ndarray[float32, shape=(freq_bins, spatial_bins)]`
- `phase_spectrum`: `np.ndarray[float32, shape=(freq_bins, spatial_bins)]`
**Parameters:**
- `window_size`: int = 64 - FFT window (min 64 frames)
- `overlap`: float = 0.5 - Window overlap
- `spatial_bins`: int = 16 - Spatial resolution
- `downsample`: float = 0.25
**Use Cases:** Flicker detection, periodic motion, rhythm analysis

#### 12. **dct_transform**
**Purpose:** Extract frequency domain features using Discrete Cosine Transform  
**Output:**
- `dct_coefficients`: `np.ndarray[float32, shape=(n, keep_coeffs)]`
- `perceptual_hashes`: `np.ndarray[uint8, shape=(n, hash_bytes)]`
**Parameters:**
- `block_size`: int = 32 - Resize before DCT
- `keep_coeffs`: int = 64 - Coefficients to retain
- `downsample`: float = 1.0
**Use Cases:** Perceptual hashing, compression analysis, similarity

### Color and Texture Analyses

#### 13. **color_histogram**
**Purpose:** Compute color distribution statistics  
**Output:**
- `histograms`: `np.ndarray[float32, shape=(n, bins)]` - Color distributions
- `dominant_colors`: `np.ndarray[uint8, shape=(n, n_colors, 3)]` - Main colors
**Parameters:**
- `bins`: int = 256 - Histogram bins
- `n_colors`: int = 5 - Dominant colors to extract
- `use_hsv`: bool = False - Use HSV instead of BGR
- `downsample`: float = 0.5
**Use Cases:** Scene changes, color grading, white balance

#### 14. **texture_descriptors**
**Purpose:** Extract local texture patterns and statistics  
**Output:**
- `texture_features`: `np.ndarray[float32, shape=(n, h, w)]` - LBP features
- `texture_statistics`: `np.ndarray[float32, shape=(n, 4)]` - Mean, std, contrast, homogeneity
**Parameters:**
- `method`: str = 'lbp' - Feature extraction method
- `radius`: int = 1 - LBP radius
- `n_points`: int = 8 - LBP sampling points
- `downsample`: float = 1.0
**Use Cases:** Material classification, surface analysis, quality assessment

---

## Performance Guidelines

### Speed Categories

**Fast (<10ms per frame at 480p)**
- `frame_diff` - Simple subtraction
- `frame_diff_advanced` - Optimized operations
- `motion_heatmap` - Incremental updates
- `color_histogram` - Binning operations

**Medium (10-50ms per frame at 480p)**
- `edge_canny` - Convolution operations
- `contour_detection` - Boundary tracing
- `background_mog2/knn` - Model updates
- `optical_flow_sparse` - Point tracking
- `texture_descriptors` - Local patterns
- `dct_transform` - Block transforms
- `flow_hsv_viz` - Color conversion

**Slow (>50ms per frame at 480p)**
- `optical_flow_dense` - Per-pixel computation
- `frequency_fft` - Requires 64+ frame windows

### Memory Usage

**Low Memory**
- Single frame analyses (edge_canny, texture_descriptors)
- Incremental analyses (motion_heatmap, background_mog2)

**Medium Memory**
- Multi-frame buffers (frame_diff_advanced)
- Point tracking (optical_flow_sparse)

**High Memory**
- Dense flow fields (optical_flow_dense)
- FFT windows (frequency_fft with 64+ frames)

---

## Usage Patterns

### Basic Motion Detection
```python
vk.add_analysis('frame_diff')
vk.add_analysis('motion_heatmap')
```

### Advanced Motion Analysis
```python
vk.add_analysis('optical_flow_dense')
vk.add_analysis('flow_hsv_viz')
vk.add_analysis('motion_heatmap')
```

### Object Detection
```python
vk.add_analysis('background_knn')
vk.add_analysis('contour_detection')
vk.add_analysis('edge_canny')
```

### Scene Analysis
```python
vk.add_analysis('color_histogram')
vk.add_analysis('texture_descriptors')
vk.add_analysis('dct_transform')
```

### Periodic Pattern Detection
```python
vk.add_analysis('frequency_fft')  # Needs 64+ frames
vk.add_analysis('frame_diff_advanced')
```

---

## Configuration Tips

### Resolution Scaling
Most analyses support `downsample` parameter:
- Use `0.25` for motion_heatmap, optical_flow_dense (heavy computation)
- Use `0.5` for general analysis
- Use `1.0` only when full resolution is critical

### Frame Selection
Configure `frame_step` globally:
- `frame_step=1` - Every frame (slow, detailed)
- `frame_step=5` - Every 5th frame (balanced)
- `frame_step=10` - Every 10th frame (fast, coarse)

### Analysis Combinations
Some analyses work well together:
- `frame_diff` + `motion_heatmap` - Activity zones
- `optical_flow_sparse` + `contour_detection` - Object tracking
- `background_mog2` + `edge_canny` - Foreground structure
- `color_histogram` + `texture_descriptors` - Scene classification

---

## Special Considerations

### Frame Count Requirements
- **Most analyses:** Need 2+ frames (compare consecutive)
- **frame_diff_advanced:** Needs 3+ frames (triple differencing)
- **frequency_fft:** Needs 64+ frames (FFT window)

### Output Dimensions
- **n-1 outputs:** frame_diff, optical_flow (compare pairs)
- **n outputs:** edge_canny, backgrounds (process each frame)
- **Single output:** motion_heatmap cumulative (aggregate)
- **Variable:** contour_detection (depends on content)

### Default Downsampling
Some analyses default to aggressive downsampling for performance:
- `optical_flow_dense`: 0.25 (quarter resolution)
- `motion_heatmap`: 0.25
- `frequency_fft`: 0.25

This can be overridden but may significantly impact performance.