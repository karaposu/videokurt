# Raw Analyses Explained

Raw analyses are the foundational layer of VideoKurt, extracting pixel-level information directly from video frames without interpretation. Each analysis produces numerical arrays or matrices that serve as input for higher-level features.

## Currently Implemented (11)

### 1. **frame_diff**
Computes pixel-wise differences between consecutive frames to detect changes. Outputs a 3D array showing the magnitude of change at each pixel location.

### 2. **edge_canny**
Detects edges in frames using the Canny edge detection algorithm. Produces binary edge maps, gradient magnitudes, and gradient directions showing object boundaries and structural elements.

### 3. **frame_diff_advanced**
Performs triple-frame differencing and maintains running averages for more robust change detection. Helps distinguish actual motion from noise or lighting changes.

### 4. **contour_detection**
Finds and traces continuous boundaries in frames using OpenCV's contour detection. Returns lists of contour points and their hierarchical relationships.

### 5. **background_mog2**
Uses Mixture of Gaussians (MOG2) algorithm to separate foreground from background. Produces binary masks showing moving or new objects against learned background.

### 6. **background_knn**
Applies K-Nearest Neighbors algorithm for background subtraction. Generally more robust to shadows than MOG2 but computationally heavier.

### 7. **optical_flow_sparse**
Tracks specific feature points across frames using Lucas-Kanade method. Returns tracked point positions and their status, useful for following specific objects.

### 8. **optical_flow_dense**
Computes motion vectors for every pixel using Farneback algorithm. Produces complete flow fields showing direction and magnitude of movement at each location.

### 9. **motion_heatmap**
Accumulates motion over time to create heat maps of activity. Shows cumulative, weighted, and snapshot views of where movement occurs most frequently.

### 10. **frequency_fft**
Applies Fast Fourier Transform to detect periodic patterns in temporal signals. Outputs frequency spectrum and phase information revealing repetitive motions or flicker.

### 11. **flow_hsv_viz**
Converts optical flow fields to HSV color space for visualization. Encodes motion direction as hue and magnitude as saturation, creating intuitive motion visualizations.

## Proposed Additions (3)

### 12. **color_histogram**
Computes distribution of color or intensity values within each frame. Provides statistical summary of frame appearance useful for detecting scene changes or lighting shifts.

### 13. **dct_transform**
Applies Discrete Cosine Transform to frames for frequency domain analysis. Produces coefficients that capture coarse image structure, useful for perceptual hashing.

### 14. **texture_descriptors**
Extracts local texture patterns using methods like Local Binary Patterns or Gabor filters. Creates feature maps that distinguish between smooth, textured, and edge regions.

## Analysis Categories

### Motion-Based
- `optical_flow_sparse` - Point tracking
- `optical_flow_dense` - Full motion fields
- `motion_heatmap` - Temporal accumulation
- `flow_hsv_viz` - Motion visualization

### Change Detection
- `frame_diff` - Simple differencing
- `frame_diff_advanced` - Sophisticated differencing
- `background_mog2` - Statistical background modeling
- `background_knn` - Neighbor-based background modeling

### Structure Detection
- `edge_canny` - Edge detection
- `contour_detection` - Boundary tracing
- `texture_descriptors` - Texture patterns

### Frequency Domain
- `frequency_fft` - Temporal frequencies
- `dct_transform` - Spatial frequencies
- `color_histogram` - Color distributions

## Output Data Types

| Analysis | Output Type | Typical Shape |
|----------|------------|---------------|
| `frame_diff` | float array | (frames-1, height, width) |
| `edge_canny` | binary + float arrays | (frames, height, width) |
| `optical_flow_dense` | vector field | (frames-1, height, width, 2) |
| `motion_heatmap` | float arrays | (height, width) + snapshots |
| `background_mog2` | binary mask | (frames, height, width) |
| `frequency_fft` | complex array | (frames, freq_bins) |
| `color_histogram` | histogram bins | (frames, channels, bins) |
| `dct_transform` | coefficient matrix | (frames, coeff_height, coeff_width) |

## Performance Notes

**Fast** (<10ms per frame):
- `frame_diff`
- `background_mog2`
- `color_histogram`

**Medium** (10-50ms per frame):
- `edge_canny`
- `contour_detection`
- `optical_flow_sparse`
- `texture_descriptors`

**Slow** (50ms+ per frame):
- `optical_flow_dense`
- `frequency_fft` (depends on window size)
- `dct_transform` (depends on block size)

## Usage Guidelines

1. **Start minimal**: Use 2-3 analyses initially, add more as needed
2. **Consider dependencies**: Some features need specific raw analyses
3. **Balance coverage**: Combine motion + structure + appearance analyses
4. **Optimize performance**: Use downsampling for expensive analyses
5. **Check memory**: Dense analyses can consume significant RAM

Raw analyses are designed to be composable - the real power comes from combining multiple analyses to build sophisticated features.