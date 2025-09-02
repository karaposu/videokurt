# Frame Differencer Module

## Overview

The Frame Differencer module is the foundation of VideoKurt's visual change detection system. It provides multiple algorithms for detecting and quantifying differences between video frames, enabling the identification of motion, scene changes, and activity patterns.

## Architecture

```
frame_differencer/
├── __init__.py          # Module exports
├── base.py              # Abstract base class and result container
├── simple.py            # Pixel-wise differencing
├── histogram.py         # Distribution-based comparison
├── ssim.py              # Structural similarity
├── hybrid.py            # Combined approach
├── factory.py           # Creation utilities
└── README.md            # This file
```

## Core Components

### DifferenceResult
Data class containing:
- `score`: Normalized difference (0.0-1.0)
- `diff_mask`: Pixel-level difference map
- `metadata`: Algorithm-specific details

### FrameDifferencer (Abstract Base)
- Defines interface for all algorithms
- Handles frame preprocessing
- Validates configurations

## Available Algorithms

### 1. SimpleFrameDiff
**Purpose**: Fast pixel-wise comparison
**Best for**: Real-time processing, initial activity detection
**Features**:
- Gaussian blur for noise reduction
- Configurable noise threshold
- Normalized scoring

```python
from videokurt.core import SimpleFrameDiff

diff = SimpleFrameDiff(blur_kernel=5, noise_threshold=10)
result = diff.compute_difference(frame1, frame2)
```

### 2. HistogramFrameDiff
**Purpose**: Compare color/brightness distributions
**Best for**: Detecting lighting changes, robust to minor movements
**Features**:
- Multiple color spaces (gray, RGB, HSV)
- Various distance metrics
- Spatial invariance

```python
from videokurt.core import HistogramFrameDiff

diff = HistogramFrameDiff(channels='hsv', distance_metric='correlation')
result = diff.compute_difference(frame1, frame2)
```

### 3. SSIMFrameDiff
**Purpose**: Perceptual similarity measurement
**Best for**: Human-like perception of changes
**Features**:
- Considers luminance, contrast, structure
- Configurable window size
- Multi-channel support

```python
from videokurt.core import SSIMFrameDiff

diff = SSIMFrameDiff(window_size=7, gaussian_weights=True)
result = diff.compute_difference(frame1, frame2)
```

### 4. HybridFrameDiff
**Purpose**: Combines multiple methods for robustness
**Best for**: High accuracy requirements
**Features**:
- Weighted combination of algorithms
- Voting mechanism
- Configurable weights

```python
from videokurt.core import HybridFrameDiff

diff = HybridFrameDiff(weights={'simple': 0.4, 'histogram': 0.3, 'ssim': 0.3})
result = diff.compute_difference(frame1, frame2)
```

## Usage Examples

### Basic Usage
```python
from videokurt.core import create_differencer

# Create differencer with factory
diff = create_differencer('simple', blur_kernel=3)

# Compare frames
result = diff.compute_difference(frame1, frame2)

# Check if changed
if result.changed:
    print(f"Change detected: score={result.score:.3f}")
```

### With Custom Threshold
```python
# Use custom threshold
if result.changed_with_threshold(0.2):
    print("Significant change detected")

# Access metadata
print(f"Changed pixels: {result.metadata['changed_pixels']}")
print(f"Change ratio: {result.metadata['changed_ratio']:.2%}")
```

### ROI Masking
```python
# Create mask for region of interest
mask = np.zeros(frame1.shape[:2], dtype=np.uint8)
mask[100:200, 150:250] = 255

# Apply mask during comparison
result = diff.compute_difference(frame1, frame2, mask=mask)
```

## Performance Considerations

### Speed Ranking (fastest to slowest)
1. **SimpleFrameDiff** - ~1000 FPS on 720p
2. **HistogramFrameDiff** - ~500 FPS on 720p
3. **SSIMFrameDiff** - ~100 FPS on 720p
4. **HybridFrameDiff** - ~50 FPS on 720p

### Memory Usage
- All algorithms work with frame pairs (no history needed)
- Diff masks require same memory as single frame
- Metadata overhead is minimal

### Optimization Tips
1. Use SimpleFrameDiff for real-time applications
2. Reduce frame resolution before processing
3. Apply ROI masks to focus on important areas
4. Cache differencer instances (avoid recreation)

## Integration with VideoKurt

### Activity Detection Pipeline
```python
from videokurt.core import SimpleFrameDiff

diff = SimpleFrameDiff(noise_threshold=15)

# Process video frames
activity = []
for i in range(1, len(frames)):
    result = diff.compute_difference(frames[i-1], frames[i])
    activity.append({
        'frame': i,
        'score': result.score,
        'active': result.score > 0.1
    })
```

### Event Detection
```python
# Detect scene changes with histogram method
hist_diff = HistogramFrameDiff(distance_metric='chi_square')

for i in range(1, len(frames)):
    result = hist_diff.compute_difference(frames[i-1], frames[i])
    if result.score > 0.7:
        print(f"Scene change at frame {i}")
```

## Testing

The module includes comprehensive smoke tests that work with the SampleMaker module:

```python
from videokurt.samplemaker import create_blank_frame, add_circle
from videokurt.core import SimpleFrameDiff

# Create test frames
frame1 = create_blank_frame((50, 50))
frame2 = add_circle(frame1, (25, 25), 10, color=(255,))

# Test differencing
diff = SimpleFrameDiff()
result = diff.compute_difference(frame1, frame2)
assert result.score > 0  # Change detected
```

## Dependencies

### Required
- numpy: Array operations
- scipy: Signal processing (fallback implementations)

### Optional (performance/features)
- opencv-cv2: Optimized image processing
- scikit-image: SSIM implementation

## Best Practices

1. **Choose the right algorithm**:
   - Simple: General activity detection
   - Histogram: Lighting-invariant detection
   - SSIM: Quality assessment
   - Hybrid: Critical applications

2. **Tune parameters**:
   - Start with defaults
   - Adjust based on false positive/negative rates
   - Consider video quality and content type

3. **Handle edge cases**:
   - Check for identical frames (score = 0)
   - Account for compression artifacts
   - Consider lighting variations

## Future Enhancements

- [ ] GPU acceleration support
- [ ] Temporal differencing (multi-frame)
- [ ] Machine learning-based methods
- [ ] Adaptive threshold calculation
- [ ] Region-specific sensitivity