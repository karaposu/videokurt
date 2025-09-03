# Basic Features Summary

VideoKurt's basic features are simple computations on raw analysis data that provide quantitative measurements without semantic interpretation. These features serve as building blocks for understanding video content.

## Core Basic Features (15)

### 1. **Binary Activity** 
- **Source**: `frame_diff`
- **Output**: Binary array (0=inactive, 1=active) per frame
- **Use Case**: Detect idle periods, segment active vs static content

### 2. **Motion Magnitude**
- **Source**: `optical_flow_dense` or `optical_flow_sparse`
- **Output**: Scalar value representing total motion per frame
- **Use Case**: Activity intensity, still vs moving scenes

### 3. **Motion Direction Histogram**
- **Source**: `optical_flow_dense`
- **Output**: Distribution of motion directions (8 or 16 bins)
- **Use Case**: Detect scrolling (vertical dominant), panning (horizontal dominant)

### 4. **Edge Density**
- **Source**: `edge_canny`
- **Output**: Percentage of edge pixels per frame
- **Use Case**: UI complexity, text-heavy vs image content, content richness

### 5. **Change Regions**
- **Source**: `frame_diff` with spatial analysis
- **Output**: Bounding boxes of changed areas per frame
- **Use Case**: Track where changes occur, UI update locations

### 6. **Stability Score**
- **Source**: `frame_diff` over temporal window
- **Output**: Measure of content stability (0=changing, 1=stable)
- **Use Case**: Detect paused video, static screens, loading states

### 7. **Repetition Indicator**
- **Source**: `frequency_fft`
- **Output**: Boolean or score for periodic patterns
- **Use Case**: Detect loops, typing rhythm, scrolling patterns

### 8. **Foreground Ratio**
- **Source**: `background_mog2` or `background_knn`
- **Output**: Percentage of frame marked as foreground
- **Use Case**: Activity level, new content appearance

### 9. **Motion Smoothness**
- **Source**: `optical_flow_dense` temporal variance
- **Output**: Smoothness score per frame window
- **Use Case**: Detect jerky movement, animation quality

### 10. **Content Persistence**
- **Source**: `motion_heatmap`
- **Output**: Duration map of active regions
- **Use Case**: Identify permanent UI elements, persistent objects

### 11. **Frame Difference Percentile**
- **Source**: `frame_diff`
- **Output**: 95th percentile of pixel differences per frame
- **Use Case**: Robust activity measure ignoring noise

### 12. **Dominant Flow Vector**
- **Source**: `optical_flow_dense`
- **Output**: Single dominant motion vector per frame
- **Use Case**: Camera movement detection, global motion

### 13. **Edge Orientation Histogram**
- **Source**: `edge_canny` gradient direction
- **Output**: Distribution of edge angles (horizontal/vertical/diagonal)
- **Use Case**: Detect UI layouts, text vs natural scenes

### 14. **Temporal Difference Energy**
- **Source**: `frame_diff_advanced` (triple diff)
- **Output**: Energy measure from multi-frame differences
- **Use Case**: Distinguish motion from illumination changes

### 15. **Motion Centroid**
- **Source**: `optical_flow_dense` or `motion_heatmap`
- **Output**: (x, y) center of motion per frame
- **Use Case**: Track focus of activity, cursor tracking

## Additional Useful Features (5)

### 16. **Change Rate**
- **Source**: `frame_diff` derivative
- **Output**: Rate of change acceleration/deceleration
- **Use Case**: Detect sudden vs gradual transitions

### 17. **Spatial Activity Distribution**
- **Source**: `frame_diff` or `motion_heatmap`
- **Output**: Grid-based activity map (e.g., 3x3 regions)
- **Use Case**: Identify screen quadrants with activity

### 18. **Edge Continuity**
- **Source**: `edge_canny` temporal
- **Output**: Percentage of edges that persist frame-to-frame
- **Use Case**: Distinguish UI elements from motion artifacts

### 19. **Motion Divergence**
- **Source**: `optical_flow_dense`
- **Output**: Divergence of flow field (expansion/contraction)
- **Use Case**: Detect zoom in/out, explosions, collapses

### 20. **Intensity Variance**
- **Source**: Raw frames or `frame_diff`
- **Output**: Variance of pixel intensities per frame
- **Use Case**: Detect flat UI vs textured content

## Feature Categories

### Motion-Based Features
- Motion Magnitude (#2)
- Motion Direction Histogram (#3)
- Motion Smoothness (#9)
- Dominant Flow Vector (#12)
- Motion Centroid (#15)
- Motion Divergence (#19)

### Change-Based Features
- Binary Activity (#1)
- Change Regions (#5)
- Stability Score (#6)
- Frame Difference Percentile (#11)
- Temporal Difference Energy (#14)
- Change Rate (#16)

### Structure-Based Features
- Edge Density (#4)
- Edge Orientation Histogram (#13)
- Edge Continuity (#18)
- Intensity Variance (#20)

### Temporal Features
- Repetition Indicator (#7)
- Content Persistence (#10)
- Stability Score (#6)
- Change Rate (#16)

### Spatial Features
- Foreground Ratio (#8)
- Spatial Activity Distribution (#17)
- Motion Centroid (#15)
- Change Regions (#5)

## Implementation Pattern

Each basic feature follows this pattern:

```python
def compute_feature(analysis_data, **params):
    """
    Compute a basic feature from raw analysis.
    
    Args:
        analysis_data: Raw analysis output (from VideoKurt)
        **params: Feature-specific parameters (thresholds, window sizes)
    
    Returns:
        numpy array or scalar per frame
    """
    # Simple mathematical computation
    # No complex logic or pattern matching
    # Returns quantitative measurement
```

## Usage Guidelines

### When to Use Each Feature

**For Screen Recordings:**
- Binary Activity - UI interaction detection
- Change Regions - Track UI updates
- Edge Density - Detect text/dialog appearance
- Stability Score - Identify loading/waiting states
- Spatial Activity Distribution - Monitor screen sections

**For Regular Video:**
- Motion Magnitude - Action intensity
- Motion Direction Histogram - Camera movement type
- Dominant Flow Vector - Global motion tracking
- Motion Divergence - Zoom detection
- Foreground Ratio - Subject tracking

**For Both:**
- Repetition Indicator - Find patterns
- Motion Smoothness - Quality assessment
- Content Persistence - Identify stable elements
- Change Rate - Transition detection

## Feature Selection Tips

1. **Start minimal**: Use 3-5 features initially
2. **Combine complementary**: Pair motion + structure features
3. **Consider temporal**: Add temporal features for pattern detection
4. **Match to use case**: Screen recordings need different features than action videos
5. **Test thresholds**: Basic features often need tuning per video type

## Performance Considerations

| Feature | Computation Cost | Memory Usage |
|---------|-----------------|--------------|
| Binary Activity | Low | Low |
| Motion Magnitude | Low | Low |
| Edge Density | Medium | Low |
| Motion Direction Histogram | Medium | Low |
| Change Regions | Medium | Medium |
| Motion Divergence | High | Low |
| Spatial Activity Distribution | Low | Medium |

Most basic features add <5ms per frame of processing time on modern hardware.