# VideoKurt Feature Refactor: Analysis-Driven Architecture

## Executive Summary

VideoKurt's feature extraction should treat analysis outputs AS features, rather than forcing them into artificial categories. This document defines a simplified, honest architecture where analyses produce features directly, features drive segmentation, and segments get classified into primitives.

---

## Core Philosophy

**Features ARE Analysis Results**

Instead of running analyses and then reorganizing their outputs into abstract categories, we treat the analysis outputs themselves as our features. This is simpler, more direct, and avoids forcing overlapping analyses into single categories.

---

## Architecture Overview

```
VIDEO → ANALYSES → FEATURES (are analysis outputs) → SEGMENTATION → PRIMITIVES
```

No extra abstraction layers. No forced categorization. Just direct data flow.

---

## Analysis Methods and Their Outputs

Each analysis method produces multiple types of information. We don't force them into categories - we use what they give us.

### 1. Optical Flow (Farneback & Lucas-Kanade)
```python
outputs:
    flow_vectors: np.ndarray        # [T, H, W, 2] - dx, dy per pixel
    
derives:
    motion_magnitude: np.ndarray    # [T] - average magnitude per frame
    motion_direction: np.ndarray    # [T] - dominant direction per frame
    motion_uniformity: np.ndarray   # [T] - variance of motion vectors
```

### 2. Frame Differencing
```python
outputs:
    pixel_differences: np.ndarray   # [T, H, W] - absolute differences
    triple_diff: np.ndarray         # [T] - acceleration detection
    
derives:
    change_percentage: np.ndarray   # [T] - % of pixels changed
    change_rate: np.ndarray         # [T] - speed of change
```

### 3. Background Subtraction (MOG2)
```python
outputs:
    foreground_mask: np.ndarray     # [T, H, W] - binary mask
    
derives:
    foreground_percentage: np.ndarray  # [T] - % of frame in foreground
    adaptation_rate: np.ndarray        # [T] - how fast bg adapts
```

### 4. Contour Detection
```python
outputs:
    contours: List[List[Contour]]   # [T][n] - list of shapes per frame
    
derives:
    contour_count: np.ndarray       # [T] - number of contours
    total_contour_area: np.ndarray  # [T] - sum of all contour areas
    largest_contour_area: np.ndarray # [T] - biggest contour size
```

### 5. Frequency Analysis (FFT on temporal signals)
```python
outputs:
    frequency_spectrum: np.ndarray   # [T, n_frequencies]
    
derives:
    dominant_frequency: np.ndarray   # [T] - main oscillation frequency
    frequency_bandwidth: np.ndarray  # [T] - range of frequencies
    periodicity_score: np.ndarray    # [T] - how periodic vs random
```

---

## Feature Organization

```python
@dataclass
class VideoFeatures:
    """All features extracted from video analysis."""
    
    # Raw analysis outputs (full resolution, optional)
    raw_outputs: Optional[RawOutputs] = None
        # Can be None to save memory after feature extraction
        
    # Frame-level features (used for segmentation)
    frame_features: FrameFeatures
        # One value per frame for each feature
        # This is what drives segmentation decisions
    
    # Metadata
    fps: float
    frame_count: int
    resolution: Tuple[int, int]

@dataclass
class RawOutputs:
    """Raw analysis outputs (optional, can be discarded)."""
    optical_flow: Optional[np.ndarray]        # [T, H, W, 2]
    frame_differences: Optional[np.ndarray]   # [T, H, W]
    foreground_masks: Optional[np.ndarray]    # [T, H, W]
    contours: Optional[List[List[Contour]]]   # [T][n]

@dataclass
class FrameFeatures:
    """Per-frame feature vectors for segmentation."""
    
    # Motion features (from optical flow)
    motion_magnitude: np.ndarray      # [T] - average flow magnitude
    motion_direction: np.ndarray      # [T] - dominant direction (degrees)
    motion_uniformity: np.ndarray     # [T] - how uniform motion is
    
    # Change features (from frame differencing)
    change_percentage: np.ndarray     # [T] - percent of pixels changed
    change_rate: np.ndarray          # [T] - derivative of change
    change_acceleration: np.ndarray   # [T] - second derivative
    
    # Structural features (from contours & edges)
    contour_count: np.ndarray        # [T] - number of shapes
    contour_area: np.ndarray         # [T] - total area of shapes
    edge_density: np.ndarray         # [T] - amount of edges
    
    # Foreground features (from background subtraction)
    foreground_percentage: np.ndarray # [T] - percent in foreground
    foreground_stability: np.ndarray  # [T] - how stable foreground is
    
    # Frequency features (from FFT)
    dominant_frequency: np.ndarray    # [T] - main oscillation freq
    frequency_spread: np.ndarray      # [T] - bandwidth
    periodicity: np.ndarray          # [T] - periodic vs random
    
    # Binary activity (for compatibility)
    is_active: np.ndarray            # [T] - boolean activity
    activity_confidence: np.ndarray   # [T] - confidence score
```

---

## Segmentation Logic

Segments are created when features change significantly:

```python
class SegmentDetector:
    """Detects segment boundaries from frame features."""
    
    def detect_segments(self, features: FrameFeatures) -> List[Segment]:
        segments = []
        current_start = 0
        
        for t in range(1, len(features.motion_magnitude)):
            # Check for significant changes
            if self.is_boundary(features, t):
                segments.append(Segment(
                    start_frame=current_start,
                    end_frame=t-1,
                    features=self.aggregate_features(features, current_start, t)
                ))
                current_start = t
        
        return segments
    
    def is_boundary(self, features: FrameFeatures, t: int) -> bool:
        """Detect if frame t is a segment boundary."""
        
        # Motion direction change
        if abs(features.motion_direction[t] - features.motion_direction[t-1]) > 45:
            return True
            
        # Activity change (idle ↔ active)
        if features.is_active[t] != features.is_active[t-1]:
            return True
            
        # Large magnitude change
        if abs(features.change_percentage[t] - features.change_percentage[t-1]) > 0.3:
            return True
            
        # Frequency shift
        if abs(features.dominant_frequency[t] - features.dominant_frequency[t-1]) > 5:
            return True
            
        return False
```

---

## Segment Classification

Segments are classified into primitives based on their feature patterns:

```python
@dataclass
class Segment:
    """A time interval with consistent visual patterns."""
    
    start_frame: int
    end_frame: int
    
    # Aggregated features over segment duration
    avg_motion_magnitude: float
    dominant_motion_direction: float
    avg_change_percentage: float
    avg_contour_count: float
    dominant_frequency: float
    
    # Classified primitive type
    primitive_type: str  # 'VERTICAL_SLIDE_UP', 'IDLE', etc.
    confidence: float
    
def classify_segment(segment: Segment) -> str:
    """Classify segment into primitive type."""
    
    # Idle detection
    if segment.avg_motion_magnitude < 0.1:
        return 'IDLE'
    
    # Scrolling detection
    if segment.avg_motion_magnitude > 1.0:
        if 80 < segment.dominant_motion_direction < 100:  # ~90 degrees
            return 'VERTICAL_SLIDE_UP'
        elif 260 < segment.dominant_motion_direction < 280:  # ~270 degrees
            return 'VERTICAL_SLIDE_DOWN'
            
    # Rapid changes (transitions)
    if segment.avg_change_percentage > 0.7:
        return 'FULL_CHANGE'
    elif segment.avg_change_percentage > 0.3:
        return 'PARTIAL_CHANGE'
        
    # High frequency (video/animation)
    if segment.dominant_frequency > 10:
        return 'HIGH_FREQUENCY_ACTIVITY'
        
    return 'UNKNOWN'
```

---

## Implementation Pipeline

```python
class VideoKurt:
    def analyze_video(self, video_path: str) -> VideoKurtResults:
        
        # 1. Load video
        frames = self.load_video(video_path)
        
        # 2. Run analyses
        raw_outputs = {
            'optical_flow': compute_optical_flow(frames),
            'frame_diff': compute_frame_differences(frames),
            'contours': detect_contours(frames),
            'foreground': extract_foreground(frames),
            'frequency': analyze_frequencies(frames)
        }
        
        # 3. Extract frame-level features
        frame_features = FrameFeatures(
            motion_magnitude=np.mean(raw_outputs['optical_flow'].magnitude, axis=(1,2)),
            motion_direction=compute_dominant_direction(raw_outputs['optical_flow']),
            change_percentage=np.mean(raw_outputs['frame_diff'], axis=(1,2)),
            contour_count=np.array([len(c) for c in raw_outputs['contours']]),
            # ... etc
        )
        
        # 4. Detect segments
        segments = self.segment_detector.detect_segments(frame_features)
        
        # 5. Classify segments
        for segment in segments:
            segment.primitive_type = classify_segment(segment)
        
        # 6. Return results
        return VideoKurtResults(
            features=frame_features,
            segments=segments,
            raw_outputs=None  # Discard to save memory
        )
```

---

## Key Advantages of This Approach

1. **No Forced Categorization**: Optical flow doesn't need to be "temporal" or "spatial" - it's just optical flow
2. **Direct and Simple**: Analysis outputs ARE features, no extra abstraction
3. **Memory Efficient**: Can discard raw outputs after extracting frame features
4. **Clear Data Flow**: Video → Analyses → Features → Segments → Primitives
5. **Easy to Extend**: Add new analysis = add new features

---

## What We're NOT Doing

- ❌ Forcing analyses into Temporal/Spatial/Spectral/Structural categories
- ❌ Creating complex hierarchies of features
- ❌ Tracking arbitrary regions during analysis
- ❌ Making semantic interpretations

---

## Summary

This refactored architecture treats features as what they really are: **outputs from our analysis algorithms**. We run analyses, extract frame-level features, detect segment boundaries when features change significantly, and classify those segments into primitive visual patterns.

The beauty is in the simplicity:
- **Features** = What our analyses measure
- **Segments** = Time intervals with consistent features
- **Primitives** = Classifications based on feature patterns

No artificial categorization. No complex hierarchies. Just a direct pipeline from video to primitive visual patterns.