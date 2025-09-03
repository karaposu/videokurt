# RawAnalysisResults Changes for Analysis Integration

## Analysis Count and Storage Feasibility

**Total Analysis Methods**: 11 different analyses identified in `list_of_all_analysis.md`
- Level 1 (Basic): 2 analyses
- Level 2 (Intermediate): 2 analyses  
- Level 3 (Advanced): 3 analyses
- Level 4 (Complex): 4 analyses

**Can we store all 11 in one dataclass?** YES, but with considerations:

1. **Memory Impact**: Running all 11 analyses on a 1080p 30fps video:
   - ~1GB per minute for raw outputs
   - Optical flow alone: [T, H, W, 2] = 30fps × 1920×1080×2×4 bytes = ~995MB/min
   - Most analyses produce [T, H, W] arrays = ~250MB/min each

2. **Practical Approach**: Use Optional fields + selective running
   - Each analysis output is Optional (can be None)
   - User selects which analyses to run
   - Raw outputs can be discarded after feature extraction

   

## Current State of RawAnalysisResults

Looking at `/Users/ns/Desktop/projects/videokurt/videokurt/models.py`:

```python
@dataclass
class RawAnalysisResults:
    # Video/Frame properties
    dimensions: tuple[int, int]
    fps: float
    duration: float
    frame_count: int
    
    # Analysis results (current)
    timeline: 'Timeline'  # Event timeline
    binary_activity: np.ndarray  # Boolean array per frame
    binary_activity_confidence: np.ndarray  # Confidence scores
    segments: List['Segment']  # High-level activity segments
    
    # Performance tracking
    elapsed_time: float
    
    # Optional source info
    filename: Optional[str] = None
```

**Current Limitations:**
1. Only stores binary activity (active/inactive) - too simplistic
2. No storage for raw analysis outputs (optical flow, contours, etc.)
3. No frame-level features beyond binary classification
4. Timeline/Segment structure assumes semantic segments (SCROLLING, CLICKING)
5. No support for spatial information (heatmaps, regions)

---

## What's Incompatible

### 1. Binary-Only Activity
**Problem**: Current results only store boolean activity array
**Reality**: We have 11 different analysis methods producing rich data
```python
# Current: Just binary
binary_activity: np.ndarray  # [T] boolean

# Needed: Raw analysis outputs
optical_flow: np.ndarray  # [T, H, W, 2] - motion vectors
foreground_masks: np.ndarray  # [T, H, W] - binary masks
contours: List[List[np.ndarray]]  # [T][n] - shape boundaries
```

### 2. No Raw Analysis Storage
**Problem**: No place to store optical flow fields, contours, masks
**Reality**: Raw analysis outputs are the primary data we need to store

### 3. Semantic Segments vs Primitives
**Problem**: Current segments assume user intent (SCROLLING, CLICKING)
**Reality**: Should store raw analysis results, derive primitives later

### 4. No Spatial Information
**Problem**: Everything is temporal, no spatial activity tracking
**Reality**: Need heatmaps, region tracking, spatial patterns from raw data

---

## Proposed New Structure

```python
@dataclass
class RawAnalysis:
    """Result from a single analysis method.
    
    Each analysis returns this standardized object containing
    its raw outputs and metadata.
    """
    # Analysis identification
    method: str  # 'frame_diff', 'optical_flow_dense', 'motion_heatmap', etc.
    
    # Raw output data (analysis-specific)
    data: Dict[str, np.ndarray]  # Named outputs
    # Examples:
    # - frame_diff: {'pixel_diff': [T,H,W]}
    # - optical_flow_dense: {'flow_field': [T,H,W,2]}
    # - contour_detection: {'contours': List, 'hierarchy': List}
    # - motion_heatmap: {'cumulative': [H,W], 'weighted': [H,W]}
    
    # Metadata
    parameters: Dict[str, Any]  # Parameters used for this analysis
    processing_time: float  # Time taken for this analysis
    memory_usage: Optional[int] = None  # Bytes used
    
    # Data shape info (for validation/debugging)
    output_shapes: Dict[str, tuple]  # Shape of each output
    dtype_info: Dict[str, str]  # Data type of each output

@dataclass
class RawAnalysisResults:
    """Collection of all analysis results for a video."""
    
    # === Video Metadata (unchanged) ===
    dimensions: tuple[int, int]  # (width, height)
    fps: float
    duration: float
    frame_count: int
    filename: Optional[str] = None
    
    # === Raw Analysis Results (NEW - MODULAR) ===
    analyses: Dict[str, RawAnalysis]  # Key: method name, Value: RawAnalysis
    # Example:
    # {
    #   'frame_diff': RawAnalysis(...),
    #   'optical_flow_dense': RawAnalysis(...),
    #   'contour_detection': RawAnalysis(...),
    #   'motion_heatmap': RawAnalysis(...)
    # }
    
    # === Legacy Support (KEEP FOR NOW) ===
    binary_activity: np.ndarray  # Derived from frame diffs
    binary_activity_confidence: np.ndarray
    timeline: Optional['Timeline'] = None
    segments: Optional[List['Segment']] = None
    
    # === Performance ===
    total_elapsed_time: float
    
    # === Convenience Methods ===
    def get_analysis(self, method: str) -> Optional[RawAnalysis]:
        """Get analysis result by method name."""
        return self.analyses.get(method)
    
    def has_analysis(self, method: str) -> bool:
        """Check if analysis was run."""
        return method in self.analyses
    
    def list_analyses(self) -> List[str]:
        """List all analyses that were run."""
        return list(self.analyses.keys())
```

---

## Migration Strategy

### Phase 1: Add Raw Analysis Storage (IMMEDIATE)
1. Add `RawAnalysisOutputs` class to models.py
2. Add `raw_analyses` field to `RawAnalysisResults`
3. Store raw outputs from existing analyses

### Phase 2: Integrate Explorations (NEXT)
1. Port exploration scripts as analysis modules
2. Add to analysis registry
3. Store their outputs in `RawAnalysisOutputs`

### Phase 3: Feature Extraction (FUTURE)
1. Create feature extraction layer (separate from raw storage)
2. Extract frame-level features from raw analyses
3. Detect primitive segments from features

---

## Implementation Steps

### Step 1: Update Results Dataclass
```python
# In models.py
@dataclass
class RawAnalysisResults:
    # ... existing fields ...
    
    # NEW: Add raw analysis storage
    raw_analyses: Optional['RawAnalysisOutputs'] = None
    analysis_methods_used: Optional[List[str]] = None
    analysis_parameters: Optional[Dict[str, Any]] = None
```

### Step 2: Custom Class for Each Analysis

```python
# Base class for all analyses
class BaseAnalysis:
    """Base class for all video analysis methods."""
    
    METHOD_NAME = None  # Must override
    
    def __init__(self, downsample: float = 1.0, **kwargs):
        """
        Args:
            downsample: Resolution scale (0.5 = half resolution)
            **kwargs: Analysis-specific parameters
        """
        self.downsample = downsample
        self.config = kwargs
        
    def preprocess_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Apply generic preprocessing like downsampling."""
        if self.downsample < 1.0:
            processed = []
            for frame in frames:
                h, w = frame.shape[:2]
                new_h = int(h * self.downsample)
                new_w = int(w * self.downsample)
                resized = cv2.resize(frame, (new_w, new_h))
                processed.append(resized)
            return processed
        return frames
    
    def analyze(self, frames: List[np.ndarray]) -> RawAnalysis:
        """Must be implemented by subclasses."""
        raise NotImplementedError

# 1. Frame Differencing (Level 1)
class FrameDiff(BaseAnalysis):
    """Simple frame differencing analysis."""
    
    METHOD_NAME = 'frame_diff'
    
    def __init__(self, downsample: float = 1.0, threshold: float = 0.1):
        super().__init__(downsample=downsample)
        self.threshold = threshold
    
    def analyze(self, frames: List[np.ndarray]) -> RawAnalysis:
        start_time = time.time()
        frames = self.preprocess_frames(frames)
        
        pixel_diffs = []
        for i in range(len(frames) - 1):
            diff = cv2.absdiff(frames[i], frames[i+1])
            pixel_diffs.append(diff)
        
        return RawAnalysis(
            method=self.METHOD_NAME,
            data={'pixel_diff': np.array(pixel_diffs)},
            parameters={'threshold': self.threshold, 'downsample': self.downsample},
            processing_time=time.time() - start_time,
            output_shapes={'pixel_diff': np.array(pixel_diffs).shape},
            dtype_info={'pixel_diff': 'uint8'}
        )

# 2. Edge Detection (Level 1)
class EdgeCanny(BaseAnalysis):
    """Canny edge detection analysis."""
    
    METHOD_NAME = 'edge_canny'
    
    def __init__(self, downsample: float = 1.0, 
                 low_threshold: int = 50, high_threshold: int = 150):
        super().__init__(downsample=downsample)
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
    
    def analyze(self, frames: List[np.ndarray]) -> RawAnalysis:
        # Implementation here
        pass

# 3. Advanced Frame Differencing (Level 2)
class FrameDiffAdvanced(BaseAnalysis):
    """Advanced frame differencing with multiple techniques."""
    
    METHOD_NAME = 'frame_diff_advanced'
    
    def __init__(self, downsample: float = 1.0, 
                 window_size: int = 5, accumulate: bool = True):
        super().__init__(downsample=downsample)
        self.window_size = window_size
        self.accumulate = accumulate
    
    def analyze(self, frames: List[np.ndarray]) -> RawAnalysis:
        # Implementation here
        pass

# 4. Contour Detection (Level 2)
class ContourDetection(BaseAnalysis):
    """Contour and shape detection analysis."""
    
    METHOD_NAME = 'contour_detection'
    
    def __init__(self, downsample: float = 0.5,  # Often downsample for contours
                 threshold: int = 127, max_contours: int = 100):
        super().__init__(downsample=downsample)
        self.threshold = threshold
        self.max_contours = max_contours
    
    def analyze(self, frames: List[np.ndarray]) -> RawAnalysis:
        # Implementation here
        pass

# 5. Background Subtraction MOG2 (Level 3)
class BackgroundMOG2(BaseAnalysis):
    """MOG2 background subtraction analysis."""
    
    METHOD_NAME = 'background_mog2'
    
    def __init__(self, downsample: float = 0.5,  # Often downsample for speed
                 history: int = 120, var_threshold: float = 16.0,
                 detect_shadows: bool = True):
        super().__init__(downsample=downsample)
        self.history = history
        self.var_threshold = var_threshold
        self.detect_shadows = detect_shadows
    
    def analyze(self, frames: List[np.ndarray]) -> RawAnalysis:
        # Implementation here
        pass

# 6. Background Subtraction KNN (Level 3)
class BackgroundKNN(BaseAnalysis):
    """KNN background subtraction analysis."""
    
    METHOD_NAME = 'background_knn'
    
    def __init__(self, downsample: float = 0.5,
                 history: int = 200, dist2_threshold: float = 400.0,
                 detect_shadows: bool = False):
        super().__init__(downsample=downsample)
        self.history = history
        self.dist2_threshold = dist2_threshold
        self.detect_shadows = detect_shadows
    
    def analyze(self, frames: List[np.ndarray]) -> RawAnalysis:
        # Implementation here
        pass

# 7. Optical Flow Sparse (Level 3)
class OpticalFlowSparse(BaseAnalysis):
    """Lucas-Kanade sparse optical flow analysis."""
    
    METHOD_NAME = 'optical_flow_sparse'
    
    def __init__(self, downsample: float = 1.0,  # Usually full res for accuracy
                 max_corners: int = 100, quality_level: float = 0.3,
                 min_distance: int = 7):
        super().__init__(downsample=downsample)
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
    
    def analyze(self, frames: List[np.ndarray]) -> RawAnalysis:
        # Implementation here
        pass

# 8. Optical Flow Dense (Level 4)
class OpticalFlowDense(BaseAnalysis):
    """Farneback dense optical flow analysis."""
    
    METHOD_NAME = 'optical_flow_dense'
    
    def __init__(self, downsample: float = 0.25,  # Heavy computation, downsample!
                 pyr_scale: float = 0.5, levels: int = 3,
                 winsize: int = 15, iterations: int = 3):
        super().__init__(downsample=downsample)
        self.pyr_scale = pyr_scale
        self.levels = levels
        self.winsize = winsize
        self.iterations = iterations
    
    def analyze(self, frames: List[np.ndarray]) -> RawAnalysis:
        start_time = time.time()
        frames = self.preprocess_frames(frames)
        
        flow_fields = []
        for i in range(len(frames) - 1):
            flow = cv2.calcOpticalFlowFarneback(
                frames[i], frames[i+1], None,
                self.pyr_scale, self.levels, self.winsize,
                self.iterations, 5, 1.2, 0
            )
            flow_fields.append(flow)
        
        flow_array = np.array(flow_fields)
        
        return RawAnalysis(
            method=self.METHOD_NAME,
            data={'flow_field': flow_array},
            parameters={
                'downsample': self.downsample,
                'pyr_scale': self.pyr_scale,
                'levels': self.levels,
                'winsize': self.winsize,
                'iterations': self.iterations
            },
            processing_time=time.time() - start_time,
            output_shapes={'flow_field': flow_array.shape},
            dtype_info={'flow_field': str(flow_array.dtype)}
        )

# 9. Motion Heatmap (Level 4)
class MotionHeatmap(BaseAnalysis):
    """Motion heatmap accumulation analysis."""
    
    METHOD_NAME = 'motion_heatmap'
    
    def __init__(self, downsample: float = 0.25,  # Heavy memory usage
                 decay_factor: float = 0.95, snapshot_interval: int = 30):
        super().__init__(downsample=downsample)
        self.decay_factor = decay_factor
        self.snapshot_interval = snapshot_interval
    
    def analyze(self, frames: List[np.ndarray]) -> RawAnalysis:
        # Implementation here
        pass

# 10. Frequency FFT (Level 4)
class FrequencyFFT(BaseAnalysis):
    """Frequency analysis using FFT."""
    
    METHOD_NAME = 'frequency_fft'
    
    def __init__(self, downsample: float = 0.1,  # Very small for FFT
                 window_size: int = 64, overlap: float = 0.5):
        super().__init__(downsample=downsample)
        self.window_size = window_size
        self.overlap = overlap
    
    def analyze(self, frames: List[np.ndarray]) -> RawAnalysis:
        # Implementation here
        pass

# 11. Flow HSV Visualization (Level 4)
class FlowHSVViz(BaseAnalysis):
    """HSV visualization of optical flow."""
    
    METHOD_NAME = 'flow_hsv_viz'
    
    def __init__(self, downsample: float = 0.5,
                 max_magnitude: float = 20.0, saturation_boost: float = 1.5):
        super().__init__(downsample=downsample)
        self.max_magnitude = max_magnitude
        self.saturation_boost = saturation_boost
    
    def analyze(self, frames: List[np.ndarray]) -> RawAnalysis:
        # Implementation here
        pass
```

### Step 3: Using Configured Analysis Instances

```python
class VideoKurt:
    def analyze_video(self, 
                      video_path: str,
                      analyses: Union[List[str], List[BaseAnalysis]] = None,
                      analysis_configs: Dict[str, dict] = None) -> RawAnalysisResults:
        """
        Args:
            video_path: Path to video file
            analyses: Either:
                - List of analysis names ['frame_diff', 'optical_flow_dense']
                - List of configured analysis instances
            analysis_configs: Config overrides for named analyses
                {'optical_flow_dense': {'downsample': 0.5, 'levels': 5}}
        """
        # Extract frames
        frames = self.extract_frames(video_path)
        
        # Prepare analysis instances
        analyzers = []
        
        if analyses is None:
            analyses = DEFAULT_PIPELINE
        
        for item in analyses:
            if isinstance(item, str):
                # Create instance from registry with optional config
                class_ref = ANALYSIS_REGISTRY[item]
                config = analysis_configs.get(item, {}) if analysis_configs else {}
                analyzer = class_ref(**config)
            elif isinstance(item, BaseAnalysis):
                # Already an instance
                analyzer = item
            else:
                raise ValueError(f"Invalid analysis: {item}")
            
            analyzers.append(analyzer)
        
        # Run analyses
        analysis_results = {}
        total_time = 0
        
        for analyzer in analyzers:
            result = analyzer.analyze(frames)
            analysis_results[analyzer.METHOD_NAME] = result
            total_time += result.processing_time
        
        return RawAnalysisResults(
            dimensions=(frames[0].shape[1], frames[0].shape[0]),
            fps=self.fps,
            duration=len(frames) / self.fps,
            frame_count=len(frames),
            filename=os.path.basename(video_path),
            analyses=analysis_results,
            total_elapsed_time=total_time,
            binary_activity=self._derive_binary_activity(analysis_results),
            binary_activity_confidence=self._derive_confidence(analysis_results)
        )

# Analysis Registry
ANALYSIS_REGISTRY = {
    # Level 1: Basic
    'frame_diff': FrameDiff,
    'edge_canny': EdgeCanny,
    
    # Level 2: Intermediate
    'frame_diff_advanced': FrameDiffAdvanced,
    'contour_detection': ContourDetection,
    
    # Level 3: Advanced
    'background_mog2': BackgroundMOG2,
    'background_knn': BackgroundKNN,
    'optical_flow_sparse': OpticalFlowSparse,
    
    # Level 4: Complex
    'optical_flow_dense': OpticalFlowDense,
    'motion_heatmap': MotionHeatmap,
    'frequency_fft': FrequencyFFT,
    'flow_hsv_viz': FlowHSVViz
}

# Default configurations for common use cases
FAST_CONFIG = {
    'frame_diff': {'downsample': 0.5},
    'optical_flow_dense': {'downsample': 0.25, 'levels': 2},
    'motion_heatmap': {'downsample': 0.25},
}

QUALITY_CONFIG = {
    'frame_diff': {'downsample': 1.0},
    'optical_flow_dense': {'downsample': 0.5, 'levels': 5, 'iterations': 5},
    'motion_heatmap': {'downsample': 0.5, 'decay_factor': 0.98},
}
```

---

## Benefits of New Structure

1. **Raw Data First**: Stores actual analysis outputs, not interpretations
2. **Memory Aware**: Can selectively run analyses based on memory
3. **Extensible**: Easy to add new analysis methods
4. **No Premature Abstraction**: Raw data preserved for later processing
5. **Backward Compatible**: Legacy fields still populated

---

## Example Usage

```python
# 1. Simple usage with defaults
vk = VideoKurt()
results = vk.analyze_video("video.mp4")  # Uses DEFAULT_PIPELINE

# 2. Using named analyses with default configs
results = vk.analyze_video(
    "video.mp4",
    analyses=['frame_diff', 'optical_flow_dense', 'motion_heatmap']
)

# 3. Using named analyses with custom configs
results = vk.analyze_video(
    "video.mp4",
    analyses=['frame_diff', 'optical_flow_dense'],
    analysis_configs={
        'optical_flow_dense': {'downsample': 0.5, 'levels': 5, 'iterations': 5}
    }
)

# 4. Using pre-configured instances for fine control
flow_analyzer = OpticalFlowDense(downsample=0.25, levels=3, iterations=5)
heatmap_analyzer = MotionHeatmap(downsample=0.5, decay_factor=0.98)

results = vk.analyze_video(
    "video.mp4",
    analyses=[
        FrameDiff(downsample=1.0),  # Full resolution for frame diff
        flow_analyzer,
        heatmap_analyzer
    ]
)

# 5. Using preset configurations
results = vk.analyze_video(
    "video.mp4",
    analyses=['frame_diff', 'optical_flow_dense', 'motion_heatmap'],
    analysis_configs=FAST_CONFIG  # Use fast presets
)

# 6. Mixed approach - some named, some instances
results = vk.analyze_video(
    "video.mp4",
    analyses=[
        'frame_diff',  # Use default config
        OpticalFlowDense(downsample=0.1),  # Custom instance
        'motion_heatmap'  # Use default config
    ],
    analysis_configs={
        'frame_diff': {'threshold': 0.2},  # Override for named
        'motion_heatmap': {'snapshot_interval': 60}
    }
)

# Access results (same for all approaches)
if results.has_analysis('optical_flow_dense'):
    flow = results.get_analysis('optical_flow_dense')
    print(f"Flow computed at {flow.parameters['downsample']} resolution")
    print(f"Processing time: {flow.processing_time:.2f}s")
    flow_field = flow.data['flow_field']  # [T, H, W, 2]
```

## Benefits of RawAnalysis Approach

1. **Modular**: Each analysis is self-contained with its own result object
2. **Extensible**: Easy to add new analyses without changing RawAnalysisResults
3. **Memory Efficient**: Can selectively load/unload individual analyses
4. **Self-Documenting**: Each RawAnalysis contains its parameters and metadata
5. **Type Safe**: Clear structure for each analysis output
6. **Debugging Friendly**: Shape and dtype info included

---

## Analysis Output Reference

What each analysis method returns in its `data` dictionary:

| Method Name | Output Keys | Data Shape | Description |
|------------|-------------|------------|-------------|
| `frame_diff` | `pixel_diff` | `[T, H, W]` | Absolute pixel differences |
| `edge_canny` | `edge_map`, `gradient_magnitude`, `gradient_direction` | `[T, H, W]` each | Edge detection results |
| `frame_diff_advanced` | `triple_diff`, `running_avg_diff`, `accumulated_diff` | Various | Advanced differencing |
| `contour_detection` | `contours`, `hierarchy` | `[T][n]` lists | Shape boundaries |
| `background_mog2` | `foreground_mask` | `[T, H, W]` | Binary foreground mask |
| `background_knn` | `foreground_mask` | `[T, H, W]` | Binary foreground mask |
| `optical_flow_sparse` | `tracked_points`, `point_status` | `[T][n_points]` | Tracked features |
| `optical_flow_dense` | `flow_field` | `[T, H, W, 2]` | Motion vectors (dx, dy) |
| `motion_heatmap` | `cumulative`, `weighted`, `snapshots` | `[H, W]` + list | Activity heatmaps |
| `frequency_fft` | `frequency_spectrum`, `phase_spectrum` | `[T, n_freq]` | Frequency analysis |
| `flow_hsv_viz` | `hsv_flow` | `[T, H, W, 3]` | HSV visualization |

---

## Priority Actions

1. **Immediate**: Add `RawAnalysisOutputs` class to models.py
2. **Today**: Update analyze_video to store raw outputs
3. **This Week**: Port explorations as analysis modules
4. **Next Week**: Create feature extraction layer (separate concern)
5. **Future**: Primitive segment detection from features

This approach focuses on storing raw analysis results first, with feature extraction and segment detection as separate, later steps.