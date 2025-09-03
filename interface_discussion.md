# VideoKurt Interface Discussion

## Required Interface Capabilities

### 1. Analysis Management
- Add analyses by name (string)
- Add analyses by configured object
- List available analyses
- Configure analyses individually
- Run selected analyses only

### 2. Feature Management  
- Add features by name
- List available features
- Get computed features after analysis
- Run features without storing raw analysis (memory optimization)
- Specify feature dependencies automatically

### 3. Configuration Options
- Global preprocessing (frame_step, resolution_scale, blur, blur_kernel_size)
- Memory optimization (process_chunks, chunk_overlap)
- Per-analysis configuration
- Override global settings per analysis
- Configuration validation

### 4. Execution Modes
- Full analysis with results storage
- Streaming analysis (process and discard)
- Feature-only mode (skip raw data storage)
- Partial video processing (time ranges)

### 5. Input Flexibility
- Video file path
- Numpy array sequences
- Video URL (future)
- Frame generator (future)

### 6. Output Options
- Full RawAnalysisResults object
- Feature-only results
- Selective data retrieval
- Debug video export (save_video method)

### 7. Memory Management
- Basic chunked processing (process_chunks parameter)
- Configure chunk overlap for continuity
- Clear frames after processing
- Selective analysis/feature storage
- Full streaming with minimal memory (future)

### 8. Error Handling
- Partial results on failure
- Skip failed analyses
- Validation before processing
- Clear error messages

## Current Implementation Analysis

The existing `VideoKurt` class uses:
- Constructor parameters for global config (frame_step, downsample)
- `analyze()` method accepts analyses as list or dict
- Returns unified `RawAnalysisResults` dataclass

### Limitations of Current Design:
1. Configuration mixed with initialization
2. No separate feature extraction step
3. No ability to add analyses incrementally
4. Features not exposed in main API

## Proposed Interface Design

### Option A: Builder Pattern (Your Suggestion)
```python
from videokurt import VideoKurt
from videokurt.raw_analysis import FrameDiff

# Initialize
vk = VideoKurt()

# Add analyses - by name or object
vk.add_analysis('frame_diff')
vk.add_analysis('optical_flow_dense')

# Or with custom config
custom_frame_diff = FrameDiff(threshold=0.3)
vk.add_analysis(custom_frame_diff)

# Configure preprocessing
vk.configure(
    frame_step=2,
    resolution_scale=0.5,
    blur=True,
    blur_kernel_size=15
)

# Add features
vk.add_feature('binary_activity')
vk.add_feature('scene_detection')

# Run analysis
results = vk.analyze('video.mp4')

# Get features only
features = vk.get_features()
```

**Pros:**
- Clear, incremental building
- Separation of concerns
- Easy to understand
- Flexible configuration

**Cons:**
- Stateful (order matters)
- Multiple method calls
- Need to track what's added

## My Recommendation: Hybrid Approach

Combine the best aspects - use builder pattern with method chaining and config objects:

```python
from videokurt import VideoKurt
from videokurt.raw_analysis import FrameDiff, OpticalFlowDense

# Basic usage - simple and clean
vk = VideoKurt()
vk.add_analysis('frame_diff')
vk.add_analysis('optical_flow_dense') 
vk.configure(frame_step=2, resolution_scale=0.5)
results = vk.analyze('video.mp4')

# Advanced usage - with custom configs
frame_diff = FrameDiff(threshold=0.3, downsample=0.5)
optical_flow = OpticalFlowDense(pyr_scale=0.5, levels=5)

vk = VideoKurt()
vk.add_analysis(frame_diff)  # Add configured object
vk.add_analysis(optical_flow)
vk.add_feature('binary_activity')
vk.add_feature('scene_detection')

# Configure and analyze
vk.configure(frame_step=2)
results = vk.analyze('video.mp4')

# Feature-only mode (no raw data storage)
vk.set_mode('features_only')  # Discards raw analysis data
features = vk.analyze('video.mp4').features

# Direct feature extraction (auto-adds required analyses)
vk2 = VideoKurt()
vk2.add_feature('scene_detection')  # Auto-adds frame_diff, edge_canny
vk2.analyze('video.mp4')
```

## Implementation Details

### 1. Analysis Registration
```python
class VideoKurt:
    def __init__(self):
        self.analyses = {}  # name -> BaseAnalysis instance
        self.features = {}  # name -> BaseFeature class
        self.config = {}    # global preprocessing config
        
    def add_analysis(self, analysis: Union[str, BaseAnalysis]):
        """Add analysis by name or configured instance."""
        if isinstance(analysis, str):
            # Create default instance
            self.analyses[analysis] = ANALYSIS_REGISTRY[analysis]()
        else:
            # Use provided instance
            name = analysis.METHOD_NAME
            self.analyses[name] = analysis
```

### 2. Feature Auto-dependency
```python
def add_feature(self, feature_name: str):
    """Add feature and auto-include required analyses."""
    feature_class = FEATURE_REGISTRY[feature_name]
    self.features[feature_name] = feature_class
    
    # Auto-add required analyses
    for required in feature_class.REQUIRED_ANALYSES:
        if required not in self.analyses:
            self.add_analysis(required)
```

### 3. Execution Modes
```python
def analyze(self, video_path: str, mode='full'):
    """
    Modes:
    - 'full': Keep all raw analysis data
    - 'features_only': Compute features, discard raw data
    - 'streaming': Process in chunks, minimal memory
    """
    if mode == 'features_only':
        # Run analyses but don't store raw arrays
        # Only compute and return features
        pass
```

## Additional Utility Methods

### save_video (Static Method)
```python
@staticmethod
def save_video(frames: List[np.ndarray], 
               output_path: str,
               fps: float = 30.0,
               codec: str = 'mp4v') -> bool:
    """Save frames as video for debugging purposes."""
```

**Use Cases:**
1. **Debug preprocessing**: Save frames after applying resolution_scale, blur, etc.
2. **Visualize analysis results**: Convert analysis arrays to viewable videos
3. **Export segments**: Save specific portions of video for inspection
4. **Create test videos**: Save generated frames from samplemaker

**Implementation Notes:**
- Should handle both grayscale and color frames
- Automatically determine frame dimensions from first frame
- Support common codecs (mp4v, XVID, MJPG, etc.)
- Provide helpful error messages for codec issues

## Key Decisions Needed

1. **Should we auto-add dependencies?**
   - When adding a feature, automatically add required analyses?
   - Pro: Convenience
   - Con: Hidden behavior

2. **Configuration precedence?**
   - Global config vs per-analysis config
   - Suggestion: Per-analysis overrides global

3. **Lazy vs Eager execution?**
   - Run analyses when `analyze()` called (lazy)
   - Or run immediately when added (eager)
   - Suggestion: Lazy (current approach)

4. **Memory management default?**
   - Keep all data by default vs streaming by default
   - Suggestion: Keep all with easy streaming option

5. **Feature computation timing?**
   - Compute during analysis or on-demand after?
   - Suggestion: During analysis for efficiency

## Proposed Final Interface

```python
# videokurt/core.py

class VideoKurt:
    def __init__(self):
        """Initialize empty VideoKurt instance."""
        self._analyses = {}
        self._features = {}
        self._config = {
            'frame_step': 1,
            'resolution_scale': 1.0,
            'blur': False,
            'blur_kernel_size': 13,
            'process_chunks': 1,  # 1 = no chunking
            'chunk_overlap': 30   # frames overlap between chunks
        }
        self._mode = 'full'
    
    def add_analysis(self, 
                     analysis: Union[str, BaseAnalysis],
                     **kwargs):
        """Add analysis by name with kwargs or pre-configured object."""
        if isinstance(analysis, str):
            analysis_class = ANALYSIS_REGISTRY[analysis]
            self._analyses[analysis] = analysis_class(**kwargs)
        else:
            self._analyses[analysis.METHOD_NAME] = analysis
    
    def add_feature(self, feature: str):
        """Add feature and auto-include dependencies."""
        feature_class = FEATURE_REGISTRY[feature]
        self._features[feature] = feature_class
        
        # Auto-add required analyses with defaults
        for req in feature_class.REQUIRED_ANALYSES:
            if req not in self._analyses:
                self.add_analysis(req)
    
    def configure(self, **kwargs):
        """Configure global preprocessing."""
        self._config.update(kwargs)
    
    def set_mode(self, mode: str):
        """Set execution mode: 'full', 'features_only', 'streaming'."""
        self._mode = mode
    
    def analyze(self, 
                video_path: Union[str, Path, np.ndarray],
                return_features_only: bool = False) -> Union[RawAnalysisResults, FeatureResults]:
        """Run analysis with configured settings."""
        # Implementation here
        pass
    
    def list_analyses(self) -> List[str]:
        """List configured analyses."""
        return list(self._analyses.keys())
    
    def list_features(self) -> List[str]:
        """List configured features.""" 
        return list(self._features.keys())
    
    def clear(self):
        """Clear all configurations."""
        self._analyses.clear()
        self._features.clear()
    
    @staticmethod
    def save_video(frames: List[np.ndarray], 
                   output_path: str,
                   fps: float = 30.0,
                   codec: str = 'mp4v') -> bool:
        """
        Save frames as video file for debugging.
        
        Args:
            frames: List of numpy arrays (frames to save)
            output_path: Path where to save the video
            fps: Frames per second for output video
            codec: Four-character code for video codec
        
        Returns:
            True if successful, False otherwise
            
        Example:
            VideoKurt.save_video(processed_frames, 'debug_output.mp4')
        """
        pass
```

This design provides:
- Clean, intuitive API
- Clear separation of concerns  
- Flexibility for different use cases
- Memory optimization options
- Auto-dependency handling

## Additional Interface Considerations

### Per-Analysis Configuration Override

Each analysis should be able to override global preprocessing:

```python
# Global config applies to all
vk.configure(resolution_scale=0.5, blur=True)

# But optical flow needs full resolution
flow = OpticalFlowDense(
    override_resolution_scale=1.0,  # Ignore global downscaling
    override_blur=False             # No blur for this analysis
)
vk.add_analysis(flow)
```

### Batch Processing Support

Process multiple videos with same configuration:

```python
vk = VideoKurt()
vk.add_analysis('frame_diff')
vk.add_feature('scene_detection')
vk.configure(frame_step=2)

# Process batch
videos = ['video1.mp4', 'video2.mp4', 'video3.mp4']
results = vk.analyze_batch(videos)  # Returns list of results
```

### Progressive Analysis

Start getting results before full completion:

```python
# With callback
def on_analysis_complete(analysis_name, result):
    print(f"Completed {analysis_name}")
    # Process result immediately

vk.analyze('video.mp4', callback=on_analysis_complete)

# Or with generator
for analysis_name, result in vk.analyze_progressive('video.mp4'):
    # Process each analysis as it completes
    pass
```

### Analysis Presets

Common configuration patterns:

```python
# Built-in presets
vk = VideoKurt.from_preset('fast_scan')  # Low quality, quick
vk = VideoKurt.from_preset('ui_analysis')  # For screen recordings
vk = VideoKurt.from_preset('movie_analysis')  # For film content

# Custom presets
preset = {
    'analyses': ['frame_diff', 'optical_flow_dense'],
    'features': ['scene_detection', 'scrolling_detection'],
    'config': {
        'frame_step': 2,
        'resolution_scale': 0.5
    }
}
vk = VideoKurt.from_dict(preset)
```

### Validation and Dry Run

Check configuration before processing:

```python
vk = VideoKurt()
vk.add_feature('scene_detection')

# Validate configuration
issues = vk.validate()
if issues:
    print(f"Configuration issues: {issues}")

# Dry run - check without processing
info = vk.dry_run('video.mp4')
print(f"Would process {info['frame_count']} frames")
print(f"Estimated memory: {info['memory_estimate']} MB")
print(f"Analyses to run: {info['analyses']}")
```

### Export/Import Configuration

Save and reuse configurations:

```python
# Export
vk = VideoKurt()
vk.add_analysis('frame_diff')
vk.configure(frame_step=2)
config_dict = vk.export_config()

# Save to file
vk.save_config('my_config.json')

# Import
vk2 = VideoKurt.from_config('my_config.json')
```

### Analysis Caching

Reuse results across runs:

```python
# Enable caching
vk = VideoKurt(cache_dir='./cache')
vk.add_analysis('optical_flow_dense')  # Expensive

# First run - computes and caches
results1 = vk.analyze('video.mp4')

# Second run - uses cache if available
results2 = vk.analyze('video.mp4')  # Much faster
```

### Direct Analysis Access

For advanced users who want specific analysis without VideoKurt wrapper:

```python
from videokurt.raw_analysis import FrameDiff

# Direct usage
analysis = FrameDiff(threshold=0.3)
frames = load_video('video.mp4')
result = analysis.analyze(frames)
```

### Debug Video Export

Save processed frames or analysis visualizations:

```python
vk = VideoKurt()
vk.add_analysis('frame_diff')
vk.configure(resolution_scale=0.5, blur=True)

# Get processed frames for debugging
frames, metadata = vk._load_video('input.mp4')
processed = vk._preprocess_frames(frames)

# Save processed frames to check preprocessing
VideoKurt.save_video(processed, 'debug_preprocessed.mp4', fps=metadata['fps'])

# Or save analysis results visualization
results = vk.analyze('input.mp4')
diff_frames = results.analyses['frame_diff'].data['pixel_diff']

# Convert single-channel diff to viewable format
visual_frames = []
for diff in diff_frames:
    # Normalize and convert to 3-channel for viewing
    normalized = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    visual = cv2.cvtColor(normalized.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    visual_frames.append(visual)

VideoKurt.save_video(visual_frames, 'debug_frame_diff.mp4', fps=30)
```

### Error Recovery Strategies

Configure how to handle failures:

```python
vk = VideoKurt()
vk.set_error_strategy('skip')  # Skip failed analyses
# or
vk.set_error_strategy('retry', max_retries=3)  # Retry failures
# or
vk.set_error_strategy('fail')  # Stop on first error

results = vk.analyze('video.mp4')
if results.has_errors():
    for error in results.errors:
        print(f"{error.analysis}: {error.message}")
```

## Usage Scenarios

### Scenario 1: Quick Activity Detection
```python
# Minimal setup for activity timeline
vk = VideoKurt()
vk.add_analysis('frame_diff')
vk.add_feature('binary_activity')
vk.configure(frame_step=5, resolution_scale=0.25)
results = vk.analyze('screen_recording.mp4')
activity = results.features['binary_activity']
```

### Scenario 2: Full Analysis Pipeline
```python
# Complete analysis with all features
vk = VideoKurt()

# Add all analyses with custom configs
analyses = [
    FrameDiff(threshold=0.2),
    OpticalFlowDense(downsample=0.5),
    EdgeCanny(low_threshold=50),
    BackgroundMOG2()
]
for analysis in analyses:
    vk.add_analysis(analysis)

# Add features
features = ['scene_detection', 'scrolling_detection', 
            'ui_change_detection', 'camera_movement']
for feature in features:
    vk.add_feature(feature)

# Configure and run
vk.configure(frame_step=1)  # Process all frames
results = vk.analyze('complex_video.mp4')
```

### Scenario 3: Memory-Constrained Environment
```python
# Optimize for low memory usage
vk = VideoKurt()
vk.add_analysis('frame_diff')
vk.add_feature('binary_activity')
vk.configure(
    resolution_scale=0.25,  # Very low resolution
    frame_step=10,          # Sample heavily
    process_chunks=4        # Divide video into 4 parts for processing
)
results = vk.analyze('large_video.mp4')
```

### Scenario 4: Basic Chunked Processing (Memory Optimization)
```python
# Process video in chunks to reduce peak memory usage
# Still returns complete results, but uses less memory during processing
vk = VideoKurt()
vk.add_analysis('frame_diff')
vk.add_analysis('optical_flow_dense')
vk.configure(
    process_chunks=4,    # Divide video into 4 chunks
    chunk_overlap=30     # 30 frame overlap for continuity
)
results = vk.analyze('large_video.mp4')  # Full results, lower peak memory
```

## Interface Principles

1. **Progressive Disclosure**: Simple tasks should be simple, complex tasks should be possible
2. **Fail Safe**: Invalid configurations should be caught early
3. **Predictable**: Same input + config = same output
4. **Extensible**: Easy to add new analyses and features
5. **Debuggable**: Clear error messages and validation
6. **Performant**: Memory and CPU usage should be controllable

## Implementation Priority

### Phase 1: Core Interface (MVP)
- `add_analysis()` with string names
- `configure()` for global preprocessing
- `analyze()` returning RawAnalysisResults
- Basic error handling

### Phase 2: Feature Integration
- `add_feature()` with auto-dependencies
- Feature computation during analysis
- `get_features()` method
- Feature-only mode

### Phase 3: Advanced Configuration
- Analysis objects with custom config
- Per-analysis preprocessing override
- Validation and dry run
- Export/import configurations

### Phase 4: Optimization
- Basic chunked processing (process_chunks parameter)
- Full streaming/chunked processing with minimal memory
- Caching system
- Batch processing
- Progressive results

## Open Questions for Discussion

1. **Should `analyze()` automatically run if no analyses are added?**
   - Option A: Raise error - explicit is better
   - Option B: Run all analyses - convenient for exploration
   - Option C: Run a default minimal set

2. **How to handle conflicting feature requirements?**
   - Example: feature A needs blur=True, feature B needs blur=False
   - Solution: Run analysis twice with different configs?
   - Or: Let user resolve manually?

3. **Should we support analysis pipelines?**
   ```python
   # Where output of one feeds into another
   vk.add_pipeline([
       'frame_diff',
       'activity_detector',  # Uses frame_diff output
       'scene_segmenter'     # Uses activity output
   ])
   ```

4. **API naming conventions:**
   - `add_analysis()` vs `with_analysis()`
   - `configure()` vs `set_config()`
   - `analyze()` vs `process()` vs `run()`

5. **Result access patterns:**
   ```python
   # Option A: Direct attribute access
   results.analyses.frame_diff.data
   
   # Option B: Dictionary style
   results['analyses']['frame_diff']['data']
   
   # Option C: Method access
   results.get_analysis('frame_diff').get_data()
   ```

## Conclusion

The proposed interface balances simplicity for basic use cases with flexibility for advanced scenarios. The builder pattern with optional chaining provides an intuitive API that can grow with user needs. Key decisions around auto-dependencies, error handling, and memory management should be finalized based on primary use cases and user feedback.