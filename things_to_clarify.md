# Things to Clarify - Feature Implementation

This document outlines critical design decisions needed for implementing the feature extraction pipeline in VideoKurt.

## 1. Feature Auto-Dependency Management

### The Challenge
When a user adds a feature, it may require specific raw analyses. For example:
- `binary_activity` requires `frame_diff`
- `scene_detection` requires both `frame_diff` and `edge_canny`
- `scrolling_detection` might need `optical_flow_dense` and `edge_canny`

### Questions to Resolve

**1.1 Auto-Add Dependencies?**
```python
vk.add_feature('scene_detection')
# Should this automatically add frame_diff and edge_canny?
```
- **Option A:** Auto-add silently (convenient but hidden behavior)
- **Option B:** Auto-add with notification (print/log what was added)
- **Option C:** Fail if dependencies missing (explicit but annoying)
- **Option D:** Add only if not present, warn if already exists with different config

**1.2 Configuration Conflicts**
```python
vk.add_analysis('frame_diff', threshold=10.0)
vk.add_feature('binary_activity')  # Expects frame_diff with threshold=30.0
```
- Which threshold should be used?
- Should we run the analysis twice with different configs?
- Should features adapt to whatever analysis config exists?
- Should we maintain multiple configs per analysis?

**1.3 Circular Dependencies**
```python
# What if:
# Feature A requires Analysis X
# Analysis X configuration depends on Feature B
# Feature B requires Feature A
```
- How to detect and handle circular dependencies?
- Should we prevent them at design time or runtime?

**1.4 Missing Optional Dependencies**
```python
# scene_detection works better with optical_flow but doesn't require it
vk.add_feature('scene_detection')
```
- Should optional dependencies be auto-added?
- How to communicate optional vs required?
- Should there be a "recommended" mode?

## 2. Feature Storage and Computation

### The Challenge
Features extract different types of data at different granularities. How should this data be stored and accessed?

### Questions to Resolve

**2.1 Storage Structure** ✅ **DECIDED: Option C - Feature Objects**
```python
# Option A: Flat storage in RawAnalysisResults
results.features = {
    'binary_activity': np.array([0, 1, 1, 0, ...]),
    'scene_detection': {...},
    'motion_magnitude': np.array([...])
}

# Option B: Hierarchical by level
results.features = {
    'basic': {
        'binary_activity': ...,
        'motion_magnitude': ...
    },
    'middle': {
        'activity_bursts': ...
    },
    'advanced': {
        'scene_detection': ...
    }
}

# ✅ Option C: Feature objects - SELECTED
results.features = {
    'binary_activity': FeatureResult(
        name='binary_activity',
        data=np.array([...]),
        metadata={'threshold': 30.0},
        dtype='binary_array',
        compute_time=0.23
    )
}
# See Recommendation 2 below for full FeatureResult definition
```

**2.2 Computation Timing**
```python
# When should features be computed?

# Option A: During analyze() - all at once
results = vk.analyze('video.mp4')  # Computes everything

# Option B: Lazy - on first access
results = vk.analyze('video.mp4')  # Only raw analyses
activity = results.get_feature('binary_activity')  # Computed now

# Option C: Progressive - as available
results = vk.analyze('video.mp4', callback=on_feature_ready)
```

**2.3 Memory Management**
```python
# Features can be large. How to handle memory?

# Option A: Keep everything in memory
results.features['all_features']  # Everything loaded

# Option B: Disk caching
results.features['large_feature']  # Loads from disk if needed

# Option C: Selective storage
vk.add_feature('scene_detection', store_intermediate=False)
```

**2.4 Feature Versioning** ⏸️ **DEFERRED**
```python
# What if feature algorithm changes?

# Option A: Version in metadata
results.features['scene_detection'].version = '1.2.0'

# Option B: Different feature names
results.features['scene_detection_v2']

# Option C: No versioning (always latest)

# Decision: Not needed for initial implementation
# Can be added later if needed
```

## 3. Feature Dependencies on Other Features

### The Challenge
Some features depend on other features, not just raw analyses.

### Questions to Resolve

**3.1 Feature-to-Feature Dependencies**
```python
# activity_bursts uses binary_activity
# How to handle this chain?

vk.add_feature('activity_bursts')
# Should this auto-add binary_activity?
# What about binary_activity's dependency on frame_diff?
```

**3.2 Computation Order**
```python
# Given complex dependencies, what order to compute?

# Dependencies:
# scene_detection -> frame_diff, edge_canny
# ui_change_detection -> scene_detection, binary_activity
# app_window_switching -> ui_change_detection

# What's the optimal computation order?
```

**3.3 Partial Computation**
```python
# If a dependency fails, what happens?

vk.add_feature('ui_change_detection')
# If scene_detection fails but binary_activity succeeds
# Should ui_change_detection:
# - Fail completely?
# - Work with partial data?
# - Skip failed dependencies?
```

## 4. Feature Configuration

### The Challenge
Features have parameters that affect their behavior. How should these be managed?

### Questions to Resolve

**4.1 Configuration API**
```python
# Option A: Parameters in add_feature
vk.add_feature('binary_activity', threshold=25.0)

# Option B: Configure after adding
vk.add_feature('binary_activity')
vk.configure_feature('binary_activity', threshold=25.0)

# Option C: Feature objects
activity = BinaryActivity(threshold=25.0)
vk.add_feature(activity)

# Option D: Global config
vk.set_feature_config({
    'binary_activity': {'threshold': 25.0}
})
```

**4.2 Default Values**
```python
# Where should defaults live?

# Option A: In feature class
class BinaryActivity:
    DEFAULT_THRESHOLD = 30.0

# Option B: In configuration file
feature_defaults.yaml

# Option C: In registry
FEATURE_DEFAULTS = {
    'binary_activity': {'threshold': 30.0}
}
```

**4.3 Configuration Validation**
```python
vk.add_feature('binary_activity', threshold=-10)  # Invalid

# When to validate?
# - On add_feature()?
# - On analyze()?
# - On feature computation?
```

## 5. Error Handling

### The Challenge
Features can fail for various reasons. How should errors be handled?

### Questions to Resolve

**5.1 Failure Modes**
```python
# What should happen when:
# - Required analysis is missing
# - Feature computation throws exception
# - Invalid data shape/type
# - Insufficient frames (e.g., need 100, have 50)
```

**5.2 Error Recovery**
```python
# Option A: Fail fast
try:
    results = vk.analyze('video.mp4')
except FeatureComputationError:
    # Everything stops

# Option B: Partial results
results = vk.analyze('video.mp4')
results.failed_features  # List of failures
results.features  # Contains successful ones

# Option C: Default/fallback values
results = vk.analyze('video.mp4')
results.features['failed_feature']  # Returns None or default
```

**5.3 Error Reporting**
```python
# How to communicate what went wrong?

# Option A: Exceptions
raise FeatureError("binary_activity failed: insufficient frames")

# Option B: Error object
results.errors['binary_activity'] = ErrorInfo(...)

# Option C: Logging
logger.error("Feature binary_activity failed")

# Option D: Callback
vk.on_feature_error = lambda f, e: print(f"{f}: {e}")
```

## 6. Feature Output Formats

### The Challenge
Features produce different types of output. How to standardize?

### Questions to Resolve

**6.1 Output Standardization**
```python
# Different features return different types:
# - binary_activity: np.array of 0/1
# - scene_detection: Dict with boundaries and confidence
# - motion_trajectories: List of trajectory objects

# Should we standardize?
# Option A: Keep native types
# Option B: Wrap in FeatureResult
# Option C: Convert to common format
```

**6.2 Metadata Storage**
```python
# Where to store feature metadata?

# Option A: Separate from data
results.features['binary_activity']  # Just the array
results.feature_metadata['binary_activity']  # Parameters, timing, etc.

# Option B: Together
results.features['binary_activity'] = {
    'data': np.array([...]),
    'metadata': {...}
}

# Option C: Feature objects
results.features['binary_activity'] = FeatureResult(
    data=...,
    metadata=...
)
```

**6.3 Temporal Alignment**
```python
# Features may have different temporal resolutions:
# - binary_activity: 1 value per frame
# - activity_bursts: sparse events
# - scene_detection: boundaries only

# How to handle alignment?
# - Interpolate to common timeline?
# - Keep native resolution?
# - Provide alignment utilities?
```

## 7. Feature Composition and Combination

### The Challenge
Users may want to combine multiple features or create custom features.

### Questions to Resolve

**7.1 Custom Features**
```python
# How to support user-defined features?

# Option A: Inherit from BaseFeature
class MyCustomFeature(BasicFeature):
    def _compute_basic(self, data):
        return ...

# Option B: Function registration
@register_feature('my_feature')
def compute_my_feature(analyses):
    return ...

# Option C: Lambda/callable
vk.add_custom_feature('my_feature', 
    lambda data: data['frame_diff'].mean())
```

**7.2 Feature Combination**
```python
# Combining multiple features into new ones

# Option A: Explicit combination
combined = vk.combine_features(
    ['binary_activity', 'motion_magnitude'],
    lambda a, m: a * m
)

# Option B: Feature expressions
vk.add_feature_expression(
    'activity_intensity',
    'binary_activity * motion_magnitude'
)
```

## 8. Performance Optimization

### The Challenge
Feature computation can be expensive. How to optimize?

### Questions to Resolve

**8.1 Parallel Computation**
```python
# Should features compute in parallel?
# - Dependencies complicate parallelization
# - Some features are CPU-intensive
# - Memory constraints
```

**8.2 Caching Strategy**
```python
# What to cache and where?

# Option A: Memory cache
# - Fast but limited size
# - Lost on restart

# Option B: Disk cache
# - Persistent
# - Slower

# Option C: Hybrid
# - Memory for recent/small
# - Disk for large/persistent
```

**8.3 Incremental Computation**
```python
# For video streams or long videos:
# - Compute features on chunks?
# - Update features incrementally?
# - Sliding window approach?
```

## Proposed Solutions

### Recommendation 1: Dependency Management
Use **Option B** - Auto-add with notification:
```python
vk.add_feature('scene_detection')
# Output: "Auto-adding required analyses: frame_diff, edge_canny"
```

### Recommendation 2: Storage Structure  
**DECIDED: Use Option C - Feature objects with metadata**

This provides the cleanest and most extensible approach:
```python
@dataclass
class FeatureResult:
    """Container for computed feature data."""
    name: str
    data: Any  # np.array, dict, list, etc.
    metadata: Dict[str, Any]  # parameters, timing, etc.
    dtype: str  # Description of data type
    shape: Optional[tuple] = None  # For array data
    compute_time: float = 0.0
    required_analyses: List[str] = field(default_factory=list)
    
    def __repr__(self):
        return f"FeatureResult({self.name}, shape={self.shape})"

# Usage:
results.features['binary_activity'] = FeatureResult(
    name='binary_activity',
    data=np.array([0, 1, 1, 0, 1]),
    metadata={'threshold': 30.0, 'activity_threshold': 0.1},
    dtype='binary_array',
    shape=(5,),
    compute_time=0.23,
    required_analyses=['frame_diff']
)
```

**Benefits of Feature Objects:**
- Clean API for accessing data and metadata
- Type safety with dataclasses
- Easy serialization for caching
- Extensible without breaking existing code
- Self-documenting with clear attributes

### Recommendation 3: Computation Timing
Use **Option A** - Compute during analyze() for simplicity:
```python
results = vk.analyze('video.mp4')  # Everything computed here
```

### Recommendation 4: Error Handling
Use **Option B** - Partial results with error tracking:
```python
results = vk.analyze('video.mp4')
if results.failed_features:
    print(f"Failed: {results.failed_features}")
# But can still use successful features
```

## Next Steps

1. **Make decisions** on each question above
2. **Update interface_discussion.md** with decisions
3. **Implement core feature pipeline** with chosen approach
4. **Test with 2-3 features** to validate design
5. **Refine based on testing** experience

## Priority Decisions Needed

**Must decide now (blocks implementation):**
1. Auto-dependency behavior
2. Feature storage structure
3. Computation timing

**Can defer (iterate later):**
1. Custom features
2. Caching strategy
3. Parallel computation

**Nice to have (future enhancement):**
1. Feature versioning
2. Feature combination
3. Incremental computation