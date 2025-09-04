# Further Discussion - Feature Architecture

## Core Understanding

### Features are NOT just views - they are higher-level computations

Initially, there was a misconception that features are merely "views" or "interpretations" of raw analysis data. This is incorrect. Features are **sophisticated computations** that:

1. **Combine multiple raw analyses** - Not just reformatting one analysis
2. **Apply domain-specific algorithms** - Scene detection uses complex boundary detection
3. **Maintain temporal state** - Activity bursts track patterns over time
4. **Produce new information** - Not present in any single raw analysis

Example:
```python
# This is NOT just a view:
scene_detection = complex_algorithm(
    frame_diff + edge_canny + color_histogram + temporal_patterns
)

# It's a genuine computation producing new insights
```

## The Multi-Path Problem

### One Feature, Multiple Computation Strategies

A critical architectural challenge: **The same feature can be computed through different analysis combinations**, each with different trade-offs.

### Example: Scroll Detection

Scroll detection can be computed using multiple approaches:

#### Strategy 1: Fast & Simple
```python
# Uses: optical_flow_sparse (fast)
# Pros: Quick, low memory
# Cons: Might miss subtle scrolls
# Accuracy: ~70%
# Speed: 5ms per frame
```

#### Strategy 2: Robust & Accurate  
```python
# Uses: optical_flow_dense + edge_canny
# Pros: Very accurate, handles complex content
# Cons: Slow, memory intensive
# Accuracy: ~95%
# Speed: 50ms per frame
```

#### Strategy 3: Specialized
```python
# Uses: frame_diff_advanced + texture_descriptors + motion_heatmap
# Pros: Great for text documents
# Cons: Fails on image-heavy content
# Accuracy: ~85% (99% on text)
# Speed: 20ms per frame
```

#### Strategy 4: Hybrid Intelligence
```python
# Uses: ALL available analyses with ML model
# Pros: Best accuracy
# Cons: Requires all analyses, very slow
# Accuracy: ~99%
# Speed: 100ms+ per frame
```

## User Choice Requirements

Users should be able to choose their strategy based on their needs:

```python
# Future API concept:
vk = VideoKurt()

# Option 1: Let VideoKurt choose
vk.detect('scrolling')  # Uses default strategy

# Option 2: User specifies preference
vk.detect('scrolling', strategy='fast')      # Quick but less accurate
vk.detect('scrolling', strategy='accurate')  # Slow but precise
vk.detect('scrolling', strategy='balanced')  # Middle ground

# Option 3: User specifies constraints
vk.detect('scrolling', max_time_ms=10)  # VideoKurt picks fastest strategy
vk.detect('scrolling', min_accuracy=0.9)  # VideoKurt picks most accurate

# Option 4: User specifies exact analyses
vk.detect('scrolling', use=['optical_flow_sparse', 'edge_canny'])
```

## Implementation Challenges

### 1. Strategy Registration
How do we register multiple strategies for the same feature?

```python
# Option A: Multiple classes
class ScrollDetectionFast(Feature):
    REQUIRED = ['optical_flow_sparse']

class ScrollDetectionAccurate(Feature):
    REQUIRED = ['optical_flow_dense', 'edge_canny']

# Option B: Single class with strategies
class ScrollDetection(Feature):
    STRATEGIES = {
        'fast': {
            'requires': ['optical_flow_sparse'],
            'compute': compute_fast
        },
        'accurate': {
            'requires': ['optical_flow_dense', 'edge_canny'],
            'compute': compute_accurate
        }
    }

# Option C: Strategy objects
class ScrollDetection(Feature):
    def __init__(self, strategy='auto'):
        self.strategy = STRATEGIES[strategy]()
```

### 2. Dependency Resolution
Different strategies need different analyses:

```python
# Problem: Which analyses to run?
vk.detect('scrolling', strategy='?')  # Don't know until runtime
vk.detect('scene_detection', strategy='?')

# Solution 1: Run all possible requirements (wasteful)
# Solution 2: Lazy analysis loading (complex)
# Solution 3: Require strategy upfront (less flexible)
```

### 3. Result Consistency
Different strategies might return different formats:

```python
# Fast strategy returns:
{
    'is_scrolling': True,
    'direction': 'down'
}

# Accurate strategy returns:
{
    'is_scrolling': True,
    'direction': 'down',
    'speed': 5.2,
    'content_type': 'text',
    'confidence': 0.95,
    'regions': [(0, 100, 800, 600)]
}

# Need unified interface
```

## Current Scope vs Future Scope

### Current Scope (MVP)
- **One working strategy per feature**
- **Simple implementation**
- **Get it working first**

```python
# For now, just:
class ScrollDetection(Feature):
    REQUIRED = ['optical_flow_dense', 'edge_canny']
    
    def compute(self, analyses):
        # One good implementation
        return detect_scrolling_v1(analyses)
```

### Future Scope (v2+)
- **Multiple strategies per feature**
- **User choice of trade-offs**
- **Auto-selection based on constraints**
- **Performance profiling**
- **Accuracy benchmarking**

## Architecture Implications

### 1. Feature Registry Needs Strategy Awareness

```python
FEATURES = {
    'scrolling': {
        'default': ScrollDetectionBalanced,
        'strategies': {
            'fast': ScrollDetectionFast,
            'accurate': ScrollDetectionAccurate,
            'balanced': ScrollDetectionBalanced
        }
    }
}
```

### 2. Results Need Strategy Metadata

```python
results.features['scrolling'] = FeatureResult(
    name='scrolling',
    data={...},
    metadata={
        'strategy_used': 'balanced',
        'accuracy_estimate': 0.85,
        'compute_time': 0.023,
        'analyses_used': ['optical_flow_dense', 'edge_canny']
    }
)
```

### 3. Configuration Gets Complex

```python
# Simple current approach:
vk.add_feature('scrolling')

# Future multi-strategy approach:
vk.add_feature('scrolling', 
    strategy='auto',
    fallback='fast',
    constraints={'max_time': 50}
)
```

## Design Decisions Needed

### For Current Implementation (MVP)

1. **Should we acknowledge multi-strategy in the architecture now?**
   - Even if we only implement one strategy
   - Pros: Easier to add later
   - Cons: Premature abstraction

2. **How to handle features that NEED multiple strategies?**
   - Some features might not work with single approach
   - Example: UI change detection varies wildly between apps

3. **Should strategies be visible in the API?**
   - Hidden: `detect('scrolling')` 
   - Visible: `detect('scrolling', strategy='default')`

### For Future Implementation

1. **How to benchmark strategies?**
   - Need ground truth data
   - Need performance metrics
   - Need accuracy metrics

2. **How to help users choose?**
   - Auto-selection?
   - Recommendation engine?
   - Profiling on sample?

3. **How to handle strategy evolution?**
   - New strategies added over time
   - Old strategies deprecated
   - Version compatibility

## Recommendation for Now

### Keep it Simple, But Don't Paint Ourselves into a Corner

1. **Implement single strategy per feature** - Get it working
2. **Use a structure that allows expansion** - Don't hard-code assumptions
3. **Document which analyses each feature uses** - For transparency
4. **Add strategy field to FeatureResult** - Even if always 'default' for now

```python
class Feature:
    """Base class that can support strategies in future."""
    
    def get_requirements(self, strategy='default'):
        """Returns required analyses for given strategy."""
        # For now, always returns same list
        return self.REQUIRED_ANALYSES
    
    def compute(self, analyses, strategy='default'):
        """Compute feature using specified strategy."""
        # For now, ignores strategy parameter
        return self._compute_default(analyses)
```

This way:
- Current implementation stays simple
- Future multi-strategy is possible without breaking changes
- Users understand that strategies exist (even if only one available)

## Examples of Other Multi-Strategy Features

### Scene Detection
- **Fast**: Just frame_diff thresholds
- **Accurate**: frame_diff + edge + color histogram
- **Cinema**: Specialized for film (shot detection, fade detection)
- **Streaming**: Optimized for live content

### Activity Detection
- **Binary**: Simple active/inactive
- **Levels**: Multiple activity levels (idle, low, medium, high)
- **Semantic**: Type of activity (typing, scrolling, video playback)

### UI Change Detection  
- **Generic**: Works on any UI
- **Web**: Optimized for browsers
- **Desktop**: Optimized for native apps
- **Mobile**: Optimized for mobile recordings

## Conclusion

The multi-strategy nature of features is a fundamental architectural consideration that we're deferring for MVP, but we must design with it in mind. Features are complex computations that can be achieved through multiple paths, each with different trade-offs. The current goal is to get one good strategy working per feature, while keeping the door open for multiple strategies in the future.