# How Segments Should Be in VideoKurt

## Core Philosophy: Primitive Visual Patterns, Not User Intent

VideoKurt must detect **what the screen shows**, not **what the user is doing**. This is the fundamental principle that guides all segment definitions.

---

## Why Current Segments Are Wrong

The current SegmentType enum violates VideoKurt's core principle:

```python
# ❌ WRONG - These infer user intent:
SCROLLING       # Assumes user is scrolling
CLICKING        # Assumes user clicked
TYPING          # Assumes keyboard input
VIDEO_PLAYING   # Assumes media playback
```

These are **semantic interpretations**, not visual primitives. VideoKurt cannot know:
- If movement is from scrolling vs auto-animation
- If changes are from clicking vs programmatic updates
- If text appears from typing vs paste vs auto-complete
- If motion is video vs animated GIF vs canvas animation

---

## Correct Primitive Segments

Segments should describe **pure visual patterns** without interpretation:

### Motion-Based Primitives

**Directional Motion Patterns:**
```
VERTICAL_SLIDE_UP     # Content moving upward
VERTICAL_SLIDE_DOWN   # Content moving downward
HORIZONTAL_SLIDE_LEFT # Content moving leftward
HORIZONTAL_SLIDE_RIGHT # Content moving rightward
DIAGONAL_MOTION       # Non-axis-aligned movement
RADIAL_MOTION        # Expanding/contracting from center
```

**Motion Characteristics:**
```
UNIFORM_MOTION       # All regions moving together
SCATTERED_MOTION     # Different regions moving differently
ACCELERATING_MOTION  # Motion speed increasing
DECELERATING_MOTION  # Motion speed decreasing
OSCILLATING_MOTION   # Back-and-forth movement
```

### Change-Based Primitives

**Change Magnitude:**
```
FULL_CHANGE         # 80%+ of frame changed
PARTIAL_CHANGE      # 20-80% of frame changed
LOCALIZED_CHANGE    # 5-20% of frame changed
MINI_CHANGE         # 1-5% of frame changed
NO_CHANGE          # <1% change (idle)
```

**Change Patterns:**
```
INSTANT_CHANGE      # Abrupt change in 1-2 frames
GRADUAL_CHANGE      # Change over many frames
FLICKERING_CHANGE   # Rapid on/off changes
PULSING_CHANGE      # Rhythmic changes
```

### Spatial Primitives

**Change Location:**
```
CENTER_ACTIVITY     # Activity in central region
EDGE_ACTIVITY       # Activity at screen edges
CORNER_ACTIVITY     # Activity in corner
DISTRIBUTED_ACTIVITY # Activity across entire frame
BANDED_ACTIVITY     # Activity in horizontal/vertical band
```

**Region Characteristics:**
```
SINGLE_REGION       # One contiguous area changing
MULTIPLE_REGIONS    # Several separate areas changing
GROWING_REGION      # Expanding area of change
SHRINKING_REGION    # Contracting area of change
MOVING_REGION       # Region translating across screen
```

### Frequency Primitives

**Change Frequency:**
```
HIGH_FREQUENCY      # >10 Hz changes (video/animation)
MEDIUM_FREQUENCY    # 2-10 Hz (UI animations)
LOW_FREQUENCY       # <2 Hz (slow transitions)
```

### Boundary Primitives

**Boundary Sharpness:**
```
SHARP_BOUNDARIES    # Clear edges (UI elements)
FUZZY_BOUNDARIES    # Gradual transitions (shadows, gradients)
NO_BOUNDARIES       # Uniform change across region
```

**Boundary Stability:**
```
STABLE_BOUNDARIES   # Edges stay in same place
MOVING_BOUNDARIES   # Edges translate
MORPHING_BOUNDARIES # Edges change shape
```

---

## Combining Primitives for Rich Descriptions

Instead of a single SegmentType, segments could have multiple primitive attributes:

```python
@dataclass
class PrimitiveSegment:
    # Temporal
    start_frame: int
    end_frame: int
    
    # Motion primitives
    motion_pattern: Optional[MotionPattern]  # VERTICAL_SLIDE_UP, etc.
    motion_characteristic: Optional[MotionCharacteristic]  # UNIFORM, SCATTERED, etc.
    motion_magnitude: float  # Average pixels/frame
    
    # Change primitives
    change_magnitude: ChangeMagnitude  # FULL, PARTIAL, LOCALIZED, etc.
    change_pattern: Optional[ChangePattern]  # INSTANT, GRADUAL, etc.
    change_percentage: float  # Percent of frame
    
    # Spatial primitives
    spatial_pattern: Optional[SpatialPattern]  # CENTER, EDGE, etc.
    region_count: int  # Number of distinct regions
    
    # Frequency primitives
    change_frequency: Optional[ChangeFrequency]  # HIGH, MEDIUM, LOW
    
    # Boundary primitives
    boundary_sharpness: Optional[BoundarySharpness]  # SHARP, FUZZY, NONE
    boundary_stability: Optional[BoundaryStability]  # STABLE, MOVING, MORPHING
    
    # Confidence
    confidence: float
```

---

## Examples: How Primitives Describe Common Patterns

### User Scrolling Down a Webpage
**Primitives detected:**
- VERTICAL_SLIDE_UP (content moves up)
- UNIFORM_MOTION (all elements move together)
- PARTIAL_CHANGE (scrollable area only)
- CENTER_ACTIVITY (main content area)
- MEDIUM_FREQUENCY (smooth scrolling at 2-10 Hz)
- SHARP_BOUNDARIES (UI elements maintain edges)
- MOVING_BOUNDARIES (content edges translate)

**Note:** VideoKurt doesn't know this is "scrolling" - just describes the visual pattern

### Modal Dialog Appearing
**Primitives detected:**
- INSTANT_CHANGE (appears in 1-2 frames)
- LOCALIZED_CHANGE (dialog area only)
- CENTER_ACTIVITY (typically centered)
- SINGLE_REGION (one rectangle)
- SHARP_BOUNDARIES (clear UI edges)
- STABLE_BOUNDARIES (doesn't change shape)

**Note:** VideoKurt doesn't know it's a "modal" - just sees a rectangular region with sharp, stable boundaries appear

### Video Playing in Corner
**Primitives detected:**
- SCATTERED_MOTION (pixels change independently)
- LOCALIZED_CHANGE (video region only)
- CORNER_ACTIVITY (if in corner)
- HIGH_FREQUENCY (>10 Hz pixel changes)
- FUZZY_BOUNDARIES (natural video content)
- MORPHING_BOUNDARIES (video content changes)

**Note:** VideoKurt doesn't know it's "video" - just sees rapid localized changes with fuzzy, morphing boundaries

### Page Transition
**Primitives detected:**
- FULL_CHANGE (entire screen changes)
- INSTANT_CHANGE (happens quickly)
- DISTRIBUTED_ACTIVITY (whole screen)

**Note:** VideoKurt doesn't know it's a "page change" - just sees full-frame replacement

---

## Benefits of Primitive Approach

1. **No False Assumptions**: Never incorrectly labels auto-scroll as user scroll
2. **Universal Application**: Works on any UI (web, mobile, desktop, games)
3. **Composable**: Primitives combine to describe complex patterns
4. **Measurable**: Each primitive has clear visual criteria
5. **Extensible**: New primitives can be added without breaking existing ones

---

## Implementation Priority

### Phase 1: Core Motion & Change
Start with the most useful primitives:
- VERTICAL_SLIDE_UP/DOWN
- HORIZONTAL_SLIDE_LEFT/RIGHT
- FULL_CHANGE/PARTIAL_CHANGE/LOCALIZED_CHANGE
- NO_CHANGE

### Phase 2: Motion Characteristics
Add detail about motion quality:
- UNIFORM_MOTION vs SCATTERED_MOTION
- ACCELERATING vs DECELERATING

### Phase 3: Spatial Patterns
Add location information:
- CENTER/EDGE/CORNER_ACTIVITY
- Region counting and tracking

### Phase 4: Advanced Patterns
Add sophisticated detection:
- OSCILLATING_MOTION
- PULSING_CHANGE
- GROWING/SHRINKING_REGION
- Frequency analysis (HIGH/MEDIUM/LOW_FREQUENCY)
- Boundary detection (SHARP/FUZZY boundaries)
- Boundary tracking (STABLE/MOVING/MORPHING)

---

## Key Insight: Let Users Interpret

VideoKurt provides the primitive building blocks. Users/applications can then interpret these patterns based on their context:

```python
# VideoKurt provides:
primitives = {
    'motion': 'VERTICAL_SLIDE_UP',
    'characteristic': 'UNIFORM_MOTION',
    'magnitude': 'PARTIAL_CHANGE'
}

# User's application interprets:
if primitives['motion'] == 'VERTICAL_SLIDE_UP' and context.is_scrollable_ui:
    user_action = "User scrolled down"
```

This separation keeps VideoKurt pure and universal while still enabling semantic interpretation when needed.

---

## Conclusion

By focusing on primitive visual patterns rather than inferred user actions, VideoKurt becomes:
- More accurate (no incorrect assumptions)
- More universal (works everywhere)
- More honest (describes what it actually sees)
- More useful (provides raw data for any interpretation)

The key is to resist the temptation to be "smart" about what the patterns mean, and instead be extremely accurate about what the patterns **are**.