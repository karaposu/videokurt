# How Heatmap Should Be Integrated to VideoKurt

## The Core Issue

Motion heatmaps with decay factors are designed for real-time or short-duration analysis. When processing long videos, early motion gets "forgotten" due to exponential decay, making the final heatmap only reflect recent activity.

## Proposed Solution: Periodic Heatmap Snapshots

VideoKurt should capture heatmap states at regular intervals throughout video processing, creating a temporal sequence of activity maps.

## Key Design Principles

### 1. Window-Based Heatmap Generation
```python
# Conceptual approach
heatmap_interval = 30  # seconds
heatmap_snapshots = []  # List of (timestamp, heatmap) tuples

# During video processing:
# - Every 30 seconds, save current heatmap state
# - Option to reset or continue accumulation
```

### 2. Multiple Heatmap Types

**Cumulative Heatmap** (no decay):
- Shows all activity zones throughout entire video
- Useful for: Identifying scrollable regions, UI hotspots, static areas
- Use case: "Where does activity happen in this video?"

**Windowed Heatmap** (with decay):
- Shows recent activity within sliding windows
- Useful for: Detecting activity patterns, idle periods
- Use case: "What's currently active in this time segment?"

**Differential Heatmap**:
- Shows changes between time windows
- Useful for: Detecting when UI layout changes
- Use case: "When did the user move to a different screen?"

## Use Cases

### 1. Activity Timeline Visualization
Instead of just binary active/inactive, periodic heatmaps provide rich spatial-temporal data:
- Minute 0-1: Heatmap shows scrolling in center region
- Minute 1-2: Heatmap shows activity in top navigation
- Minute 2-3: Heatmap shows video playing in bottom half

### 2. Segment Detection Enhancement
Heatmaps can improve primitive segment detection:
- Concentrated motion in vertical strip → SCROLLBAR_INTERACTION
- Distributed motion across frame → FULL_SCREEN_TRANSITION
- Localized motion in corner → NOTIFICATION_POPUP

### 3. Activity Zone Identification
Periodic heatmaps reveal UI structure over time:
- Static header/footer regions (consistently low activity)
- Content areas (variable activity)
- Interactive zones (sporadic high activity)

## Integration Points with VideoKurt

### As Part of Results
```python
@dataclass
class VideoKurtResults:
    binary_activity: np.ndarray
    binary_activity_confidence: np.ndarray
    heatmap_snapshots: List[Tuple[float, np.ndarray]]  # (timestamp, heatmap)
    heatmap_cumulative: np.ndarray  # Full video cumulative
    # ... other fields
```

### Configuration Options
```python
def analyze_video(
    video_path: str,
    heatmap_interval: Optional[float] = 30.0,  # Seconds between snapshots
    heatmap_decay: float = 0.98,  # Decay factor for windowed heatmaps
    save_heatmap_images: bool = False,  # Export as PNG files
):
```

## Benefits for VideoKurt's Goals

1. **Richer Binary Activity Context**: Instead of just 1/0 for activity, we know WHERE activity occurred

2. **Better Primitive Detection**: Spatial patterns help distinguish between different types of motion (scroll vs transition vs animation)

3. **Video Summarization**: Heatmap snapshots provide visual summary of video activity without watching entire video

4. **Performance Optimization**: Can process only regions showing activity in cumulative heatmap

## Example Output Structure
```
video_analysis/
├── binary_timeline.json
├── heatmaps/
│   ├── cumulative.png          # Entire video
│   ├── snapshot_000030.png     # 30 seconds
│   ├── snapshot_000060.png     # 60 seconds
│   ├── snapshot_000090.png     # 90 seconds
│   └── ...
└── heatmap_metadata.json       # Timestamps, stats, zones
```

## Why This Matters

Periodic heatmaps transform VideoKurt from a temporal activity detector to a spatial-temporal analyzer. This aligns with the goal of detecting primitive visual patterns - not just WHEN things change, but WHERE and HOW they change across the screen over time.