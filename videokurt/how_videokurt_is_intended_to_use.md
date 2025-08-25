# How VideoKurt is Intended to Use

## Quick Start

```python
from videokurt import VideoKurt

# Initialize with default calibration
vk = VideoKurt()

# Analyze a video
results = vk.analyze("path/to/video.mp4")

# Get activity timeline and events
activity_timeline = results["activity"]
detected_events = results["events"]
```

## Core Usage Patterns

### 1. Basic Video Analysis

```python
# Simplest usage - auto-detect everything
vk = VideoKurt()
analysis = vk.analyze("recording.mp4")

# Check when things happened
for period in analysis["activity"]:
    if period["active"]:
        print(f"Activity from {period['start']}s to {period['end']}s")
```

### 2. Calibrated Detection for Specific Use Cases

```python
# For mobile app recordings
vk = VideoKurt(preset="mobile_app")

# For desktop screen recordings  
vk = VideoKurt(preset="desktop")

# For presentation/slides
vk = VideoKurt(preset="presentation")

# For app navigation tracking (Advanced)
vk = VideoKurt(preset="app_navigation")

# Custom calibration
vk = VideoKurt(calibration={
    "scene_change": {"threshold": 0.2},  # More sensitive
    "scroll": {"min_velocity": 30},      # Detect slower scrolls
    "idle_wait": {"min_duration": 2.0}   # Longer idle threshold
})
```

### 3. Performance-Optimized Processing

```python
# Fast mode for quick overview
vk = VideoKurt(mode="fast")
quick_analysis = vk.analyze("long_video.mp4")

# Thorough mode for detailed analysis
vk = VideoKurt(mode="thorough")
detailed_analysis = vk.analyze("important_recording.mp4")

# With specific optimization settings
vk = VideoKurt(
    resolution="720p",      # Downscale for speed
    skip_frames=2,          # Process every 3rd frame
    parallel_processing=True # Use multiple cores
)
```

## Integration with VideoQuery

### Primary Use Case: Frame Selection

```python
# VideoKurt tells VideoQuery WHEN to look
vk = VideoKurt()
timeline = vk.analyze(video_path)

# VideoQuery uses this to sample frames intelligently
frames_to_analyze = []
for period in timeline["activity"]:
    if period["active"]:
        # Extract frames from active periods
        frames_to_analyze.extend(
            extract_frames(video_path, period["start"], period["end"])
        )
```

### Event-Driven Analysis

```python
# VideoKurt identifies WHAT happened mechanically
events = vk.analyze(video_path)["events"]

# VideoQuery interprets what it means semantically
for event in events:
    if event["type"] == "scene_change":
        # VideoQuery: "User navigated to new screen"
        analyze_new_screen(video_path, event["start"])
    
    elif event["type"] == "scroll":
        # VideoQuery: "User searching through content"
        check_for_target_content(video_path, event["start"], event["end"])
    
    elif event["type"] == "popup":
        # VideoQuery: "Dialog appeared, read its contents"
        extract_dialog_text(video_path, event["start"])
```

### Segmentation for Batch Processing

```python
# VideoKurt segments video into logical chunks
segments = vk.analyze(video_path)["segments"]

# VideoQuery processes each segment based on activity
for segment in segments:
    if segment["activity_score"] > 0.7:
        # High activity - detailed analysis
        detailed_llm_analysis(segment)
    elif segment["activity_score"] > 0.3:
        # Medium activity - standard analysis
        standard_llm_analysis(segment)
    else:
        # Low activity - skip or minimal analysis
        skip_or_quick_check(segment)
```

## Advanced Usage

### 1. Custom Event Detection

```python
# Add custom loading indicator patterns
vk = VideoKurt()
vk.add_custom_pattern(
    name="custom_spinner",
    template_path="./templates/app_spinner.png",
    event_type="prompted_wait"
)

# Detect platform-specific UI elements
vk.add_custom_pattern(
    name="instagram_story_ring",
    template_path="./templates/ig_story.png",
    event_type="ui_element"
)
```

### 1.5 Image Detection (Advanced)

```python
# Detect specific images/UI elements in video
vk = VideoKurt()

# Single image detection
vk.detect_image(
    video_path="recording.mp4",
    image_path="button_screenshot.png",
    threshold=0.85  # Confidence threshold
)
# Returns: [{"timestamp": 5.2, "confidence": 0.92, "location": (x, y, w, h)}, ...]

# Multiple image detection
images_to_find = [
    {"path": "login_button.png", "name": "login"},
    {"path": "error_dialog.png", "name": "error"},
    {"path": "success_checkmark.png", "name": "success"}
]

detections = vk.detect_images(
    video_path="app_test.mp4",
    images=images_to_find,
    method="feature_matching"  # More robust than template matching
)

# Output includes all appearances
for detection in detections:
    print(f"{detection['name']} found at {detection['timestamp']}s")

# Track UI element lifecycle
element_timeline = vk.track_element(
    video_path="recording.mp4",
    element_image="popup_dialog.png",
    track_lifecycle=True  # Track appear/disappear events
)
# Returns: {"first_appearance": 3.2, "last_appearance": 8.7, "total_duration": 5.5}
```

### 1.6 Persistent UI Frame Detection (Advanced)

```python
# Detect and track persistent navigation elements
vk = VideoKurt(preset="app_navigation")

# Analyze app with bottom navigation
analysis = vk.analyze("instagram_recording.mp4")

# Extract persistent UI information
for event in analysis["events"]:
    if event["type"] == "persistent_ui":
        print(f"App UI detected: {event['metadata']['app_identifier']}")
        print(f"Active tab: {event['metadata']['active_section']}")
        print(f"Navigation type: {event['metadata']['layout_type']}")

# Track navigation changes within same app
nav_changes = vk.track_navigation_state(
    video_path="app_recording.mp4",
    reference_ui={
        "bottom_nav": "templates/instagram_bottom_nav.png",
        "top_bar": "templates/instagram_header.png"
    }
)
# Returns timeline of tab/section changes

# Detect app switches based on UI chrome changes
app_sessions = vk.detect_app_sessions(
    video_path="multi_app_recording.mp4",
    ui_templates=[
        {"name": "instagram", "template": "templates/ig_nav.png"},
        {"name": "twitter", "template": "templates/twitter_nav.png"},
        {"name": "tiktok", "template": "templates/tiktok_nav.png"}
    ]
)
# Returns: [{"app": "instagram", "start": 0, "end": 45}, {"app": "twitter", "start": 45, "end": 120}]

# Monitor specific UI regions for stability
vk.monitor_ui_regions(
    video_path="app_test.mp4",
    regions={
        "bottom": {"y": 0.85, "height": 0.15},  # Bottom 15% of screen
        "top": {"y": 0, "height": 0.1}          # Top 10% of screen
    },
    stability_threshold=0.9  # Consider stable if 90% similar across frames
)
```

### 2. Real-time Analysis (Streaming)

```python
# Process video in chunks as it's being recorded
vk = VideoKurt(streaming=True)

for chunk in video_stream:
    partial_analysis = vk.analyze_chunk(chunk)
    if partial_analysis["activity"][-1]["active"]:
        # Trigger immediate processing if activity detected
        process_immediately(chunk)
```

### 3. Multi-pass Analysis

```python
# First pass: Quick activity detection
vk_fast = VideoKurt(mode="fast")
overview = vk_fast.analyze(video_path)

# Second pass: Detailed analysis of active regions only
vk_detailed = VideoKurt(mode="thorough")
for period in overview["activity"]:
    if period["active"]:
        detailed = vk_detailed.analyze_segment(
            video_path, 
            start=period["start"], 
            end=period["end"]
        )
```

### 4. Export for Manual Review

```python
# Generate visualization of activity
vk = VideoKurt()
analysis = vk.analyze(video_path)

# Export timeline visualization
vk.export_timeline(
    analysis,
    output_path="timeline.html",
    include_thumbnails=True
)

# Export key frames from events
vk.export_event_frames(
    analysis,
    output_dir="./event_frames/",
    events_filter=["scene_change", "popup"]
)
```

## Common Patterns

### Pattern 1: Skip Idle Periods

```python
def process_video_efficiently(video_path):
    # Detect idle zones
    vk = VideoKurt()
    analysis = vk.analyze(video_path)
    
    # Only process active periods
    for period in analysis["activity"]:
        if period["active"]:
            process_segment(video_path, period["start"], period["end"])
        else:
            log(f"Skipping idle period: {period['start']}-{period['end']}")
```

### Pattern 2: Event-Triggered Actions

```python
def monitor_for_specific_events(video_path, target_events):
    vk = VideoKurt()
    analysis = vk.analyze(video_path)
    
    triggers = []
    for event in analysis["events"]:
        if event["type"] in target_events:
            triggers.append({
                "timestamp": event["start"],
                "event": event["type"],
                "confidence": event["confidence"]
            })
    
    return triggers
```

### Pattern 3: Adaptive Sampling

```python
def adaptive_frame_extraction(video_path):
    vk = VideoKurt()
    analysis = vk.analyze(video_path)
    
    frames = []
    for segment in analysis["segments"]:
        # Sample rate based on activity intensity
        if segment["activity_score"] > 0.8:
            sample_rate = 5  # 5 fps for high activity
        elif segment["activity_score"] > 0.4:
            sample_rate = 2  # 2 fps for medium
        else:
            sample_rate = 0.5  # 0.5 fps for low
        
        frames.extend(
            extract_frames_at_rate(
                video_path, 
                segment["start"], 
                segment["end"],
                sample_rate
            )
        )
    
    return frames
```

## Best Practices

### 1. Choose the Right Preset

- **mobile_app**: Touch interactions, swipes, app transitions
- **desktop**: Mouse movements, window changes, keyboard input
- **presentation**: Slide changes, minimal motion
- **gaming**: High motion, rapid changes
- **surveillance**: Long idle periods, motion detection
- **app_navigation**: Persistent UI tracking, navigation state monitoring

### 2. Calibrate for Your Content

```python
# Too many false positives?
vk = VideoKurt(calibration={
    "scene_change": {"threshold": 0.5}  # Increase threshold
})

# Missing important events?
vk = VideoKurt(calibration={
    "scene_change": {"threshold": 0.2}  # Decrease threshold
})
```

### 3. Use Activity Scores

```python
# Don't just check active/inactive
for segment in analysis["segments"]:
    if segment["activity_score"] > 0.9:
        # Critical moment - maximum sampling
    elif segment["activity_score"] > 0.5:
        # Moderate activity - normal sampling
    else:
        # Low activity - minimal sampling
```

### 4. Combine Events for Context

```python
# A scroll followed by a pause might indicate the user found something
prev_event = None
for event in analysis["events"]:
    if prev_event and prev_event["type"] == "scroll" and event["type"] == "idle_wait":
        # User likely found what they were looking for
        investigate_frame(video_path, event["start"])
    prev_event = event
```

## Performance Tips

### 1. Pre-process Long Videos

```python
# For videos > 10 minutes, do a quick pass first
if video_duration > 600:
    vk_quick = VideoKurt(mode="fast", skip_frames=5)
    overview = vk_quick.analyze(video_path)
    
    # Then detailed analysis only on interesting parts
    interesting_segments = [s for s in overview["segments"] 
                          if s["activity_score"] > 0.3]
```

### 2. Cache Analysis Results

```python
# VideoKurt analysis can be cached
import pickle

cache_file = f"{video_path}.vk_cache"
if os.path.exists(cache_file):
    with open(cache_file, 'rb') as f:
        analysis = pickle.load(f)
else:
    vk = VideoKurt()
    analysis = vk.analyze(video_path)
    with open(cache_file, 'wb') as f:
        pickle.dump(analysis, f)
```

### 3. Use GPU When Available

```python
# Enable GPU acceleration if available
vk = VideoKurt(use_gpu=True)

# Check if GPU is being used
if vk.gpu_available:
    print("GPU acceleration enabled")
```

## Error Handling

```python
from videokurt import VideoKurt, VideoKurtError

try:
    vk = VideoKurt()
    analysis = vk.analyze("video.mp4")
except VideoKurtError as e:
    if e.error_type == "corrupted_video":
        # Try to recover with more robust settings
        vk = VideoKurt(error_recovery=True)
        analysis = vk.analyze("video.mp4", skip_corrupted_frames=True)
    elif e.error_type == "unsupported_format":
        # Convert video first
        convert_video("video.mp4", "video_converted.mp4")
    else:
        raise
```

## Testing and Validation

```python
# Validate detection accuracy
vk = VideoKurt()
analysis = vk.analyze("test_video.mp4")

# Generate report for manual verification
vk.generate_validation_report(
    analysis,
    output_path="validation_report.html",
    include_confidence_scores=True,
    include_debug_frames=True
)

# Compare with ground truth
ground_truth = load_annotations("test_video_annotations.json")
accuracy = vk.validate_against_ground_truth(analysis, ground_truth)
print(f"Detection accuracy: {accuracy:.2%}")
```

## Summary

VideoKurt is designed to be:
1. **The eyes** that watch the video and detect mechanical changes
2. **The filter** that identifies when interesting things happen  
3. **The optimizer** that helps VideoQuery work efficiently

It doesn't understand meaning - it just knows when pixels change in interesting ways. VideoQuery takes these mechanical observations and interprets them semantically with LLM assistance.

Together, VideoKurt's mechanical detection + VideoQuery's semantic understanding = Complete video intelligence.