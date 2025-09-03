
# Different ways how VideoKurt is can bed used

videokurt interface works in such way 


from videokurt import VideoKurt


vk = VideoKurt()

vk.add_analysis(analysis_name_here)
vk.add_analysis(another_analysis_name_here)

this way we tell videokurt which analysis we are interested. 

vk.configure(frame_step=5, resolution_scale=0.2, )  



Simple usage - just enable blur with defaults
vk.configure(blur=True)  # Uses kernel_size=13


vk.configure(blur=True, blur_kernel_size=21)  # Stronger blur

VideoKurt provides three preprocessing techniques that can be applied to any analysis:

  1. Downsampling (Temporal Reduction)

  - What: Skip frames to reduce processing load
  - Parameter: frame_step=N (process every Nth frame)
  - Example: frame_step=3 → process frames 0, 3, 6, 9...
  - Use when: Video has high frame rate or redundant frames

  2. Downscaling (Spatial Reduction)

  - What: Reduce resolution of each frame
  - Parameter: resolution_scale=0.X (fraction of original size)
  - Example: resolution_scale=0.5 → 1920×1080 becomes 960×540
  - Use when: High resolution isn't needed for detection

  3. Blur (Detail Reduction)

  - What: Apply Gaussian blur to remove fine details
  - Parameters: blur=True/False, blur_kernel_size=N (odd number)
  - Example: blur=True, blur_kernel_size=13 → smooth out text/noise
  - Use when: Small details interfere with motion detection


vk.configure(
      frame_step=2,         # Half the frames
      resolution_scale=0.5, # Quarter the pixels
      blur=True            # Remove noise
  )
  Result: 8x faster processing with cleaner motion detection


analysis_results= vk.analyze(path_to_the_video)
this is valid usecase, it will take so much space in RAM but we let user do that. 

vk.analyze(path_to_the_video)
























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