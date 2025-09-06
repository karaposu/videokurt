"""Scrolling detection example - identifies scrolling patterns in screen recordings."""

# to run python videokurt/smoke_tests/feat_test_scrolling.py

from videokurt import VideoKurt
import numpy as np

# Create VideoKurt instance
vk = VideoKurt()

# Configure for faster processing of screen recording
vk.configure(frame_step=3, resolution_scale=0.5)

# Add scrolling detection (requires optical_flow_dense)
vk.add_feature('scrolling_detection', 
               consistency_threshold=0.7,  # How uniform the flow must be
               min_scroll_frames=5)  # Minimum frames for a scroll event

print("Analyzing video for scrolling patterns...")
results = vk.analyze('sample_recording.MP4')

# Get scrolling detection results
scrolling = results.features['scrolling_detection'].data

print("\nScrolling Detection Results:")
print(f"  Data type: {type(scrolling)}")

if isinstance(scrolling, dict):
    # Extract scroll events
    if 'scroll_events' in scrolling:
        events = scrolling['scroll_events']
        print(f"  Found {len(events)} scrolling events")
        
        for i, event in enumerate(events[:5]):  # Show first 5 events
            start = event.get('start_frame', 0)
            end = event.get('end_frame', 0)
            duration = end - start + 1 if end >= start else 0
            
            print(f"\n  Event {i+1}:")
            print(f"    Start frame: {start}")
            print(f"    End frame: {end}")
            print(f"    Duration: {duration} frames")
            print(f"    Direction: {event.get('direction', 'N/A')}")
            if 'avg_speed' in event:
                print(f"    Avg speed: {event['avg_speed']:.2f} pixels/frame")
    
    # Scrolling statistics
    if 'statistics' in scrolling:
        stats = scrolling['statistics']
        print(f"\n  Overall Statistics:")
        print(f"    Total scroll frames: {stats.get('total_scroll_frames', 0)}")
        print(f"    Vertical scrolls: {stats.get('vertical_scrolls', 0)}")
        print(f"    Horizontal scrolls: {stats.get('horizontal_scrolls', 0)}")
        
    # Frame-by-frame scroll indicators
    if 'is_scrolling' in scrolling:
        is_scrolling = scrolling['is_scrolling']
        scroll_ratio = np.mean(is_scrolling)
        print(f"\n  {scroll_ratio*100:.1f}% of frames contain scrolling")

# Combine with other features for context
print("\n" + "="*50)
print("Combined analysis with motion features:")

vk2 = VideoKurt()
vk2.configure(frame_step=3, resolution_scale=0.5)

vk2.add_feature('scrolling_detection')
vk2.add_feature('binary_activity')
vk2.add_feature('stability_score')

print("\nProcessing with multiple features...")
results2 = vk2.analyze('sample_recording.MP4')

scrolling2 = results2.features['scrolling_detection'].data
activity = results2.features['binary_activity'].data
stability = results2.features['stability_score'].data

# Analyze scrolling vs other motion
if 'is_scrolling' in scrolling2:
    is_scrolling = scrolling2['is_scrolling']
    
    # Align arrays if needed
    min_len = min(len(is_scrolling), len(activity))
    is_scrolling = is_scrolling[:min_len]
    activity = activity[:min_len]
    
    # Types of activity
    scrolling_frames = np.sum(is_scrolling)
    other_motion = np.sum(activity & ~is_scrolling)
    no_motion = np.sum(~activity)
    
    print(f"\nActivity Breakdown:")
    print(f"  Scrolling: {scrolling_frames} frames")
    print(f"  Other motion: {other_motion} frames")
    print(f"  No motion: {no_motion} frames")
    
    # Scrolling stability
    if scrolling_frames > 0:
        scroll_stability = stability[is_scrolling].mean() if np.any(is_scrolling) else 0
        print(f"\n  Average stability during scrolling: {scroll_stability:.2f}")