"""App/Window switching detection - identifies when users switch between applications or windows."""

# to run python videokurt/smoke_tests/feat_test_app_switching.py

from videokurt import VideoKurt
import numpy as np

# Create VideoKurt instance
vk = VideoKurt()

# Configure for screen recording analysis
vk.configure(frame_step=2, resolution_scale=0.6)  # Higher res for better signature detection

# Add app_window_switching feature (requires multiple analyses)
vk.add_feature('app_window_switching', 
               switch_threshold=0.4,  # Sensitivity to detect switches
               signature_threshold=0.35,  # Threshold for visual signatures
               min_stability_frames=5)  # Frames to consider stable

print("Analyzing video for app/window switching patterns...")
print("This feature uses frame_diff, edge_canny, color_histogram, and dct_transform")
print()

results = vk.analyze('sample_recording.MP4')

# Get app switching results
switching = results.features['app_window_switching'].data

print("\nApp/Window Switching Detection Results:")
print(f"  Data type: {type(switching)}")

if isinstance(switching, dict):
    # Extract switch events
    if 'switch_events' in switching:
        events = switching['switch_events']
        print(f"\n  Found {len(events)} switching events")
        
        for i, event in enumerate(events[:10]):  # Show first 10 events
            print(f"\n  Switch Event {i+1}:")
            print(f"    Frame: {event.get('frame', 'N/A')}")
            print(f"    Type: {event.get('type', 'N/A')}")
            
            if 'confidence' in event:
                print(f"    Confidence: {event['confidence']:.2%}")
            
            if 'transition_type' in event:
                print(f"    Transition: {event['transition_type']}")
                
            if 'visual_change' in event:
                print(f"    Visual change: {event['visual_change']:.2%}")
    
    # App signatures (visual fingerprints of different apps/windows)
    if 'app_signatures' in switching:
        signatures = switching['app_signatures']
        print(f"\n  Detected {len(signatures)} unique app/window signatures")
        
        for i, sig in enumerate(signatures[:5]):
            print(f"\n  Signature {i+1}:")
            if 'frame_ranges' in sig:
                ranges = sig['frame_ranges']
                print(f"    Appears in {len(ranges)} segments")
                if ranges:
                    print(f"    First appearance: frames {ranges[0]}")
            
            if 'characteristics' in sig:
                chars = sig['characteristics']
                if 'dominant_color' in chars:
                    print(f"    Dominant color: {chars['dominant_color']}")
                if 'edge_density' in chars:
                    print(f"    Edge density: {chars['edge_density']:.3f}")
    
    # Statistics
    if 'statistics' in switching:
        stats = switching['statistics']
        print(f"\n  Overall Statistics:")
        print(f"    Total switches: {stats.get('total_switches', 0)}")
        print(f"    Unique apps/windows: {stats.get('unique_apps', 0)}")
        print(f"    Average time per app: {stats.get('avg_time_per_app', 0):.1f} frames")
        print(f"    Most frequent app: {stats.get('most_frequent_app', 'N/A')}")

# Summary of what was detected
if isinstance(switching, dict) and 'switch_events' in switching:
    events = switching['switch_events']
    
    # Analyze event types
    event_types = {}
    for event in events:
        evt_type = event.get('type', 'unknown')
        event_types[evt_type] = event_types.get(evt_type, 0) + 1
    
    print("\n" + "="*50)
    print("Event Type Summary:")
    for evt_type, count in sorted(event_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  {evt_type}: {count} events ({100*count/len(events):.1f}%)")
    
    # Analyze transition types
    transition_types = {}
    for event in events:
        trans_type = event.get('transition_type', 'unknown')
        transition_types[trans_type] = transition_types.get(trans_type, 0) + 1
    
    print("\nTransition Type Summary:")
    for trans_type, count in sorted(transition_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  {trans_type}: {count} events")
    
    # Find rapid switching periods
    if len(events) > 1:
        event_frames = [e.get('frame', 0) for e in events]
        frame_diffs = np.diff(event_frames)
        rapid_switches = np.sum(frame_diffs < 10)  # Switches within 10 frames
        
        print(f"\nTemporal Analysis:")
        print(f"  Rapid switches (<10 frames apart): {rapid_switches}")
        print(f"  Average frames between switches: {np.mean(frame_diffs):.1f}")
        print(f"  Median frames between switches: {np.median(frame_diffs):.1f}")

print("\n✓ App switching detection complete")