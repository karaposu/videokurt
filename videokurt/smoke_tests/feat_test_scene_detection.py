"""Scene detection example - identifies scene boundaries and transitions."""

# to run python videokurt/smoke_tests/feat_test_scene_detection.py
from videokurt import VideoKurt
import numpy as np

# Create VideoKurt instance
vk = VideoKurt()

# Configure for scene detection
vk.configure(frame_step=3, resolution_scale=0.5)  # Balance speed and accuracy

# Add scene_detection feature
vk.add_feature('scene_detection', 
               cut_threshold=0.4,  # Threshold for hard cuts
               fade_threshold=0.2,  # Threshold for fades
               min_scene_length=10)  # Minimum 10 frames per scene

print("Detecting scene boundaries in video...")
print("This analyzes visual changes to find cuts, fades, and transitions")
print()

results = vk.analyze('sample_recording.MP4')

# Get scene detection results
scenes = results.features['scene_detection'].data

print("\nScene Detection Results:")
print(f"  Data type: {type(scenes)}")

if isinstance(scenes, dict):
    # Scene boundaries
    if 'scenes' in scenes:
        scene_list = scenes['scenes']
        print(f"\n  Found {len(scene_list)} scenes")
        
        # Show first few scenes
        for i, scene in enumerate(scene_list[:10]):
            print(f"\n  Scene {i+1}:")
            print(f"    Start: frame {scene.get('start', 0)}")
            print(f"    End: frame {scene.get('end', 0)}")
            duration = scene.get('end', 0) - scene.get('start', 0) + 1
            print(f"    Duration: {duration} frames")
            
            if 'transition_in' in scene:
                print(f"    Entry: {scene['transition_in']}")
            if 'transition_out' in scene:
                print(f"    Exit: {scene['transition_out']}")
            if 'confidence' in scene:
                print(f"    Confidence: {scene['confidence']:.1%}")
    
    # Transitions
    if 'transitions' in scenes:
        transitions = scenes['transitions']
        print(f"\n  Detected {len(transitions)} transitions")
        
        # Count transition types
        transition_types = {}
        for trans in transitions:
            t_type = trans.get('type', 'unknown')
            transition_types[t_type] = transition_types.get(t_type, 0) + 1
        
        print("\n  Transition Types:")
        for t_type, count in sorted(transition_types.items(), key=lambda x: x[1], reverse=True):
            print(f"    {t_type}: {count}")
    
    # Statistics
    if 'statistics' in scenes:
        stats = scenes['statistics']
        print(f"\n  Scene Statistics:")
        print(f"    Total scenes: {stats.get('num_scenes', 0)}")
        print(f"    Average scene length: {stats.get('avg_scene_length', 0):.1f} frames")
        print(f"    Shortest scene: {stats.get('min_scene_length', 0)} frames")
        print(f"    Longest scene: {stats.get('max_scene_length', 0)} frames")

# Analyze scene characteristics
print("\n" + "="*50)
print("Scene Length Analysis:")

if isinstance(scenes, dict) and 'scenes' in scenes:
    scene_list = scenes['scenes']
    
    if len(scene_list) > 0:
        # Calculate scene durations
        durations = [s.get('end', 0) - s.get('start', 0) + 1 for s in scene_list]
        
        # Categorize by length
        very_short = sum(1 for d in durations if d < 30)  # Less than 1 second @ 30fps
        short = sum(1 for d in durations if 30 <= d < 90)  # 1-3 seconds
        medium = sum(1 for d in durations if 90 <= d < 300)  # 3-10 seconds
        long = sum(1 for d in durations if d >= 300)  # 10+ seconds
        
        print(f"\n  Scene Duration Distribution:")
        print(f"    Very short (<1s): {very_short} scenes")
        print(f"    Short (1-3s): {short} scenes")
        print(f"    Medium (3-10s): {medium} scenes")
        print(f"    Long (>10s): {long} scenes")
        
        # Find rapid cuts
        if len(durations) > 1:
            rapid_cuts = sum(1 for d in durations if d < 15)
            if rapid_cuts > 0:
                print(f"\n  Rapid cuts detected: {rapid_cuts} scenes shorter than 0.5s")

# Combine with other features for context
print("\n" + "="*50)
print("Combined analysis with transitions:")

vk2 = VideoKurt()
vk2.configure(frame_step=3, resolution_scale=0.5)

# Add scene detection with transition detection
vk2.add_feature('scene_detection')
vk2.add_feature('binary_activity')  # To correlate with motion

print("\nProcessing with scene + activity detection...")
results2 = vk2.analyze('sample_recording.MP4')

scenes = results2.features['scene_detection'].data
activity = results2.features['binary_activity'].data

# Analyze scene changes vs activity
if isinstance(scenes, dict) and 'transitions' in scenes:
    transitions = scenes['transitions']
    
    # Check activity around transitions
    active_transitions = 0
    for trans in transitions:
        frame = trans.get('frame', 0)
        # Check activity in a window around transition
        window_start = max(0, frame - 5)
        window_end = min(len(activity), frame + 5)
        if window_end > window_start:
            window_activity = activity[window_start:window_end]
            if np.mean(window_activity) > 0.5:
                active_transitions += 1
    
    print(f"\n  Scene-Activity Correlation:")
    print(f"    Total transitions: {len(transitions)}")
    print(f"    Transitions with high activity: {active_transitions}")
    if len(transitions) > 0:
        print(f"    Activity correlation: {100*active_transitions/len(transitions):.1f}%")

# Determine video type based on scene patterns
print("\n" + "="*50)
print("Video Type Assessment:")

if isinstance(scenes, dict) and 'statistics' in scenes:
    stats = scenes['statistics']
    num_scenes = stats.get('num_scenes', 0)
    avg_length = stats.get('avg_scene_length', 0)
    
    if num_scenes == 0:
        print("  Type: Single continuous shot")
    elif num_scenes < 5 and avg_length > 300:
        print("  Type: Long-form content with few cuts")
    elif num_scenes > 20 and avg_length < 60:
        print("  Type: Rapid editing style (music video, action)")
    elif num_scenes > 10 and avg_length < 150:
        print("  Type: Standard edited content")
    else:
        print("  Type: Documentary/presentation style")
    
    # Check for regular cutting pattern
    if 'scenes' in scenes and len(scenes['scenes']) > 5:
        durations = [s.get('end', 0) - s.get('start', 0) + 1 for s in scenes['scenes']]
        duration_std = np.std(durations)
        duration_mean = np.mean(durations)
        cv = duration_std / duration_mean if duration_mean > 0 else 0
        
        if cv < 0.3:
            print("  Pattern: Regular, rhythmic cutting")
        elif cv < 0.7:
            print("  Pattern: Moderate variation in scene lengths")
        else:
            print("  Pattern: Highly variable scene lengths")

print("\n✓ Scene detection complete")