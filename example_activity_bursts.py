"""Activity bursts detection example - identifies periods of intense activity."""

# to run python example_activity_bursts.py

from videokurt import VideoKurt
import numpy as np

# Create VideoKurt instance
vk = VideoKurt()

# Configure for burst detection
vk.configure(frame_step=2, resolution_scale=0.5)

# Add activity_bursts feature (requires frame_diff)
vk.add_feature('activity_bursts', 
               burst_threshold=0.5,  # Normalized threshold for high activity
               min_burst_length=3,  # Minimum 3 frames for a burst
               smoothing_window=5)  # Smooth over 5 frames

print("Detecting activity bursts in video...")
print("This identifies periods of intense activity followed by calm")
print()

results = vk.analyze('sample_recording.MP4')

# Get burst detection results
bursts = results.features['activity_bursts'].data

print("\nActivity Burst Detection Results:")
print(f"  Data type: {type(bursts)}")

if isinstance(bursts, dict):
    # Burst events
    if 'bursts' in bursts:
        burst_list = bursts['bursts']
        print(f"\n  Found {len(burst_list)} activity bursts")
        
        # Show first few bursts
        for i, burst in enumerate(burst_list[:5]):
            print(f"\n  Burst {i+1}:")
            print(f"    Start frame: {burst.get('start', 0)}")
            print(f"    End frame: {burst.get('end', 0)}")
            duration = burst.get('end', 0) - burst.get('start', 0) + 1
            print(f"    Duration: {duration} frames")
            
            if 'intensity' in burst:
                print(f"    Intensity: {burst['intensity']:.2f}")
            if 'peak_frame' in burst:
                print(f"    Peak at frame: {burst['peak_frame']}")
            if 'peak_value' in burst:
                print(f"    Peak value: {burst['peak_value']:.3f}")
    
    # Statistics
    if 'num_bursts' in bursts:
        print(f"\n  Burst Statistics:")
        print(f"    Total bursts: {bursts['num_bursts']}")
        
    if 'burst_ratio' in bursts:
        print(f"    Burst ratio: {bursts['burst_ratio']:.1%} of video")
        
    if 'avg_burst_intensity' in bursts:
        print(f"    Average burst intensity: {bursts['avg_burst_intensity']:.2f}")
        
    if 'avg_burst_duration' in bursts:
        print(f"    Average burst duration: {bursts['avg_burst_duration']:.1f} frames")

# Analyze burst patterns
print("\n" + "="*50)
print("Burst Pattern Analysis:")

if isinstance(bursts, dict) and 'bursts' in bursts:
    burst_list = bursts['bursts']
    
    if len(burst_list) > 0:
        # Calculate inter-burst intervals
        if len(burst_list) > 1:
            intervals = []
            for i in range(1, len(burst_list)):
                interval = burst_list[i]['start'] - burst_list[i-1]['end']
                intervals.append(interval)
            
            if intervals:
                print(f"\n  Inter-burst intervals:")
                print(f"    Average: {np.mean(intervals):.1f} frames")
                print(f"    Min: {min(intervals)} frames")
                print(f"    Max: {max(intervals)} frames")
                
                # Check for regular bursting
                if np.std(intervals) < np.mean(intervals) * 0.3:
                    print("    Pattern: Regular bursting (consistent intervals)")
                else:
                    print("    Pattern: Irregular bursting (variable intervals)")
        
        # Classify burst intensities
        intensities = [b.get('intensity', 0) for b in burst_list]
        if intensities:
            high_intensity = sum(1 for i in intensities if i > np.mean(intensities))
            print(f"\n  Intensity Distribution:")
            print(f"    High intensity bursts: {high_intensity}")
            print(f"    Low intensity bursts: {len(intensities) - high_intensity}")

# Combine with other features for context
print("\n" + "="*50)
print("Combined analysis with activity patterns:")

vk2 = VideoKurt()
vk2.configure(frame_step=2, resolution_scale=0.5)

# Add burst detection with complementary features
vk2.add_feature('activity_bursts')
vk2.add_feature('temporal_activity_patterns', window_size=20)  # Temporal patterns
vk2.add_feature('stability_score')  # Content stability

print("\nProcessing with temporal analysis features...")
results2 = vk2.analyze('sample_recording.MP4')

bursts = results2.features['activity_bursts'].data
temporal = results2.features['temporal_activity_patterns'].data
stability = results2.features['stability_score'].data

# Correlate bursts with stability
print(f"\nBurst-Stability Correlation:")
if 'bursts' in bursts and len(stability) > 0:
    burst_frames = []
    for burst in bursts['bursts']:
        for f in range(burst['start'], min(burst['end'] + 1, len(stability))):
            burst_frames.append(f)
    
    if burst_frames:
        burst_stability = np.mean([stability[f] for f in burst_frames if f < len(stability)])
        non_burst_frames = [f for f in range(len(stability)) if f not in burst_frames]
        if non_burst_frames:
            non_burst_stability = np.mean([stability[f] for f in non_burst_frames])
            
            print(f"  Stability during bursts: {burst_stability:.2f}")
            print(f"  Stability outside bursts: {non_burst_stability:.2f}")
            
            if burst_stability < non_burst_stability * 0.7:
                print("  Interpretation: Bursts correlate with unstable content")
            else:
                print("  Interpretation: Bursts don't strongly affect stability")

# Interpret the activity pattern
print("\n" + "="*50)
print("Activity Pattern Interpretation:")

if isinstance(bursts, dict):
    num_bursts = bursts.get('num_bursts', 0)
    burst_ratio = bursts.get('burst_ratio', 0)
    
    if num_bursts == 0:
        print("  Pattern: Steady activity (no bursts detected)")
    elif num_bursts < 5:
        print("  Pattern: Occasional bursts")
        print("  Interpretation: Mostly calm with sporadic activity")
    elif num_bursts < 20:
        print("  Pattern: Moderate bursting")
        print("  Interpretation: Regular periods of high activity")
    else:
        print("  Pattern: Frequent bursting")
        print("  Interpretation: Highly dynamic content with many activity peaks")
    
    if burst_ratio > 0.3:
        print(f"  High burst ratio ({burst_ratio:.1%}): Significant portion in burst state")
    elif burst_ratio > 0.1:
        print(f"  Moderate burst ratio ({burst_ratio:.1%}): Balanced activity distribution")
    else:
        print(f"  Low burst ratio ({burst_ratio:.1%}): Mostly calm with brief bursts")

print("\n✓ Activity burst analysis complete")