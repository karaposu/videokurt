"""Repetition detection example - finds periodic patterns in video."""

# to run python example_repetition.py
from videokurt import VideoKurt
import numpy as np

# Create VideoKurt instance
vk = VideoKurt()

# Configure for faster processing
vk.configure(frame_step=2, resolution_scale=0.5)

# Add repetition_indicator feature (requires frequency_fft analysis)
vk.add_feature('repetition_indicator', peak_threshold=0.3)

print("Analyzing video for repetitive patterns...")
results = vk.analyze('sample_recording.MP4')

# Get repetition detection results
repetition = results.features['repetition_indicator'].data

print("\nRepetition Detection Results:")
print(f"  Data type: {type(repetition)}")

if isinstance(repetition, dict):
    print(f"  Has repetition: {repetition.get('has_repetition', False)}")
    
    if repetition.get('has_repetition'):
        print(f"  Repetition score: {repetition.get('repetition_score', 0):.2%}")
        print(f"  Dominant frequency index: {repetition.get('dominant_frequency_idx', 0)}")
        print(f"  Number of frequency peaks: {repetition.get('num_peaks', 0)}")
        
        # Estimate period if repetition found
        if repetition.get('dominant_frequency_idx', 0) > 0:
            # The frequency index gives us a rough period estimate
            # Higher index = higher frequency = shorter period
            print(f"  Frequency characteristics detected")
    else:
        print("  No significant repetitive patterns detected")

# Combine with other features for deeper analysis
print("\n" + "="*50)
print("Combined analysis with motion and periodicity:")

vk2 = VideoKurt()
vk2.configure(frame_step=2, resolution_scale=0.5)

# Add multiple features to understand the nature of repetition
vk2.add_feature('repetition_indicator')
vk2.add_feature('binary_activity')
vk2.add_feature('motion_magnitude', normalize=True)
vk2.add_feature('periodicity_strength')  # More detailed periodicity analysis

print("\nProcessing with multiple periodicity features...")
results2 = vk2.analyze('sample_recording.MP4')

repetition = results2.features['repetition_indicator'].data
activity = results2.features['binary_activity'].data
motion_mag = results2.features['motion_magnitude'].data
periodicity = results2.features['periodicity_strength'].data

# Analyze the relationship between repetition and motion
if repetition.get('has_repetition') and len(activity) > 0:
    print(f"\nRepetition with Motion Analysis:")
    print(f"  Repetition detected: Yes")
    print(f"  Activity rate: {activity.mean():.1%}")
    print(f"  Average motion magnitude: {motion_mag.mean():.3f}")
    
    # Check if repetition correlates with activity bursts
    if activity.mean() > 0.3:
        print(f"  Pattern type: Likely repetitive action or animation")
    else:
        print(f"  Pattern type: Likely static periodic element")

# Analyze periodicity strength details
if isinstance(periodicity, dict):
    print(f"\nDetailed Periodicity Analysis:")
    
    if 'dominant_frequencies' in periodicity:
        freqs = periodicity['dominant_frequencies']
        print(f"  Found {len(freqs)} dominant frequencies")
        if isinstance(freqs, list) and len(freqs) > 0:
            # Handle different data formats
            for i, freq in enumerate(freqs[:3]):  # Show top 3
                if isinstance(freq, dict):
                    print(f"    Frequency {i+1}: {freq.get('frequency', 0):.3f} Hz")
                    print(f"      Strength: {freq.get('strength', 0):.3f}")
                else:
                    print(f"    Frequency {i+1}: {freq:.3f}")
    
    if 'periodicity_score' in periodicity:
        score = periodicity['periodicity_score']
        print(f"\n  Overall periodicity score: {score:.2%}")
        
        if score > 0.7:
            print("  Assessment: Strong periodic behavior")
        elif score > 0.3:
            print("  Assessment: Moderate periodic behavior")
        else:
            print("  Assessment: Weak or no periodic behavior")

# Look for specific types of repetition
print("\n" + "="*50)
print("Repetition Type Classification:")

if repetition.get('has_repetition'):
    # Estimate repetition characteristics
    if repetition.get('num_peaks', 0) == 1:
        print("  Type: Single dominant frequency (simple loop)")
    elif repetition.get('num_peaks', 0) > 1:
        print("  Type: Multiple frequencies (complex pattern)")
    
    score = repetition.get('repetition_score', 0)
    if score > 0.8:
        print("  Strength: Very strong repetition")
    elif score > 0.5:
        print("  Strength: Clear repetition")
    else:
        print("  Strength: Weak repetition")
else:
    print("  No repetitive patterns detected")
    print("  Video content appears non-periodic")

print("\n✓ Repetition analysis complete")