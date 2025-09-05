"""Simple motion analysis example comparing different motion features."""

# to run python example_motion_simple.py
from videokurt import VideoKurt
import numpy as np

# Create VideoKurt instance with faster processing
vk = VideoKurt()

# Configure for faster processing - process every 5th frame, half resolution
vk.configure(frame_step=5, resolution_scale=0.5)

# Add multiple motion-related features
vk.add_feature('binary_activity', threshold=30.0)  # Simple yes/no activity
vk.add_feature('stability_score', window_size=10)  # How stable content is
vk.add_feature('frame_difference_percentile', percentile=95)  # 95th percentile of changes

print("Processing video with 3 motion features...")
print("(Using frame_step=5 and resolution_scale=0.5 for speed)")

results = vk.analyze('sample_recording.MP4')

# Get all motion features
binary = results.features['binary_activity'].data
stability = results.features['stability_score'].data  
percentile_95 = results.features['frame_difference_percentile'].data

print(f"\nProcessed {len(binary)} frames")
print(f"Activity detected: {binary.sum()} frames ({100*binary.mean():.1f}%)")
print(f"Average stability: {stability.mean():.2f} (1=stable, 0=changing)")
print(f"95th percentile range: {percentile_95.min():.1f} to {percentile_95.max():.1f}")

# Find interesting moments
unstable = stability < 0.5
high_change = percentile_95 > np.percentile(percentile_95, 90)

print(f"\nUnstable moments: {unstable.sum()} frames")
print(f"High change moments: {high_change.sum()} frames")

# Identify different types of activity
static = (binary == 0) & (stability > 0.9)
gentle_motion = (binary == 1) & (stability > 0.5)
vigorous_motion = (binary == 1) & (stability < 0.5)

print(f"\nActivity breakdown:")
print(f"  Static: {static.sum()} frames")
print(f"  Gentle motion: {gentle_motion.sum()} frames")  
print(f"  Vigorous motion: {vigorous_motion.sum()} frames")