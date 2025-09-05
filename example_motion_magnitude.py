"""Example showing MotionMagnitude feature - measures intensity of motion per frame."""

# to run python example_motion_magnitude.py
from videokurt import VideoKurt
import numpy as np

# Create VideoKurt instance
vk = VideoKurt()

# Add motion_magnitude feature (auto-configures optical_flow_dense)
vk.add_feature('motion_magnitude', normalize=True)  # Normalize by frame size

# Process video
results = vk.analyze('sample_recording.MP4')

# Get motion magnitude data
motion_mag = results.features['motion_magnitude'].data
print(f"Motion magnitude shape: {motion_mag.shape}")
print(f"Motion range: {motion_mag.min():.2f} to {motion_mag.max():.2f}")
print(f"Average motion: {motion_mag.mean():.2f}")

# Find high motion periods (above average)
high_motion_threshold = motion_mag.mean() + motion_mag.std()
high_motion_frames = np.where(motion_mag > high_motion_threshold)[0]
print(f"\nHigh motion detected in {len(high_motion_frames)} frames")

# Find the most active segment
window_size = 30  # 30-frame windows
if len(motion_mag) >= window_size:
    windowed_activity = np.convolve(motion_mag, np.ones(window_size)/window_size, mode='valid')
    most_active_start = np.argmax(windowed_activity)
    print(f"Most active period: frames {most_active_start} to {most_active_start + window_size}")
    print(f"Peak activity level: {windowed_activity[most_active_start]:.2f}")

# Compare with binary activity for richer analysis
vk2 = VideoKurt()
vk2.add_feature('binary_activity')
vk2.add_feature('motion_magnitude')

results2 = vk2.analyze('sample_recording.MP4')

binary = results2.features['binary_activity'].data
magnitude = results2.features['motion_magnitude'].data

# Find frames that are active but with low magnitude (small movements)
active_but_gentle = (binary == 1) & (magnitude < magnitude.mean())
print(f"\nFrames with gentle motion: {active_but_gentle.sum()}")

# Find frames with high magnitude (vigorous motion)
vigorous = magnitude > np.percentile(magnitude, 90)
print(f"Frames with vigorous motion: {vigorous.sum()}")