"""Edge density feature example - measures visual complexity and structure."""

# to run python example_edge_density.py
from videokurt import VideoKurt
import numpy as np

# Create VideoKurt instance
vk = VideoKurt()

# Configure for reasonable speed
vk.configure(frame_step=5, resolution_scale=0.5)

# Add edge_density feature (auto-configures edge_canny analysis)
vk.add_feature('edge_density', use_gradient=False)  # Use binary edges

print("Processing video for edge density analysis...")
results = vk.analyze('sample_recording.MP4')

# Get edge density data
edge_density = results.features['edge_density'].data

print(f"\nEdge Density Statistics:")
print(f"  Shape: {edge_density.shape}")
print(f"  Range: {edge_density.min():.3f} to {edge_density.max():.3f}")
print(f"  Mean: {edge_density.mean():.3f}")
print(f"  Std: {edge_density.std():.3f}")

# Classify frames by edge density
low_complexity = edge_density < np.percentile(edge_density, 25)
medium_complexity = (edge_density >= np.percentile(edge_density, 25)) & (edge_density < np.percentile(edge_density, 75))
high_complexity = edge_density >= np.percentile(edge_density, 75)

print(f"\nVisual Complexity Distribution:")
print(f"  Low complexity (minimal edges): {low_complexity.sum()} frames")
print(f"  Medium complexity: {medium_complexity.sum()} frames")
print(f"  High complexity (lots of edges): {high_complexity.sum()} frames")

# Find sudden changes in visual complexity (potential scene changes)
edge_diff = np.abs(np.diff(edge_density))
sudden_changes = np.where(edge_diff > edge_diff.std() * 2)[0]
print(f"\nSudden complexity changes: {len(sudden_changes)} locations")
if len(sudden_changes) > 0:
    print(f"  First 5 change points: {sudden_changes[:5]}")

# Combine with other features for richer analysis
print("\n" + "="*50)
print("Combined analysis with multiple features:")

vk2 = VideoKurt()
vk2.configure(frame_step=5, resolution_scale=0.5)

# Add multiple complementary features
vk2.add_feature('edge_density')
vk2.add_feature('binary_activity')
vk2.add_feature('texture_uniformity')  # Requires texture_descriptors

print("\nProcessing with edge + activity + texture features...")
results2 = vk2.analyze('sample_recording.MP4')

edges = results2.features['edge_density'].data
activity = results2.features['binary_activity'].data
texture = results2.features['texture_uniformity'].data

# Identify different content types
# Handle potential shape mismatches (frame_diff produces N-1 frames)
min_len = min(len(edges), len(activity), len(texture))
edges_aligned = edges[:min_len]
activity_aligned = activity[:min_len]
texture_aligned = texture[:min_len]

text_heavy = (edges_aligned > np.percentile(edges_aligned, 70)) & (activity_aligned == 0)
smooth_motion = (edges_aligned < np.percentile(edges_aligned, 30)) & (activity_aligned == 1)
detailed_static = (edges_aligned > np.percentile(edges_aligned, 50)) & (activity_aligned == 0) & (texture_aligned < 0.5)

print(f"\nContent Type Analysis:")
print(f"  Text-heavy/UI frames: {text_heavy.sum()}")
print(f"  Smooth motion areas: {smooth_motion.sum()}")
print(f"  Detailed static content: {detailed_static.sum()}")

# Find the most visually complex moment
max_complexity_idx = np.argmax(edge_density)
print(f"\nMost complex frame: #{max_complexity_idx}")
print(f"  Edge density: {edge_density[max_complexity_idx]:.3f}")
if max_complexity_idx < len(activity):
    print(f"  Has motion: {'Yes' if activity[max_complexity_idx] else 'No'}")
    print(f"  Texture uniformity: {texture[max_complexity_idx]:.3f}")