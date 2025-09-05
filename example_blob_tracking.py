"""Blob tracking example - tracks moving objects/regions in video."""

# to run python example_blob_tracking.py
from videokurt import VideoKurt
import numpy as np

# Create VideoKurt instance
vk = VideoKurt()

# Configure for blob tracking
vk.configure(frame_step=2, resolution_scale=0.6)

# Add blob_tracking feature (requires background_mog2)
vk.add_feature('blob_tracking', 
               min_area=100,  # Minimum blob size in pixels
               max_area=10000)  # Maximum blob size

print("Tracking blobs (moving objects/regions) in video...")
print("This identifies and tracks distinct moving regions")
print()

results = vk.analyze('sample_recording.MP4')

# Get blob tracking results
blobs = results.features['blob_tracking'].data

print("\nBlob Tracking Results:")
print(f"  Data type: {type(blobs)}")

if isinstance(blobs, dict):
    # Blob counts per frame
    if 'counts' in blobs:
        counts = blobs['counts']
        print(f"\n  Blob count statistics:")
        print(f"    Total frames analyzed: {len(counts)}")
        print(f"    Average blobs per frame: {np.mean(counts):.1f}")
        print(f"    Max blobs in a frame: {max(counts) if counts else 0}")
        
        # Frames with blobs
        frames_with_blobs = sum(1 for c in counts if c > 0)
        print(f"    Frames with blobs: {frames_with_blobs} ({100*frames_with_blobs/len(counts):.1f}%)")
    
    # Blob sizes
    if 'sizes' in blobs:
        all_sizes = [s for frame_sizes in blobs['sizes'] for s in frame_sizes]
        if all_sizes:
            print(f"\n  Blob size statistics:")
            print(f"    Total blobs detected: {len(all_sizes)}")
            print(f"    Average size: {np.mean(all_sizes):.0f} pixels")
            print(f"    Size range: {min(all_sizes):.0f} - {max(all_sizes):.0f} pixels")
            
            # Size distribution
            small = sum(1 for s in all_sizes if s < 500)
            medium = sum(1 for s in all_sizes if 500 <= s < 2000)
            large = sum(1 for s in all_sizes if s >= 2000)
            
            print(f"\n  Size distribution:")
            print(f"    Small (<500px): {small} blobs")
            print(f"    Medium (500-2000px): {medium} blobs")
            print(f"    Large (>2000px): {large} blobs")
    
    # Trajectories
    if 'trajectories' in blobs:
        trajectories = blobs['trajectories']
        if trajectories:
            print(f"\n  Trajectory information:")
            print(f"    Number of tracked paths: {len(trajectories)}")
            
            # Analyze trajectory lengths
            traj_lengths = [len(t) for t in trajectories]
            if traj_lengths:
                print(f"    Average trajectory length: {np.mean(traj_lengths):.1f} frames")
                print(f"    Longest trajectory: {max(traj_lengths)} frames")
                
                # Classify trajectories
                short_traj = sum(1 for l in traj_lengths if l < 5)
                medium_traj = sum(1 for l in traj_lengths if 5 <= l < 20)
                long_traj = sum(1 for l in traj_lengths if l >= 20)
                
                print(f"\n  Trajectory persistence:")
                print(f"    Short (<5 frames): {short_traj}")
                print(f"    Medium (5-20 frames): {medium_traj}")
                print(f"    Long (>20 frames): {long_traj}")
    
    # Centroids (positions)
    if 'centroids' in blobs:
        centroids = blobs['centroids']
        # Analyze spatial distribution
        all_x = []
        all_y = []
        for frame_centroids in centroids:
            for centroid in frame_centroids:
                if len(centroid) >= 2:
                    all_x.append(centroid[0])
                    all_y.append(centroid[1])
        
        if all_x and all_y:
            print(f"\n  Spatial distribution:")
            print(f"    X range: {min(all_x):.0f} - {max(all_x):.0f}")
            print(f"    Y range: {min(all_y):.0f} - {max(all_y):.0f}")
            print(f"    Center of activity: ({np.mean(all_x):.0f}, {np.mean(all_y):.0f})")

# Analyze blob patterns
print("\n" + "="*50)
print("Blob Movement Patterns:")

if isinstance(blobs, dict) and 'counts' in blobs:
    counts = blobs['counts']
    
    # Detect blob appearance patterns
    if len(counts) > 10:
        # Check for sudden appearances
        diffs = np.diff(counts)
        sudden_increases = sum(1 for d in diffs if d > 2)
        sudden_decreases = sum(1 for d in diffs if d < -2)
        
        print(f"\n  Blob dynamics:")
        print(f"    Sudden appearances: {sudden_increases}")
        print(f"    Sudden disappearances: {sudden_decreases}")
        
        # Check stability
        if np.std(counts) < 1:
            print("    Pattern: Stable blob count")
        elif np.std(counts) < 3:
            print("    Pattern: Moderate blob variation")
        else:
            print("    Pattern: Highly variable blob count")

# Combine with interaction analysis
print("\n" + "="*50)
print("Combined blob analysis:")

vk2 = VideoKurt()
vk2.configure(frame_step=2, resolution_scale=0.6)

# Add blob tracking with interaction detection
vk2.add_feature('blob_tracking')
vk2.add_feature('blob_stability', min_persistence=5)  # Track blob persistence
vk2.add_feature('interaction_zones')  # Detect blob interactions

print("\nProcessing with blob interaction features...")
results2 = vk2.analyze('sample_recording.MP4')

tracking = results2.features['blob_tracking'].data
stability = results2.features['blob_stability'].data
interactions = results2.features['interaction_zones'].data

print(f"\nBlob Analysis Summary:")
if 'counts' in tracking:
    print(f"  Average blobs: {np.mean(tracking['counts']):.1f}")

if isinstance(stability, dict):
    if 'persistence_scores' in stability:
        scores = stability['persistence_scores']
        if scores:
            print(f"  Blob persistence: {np.mean(scores):.1f} frames average")
    
    if 'stable_blobs' in stability:
        stable = stability['stable_blobs']
        print(f"  Stable blobs found: {len(stable)}")

if isinstance(interactions, dict):
    if 'interaction_events' in interactions:
        events = interactions['interaction_events']
        print(f"  Interaction events: {len(events)}")
    
    if 'avg_overlap' in interactions:
        print(f"  Average overlap: {interactions['avg_overlap']:.2%}")

# Interpret blob patterns
print("\n" + "="*50)
print("Blob Pattern Interpretation:")

if isinstance(blobs, dict) and 'counts' in blobs:
    avg_blobs = np.mean(blobs['counts'])
    
    if avg_blobs == 0:
        print("  No moving objects detected")
    elif avg_blobs < 1:
        print("  Occasional moving objects")
        print("  Interpretation: Mostly static with rare movement")
    elif avg_blobs < 3:
        print("  Few moving objects")
        print("  Interpretation: Limited movement, possibly focused activity")
    elif avg_blobs < 10:
        print("  Multiple moving objects")
        print("  Interpretation: Active scene with several moving elements")
    else:
        print("  Many moving objects")
        print("  Interpretation: Highly dynamic scene with numerous movements")

print("\n✓ Blob tracking analysis complete")