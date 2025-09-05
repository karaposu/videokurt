"""Blob stability example - tracks persistence of moving objects."""

# to run python example_blob_stability.py
from videokurt import VideoKurt
import numpy as np

# Create VideoKurt instance
vk = VideoKurt()

# Configure for blob stability analysis
vk.configure(frame_step=2, resolution_scale=0.6)

# Add blob_stability feature (requires background_mog2)
vk.add_feature('blob_stability', 
               min_persistence=5,  # Minimum 5 frames to be considered stable
               min_area=100)  # Minimum blob area to track

print("Analyzing blob stability and persistence...")
print("This tracks how long objects remain visible and trackable")
print()

results = vk.analyze('sample_recording.MP4')

# Get blob stability results
stability = results.features['blob_stability'].data

print("\nBlob Stability Analysis Results:")
print(f"  Data type: {type(stability)}")

if isinstance(stability, dict):
    # Persistence scores
    if 'persistence_scores' in stability:
        scores = stability['persistence_scores']
        if isinstance(scores, (list, np.ndarray)) and len(scores) > 0:
            print(f"\n  Persistence Statistics:")
            print(f"    Number of tracked blobs: {len(scores)}")
            print(f"    Average persistence: {np.mean(scores):.1f} frames")
            print(f"    Max persistence: {max(scores):.0f} frames")
            print(f"    Min persistence: {min(scores):.0f} frames")
            
            # Classify by persistence
            short_lived = sum(1 for s in scores if s < 5)
            medium_lived = sum(1 for s in scores if 5 <= s < 20)
            long_lived = sum(1 for s in scores if s >= 20)
            
            print(f"\n  Blob Lifespan Distribution:")
            print(f"    Short-lived (<5 frames): {short_lived}")
            print(f"    Medium-lived (5-20 frames): {medium_lived}")
            print(f"    Long-lived (>20 frames): {long_lived}")
    
    # Stable blobs
    if 'stable_blobs' in stability:
        stable = stability['stable_blobs']
        print(f"\n  Stable Blobs Found: {len(stable)}")
        
        # Show details of first few stable blobs
        for i, blob_info in enumerate(stable[:5]):
            print(f"\n  Stable Blob {i+1}:")
            if 'start_frame' in blob_info:
                print(f"    Start frame: {blob_info['start_frame']}")
            if 'end_frame' in blob_info:
                print(f"    End frame: {blob_info['end_frame']}")
            if 'persistence' in blob_info:
                print(f"    Persistence: {blob_info['persistence']} frames")
            if 'avg_size' in blob_info:
                print(f"    Average size: {blob_info['avg_size']:.0f} pixels")
            if 'stability_score' in blob_info:
                print(f"    Stability score: {blob_info['stability_score']:.2f}")
    
    # Stability timeline
    if 'stability_timeline' in stability:
        timeline = stability['stability_timeline']
        if len(timeline) > 0:
            print(f"\n  Stability Over Time:")
            print(f"    Average stability: {np.mean(timeline):.2f}")
            print(f"    Stability variance: {np.var(timeline):.3f}")
            
            # Find periods of high stability
            high_stability_frames = sum(1 for s in timeline if s > 0.7)
            print(f"    High stability frames: {high_stability_frames} ({100*high_stability_frames/len(timeline):.1f}%)")
    
    # Gap analysis
    if 'gap_statistics' in stability:
        gaps = stability['gap_statistics']
        if gaps:
            print(f"\n  Gap Statistics:")
            if 'total_gaps' in gaps:
                print(f"    Total gaps detected: {gaps['total_gaps']}")
            if 'avg_gap_length' in gaps:
                print(f"    Average gap length: {gaps['avg_gap_length']:.1f} frames")
            if 'recovered_tracks' in gaps:
                print(f"    Recovered tracks after gaps: {gaps['recovered_tracks']}")

# Analyze stability patterns
print("\n" + "="*50)
print("Stability Pattern Analysis:")

if isinstance(stability, dict):
    if 'persistence_scores' in stability and len(stability.get('persistence_scores', [])) > 0:
        scores = stability['persistence_scores']
        avg_persistence = np.mean(scores)
        
        if avg_persistence < 5:
            print("  Pattern: Highly unstable")
            print("  Interpretation: Objects appear briefly and disappear")
        elif avg_persistence < 15:
            print("  Pattern: Moderately stable")
            print("  Interpretation: Objects tracked for short periods")
        else:
            print("  Pattern: Highly stable")
            print("  Interpretation: Objects persist for extended periods")
        
        # Check for consistency
        if 'stable_blobs' in stability:
            stable_count = len(stability['stable_blobs'])
            total_count = len(scores)
            stable_ratio = stable_count / total_count if total_count > 0 else 0
            
            print(f"\n  Stable blob ratio: {stable_ratio:.1%}")
            if stable_ratio > 0.5:
                print("  Most tracked objects are stable")
            else:
                print("  Most tracked objects are transient")

# Combine with tracking quality analysis
print("\n" + "="*50)
print("Combined tracking quality analysis:")

vk2 = VideoKurt()
vk2.configure(frame_step=2, resolution_scale=0.6)

# Add tracking quality features
vk2.add_feature('blob_stability')
vk2.add_feature('blob_tracking')

print("\nProcessing with combined tracking features...")
results2 = vk2.analyze('sample_recording.MP4')

stability = results2.features['blob_stability'].data
tracking = results2.features['blob_tracking'].data

if 'stable_blobs' in stability and 'trajectories' in tracking:
    stable_count = len(stability['stable_blobs'])
    total_trajectories = len(tracking['trajectories'])
    
    print(f"\nTracking Quality Summary:")
    print(f"  Total trajectories: {total_trajectories}")
    print(f"  Stable trajectories: {stable_count}")
    print(f"  Stability rate: {stable_count/total_trajectories:.1%}" if total_trajectories > 0 else "  No trajectories")
    
    # Analyze trajectory smoothness
    if total_trajectories > 0:
        avg_traj_length = np.mean([len(t) for t in tracking['trajectories']])
        print(f"  Average trajectory length: {avg_traj_length:.1f} frames")
        
        if avg_traj_length < 10:
            print("  Tracking quality: Poor (short trajectories)")
        elif avg_traj_length < 30:
            print("  Tracking quality: Moderate")
        else:
            print("  Tracking quality: Good (long trajectories)")

print("\n✓ Blob stability analysis complete")