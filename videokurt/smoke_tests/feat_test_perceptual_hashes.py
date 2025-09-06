"""Perceptual hashes example - generates hashes for visual similarity comparison."""

# to run python videokurt/smoke_tests/feat_test_perceptual_hashes.py
from videokurt import VideoKurt
import numpy as np

# Create VideoKurt instance
vk = VideoKurt()

# Configure for perceptual hash generation
vk.configure(frame_step=5, resolution_scale=0.5)

# Add perceptual_hashes feature
vk.add_feature('perceptual_hashes',
               hamming_threshold=10)  # Threshold for similarity

print("Generating perceptual hashes for video frames...")
print("This creates compact signatures for visual similarity comparison")
print()

results = vk.analyze('sample_recording.MP4')

# Get perceptual hash results
hashes = results.features['perceptual_hashes'].data

print("\nPerceptual Hash Results:")
print(f"  Data type: {type(hashes)}")

if isinstance(hashes, dict):
    # Hash values
    if 'hashes' in hashes:
        hash_list = hashes['hashes']
        print(f"\n  Hash Statistics:")
        print(f"    Total frames hashed: {len(hash_list)}")
        
        if hash_list and len(hash_list) > 0:
            # Show first few hashes
            print(f"\n  Sample hashes (first 5):")
            for i, hash_val in enumerate(hash_list[:5]):
                if isinstance(hash_val, str):
                    print(f"    Frame {i}: {hash_val[:16]}...")  # Show first 16 chars
                else:
                    print(f"    Frame {i}: {hash_val}")
            
            # Find unique hashes
            unique_hashes = len(set(hash_list))
            print(f"\n  Unique hashes: {unique_hashes}/{len(hash_list)}")
            print(f"  Uniqueness ratio: {unique_hashes/len(hash_list):.1%}")
    
    # Similarity scores between consecutive frames
    if 'similarities' in hashes:
        similarities = hashes['similarities']
        if isinstance(similarities, (list, np.ndarray)) and len(similarities) > 0:
            print(f"\n  Frame-to-Frame Similarity:")
            print(f"    Average similarity: {np.mean(similarities):.3f}")
            print(f"    Min similarity: {min(similarities):.3f}")
            print(f"    Max similarity: {max(similarities):.3f}")
            
            # Detect significant changes
            low_similarity = sum(1 for s in similarities if s < 0.7)
            print(f"    Significant changes: {low_similarity} transitions")
    
    # Duplicate detection
    if 'duplicates' in hashes:
        duplicates = hashes['duplicates']
        print(f"\n  Duplicate Detection:")
        print(f"    Duplicate groups found: {len(duplicates)}")
        
        if duplicates:
            for i, dup_group in enumerate(duplicates[:3]):
                print(f"\n    Group {i+1}:")
                print(f"      Frames: {dup_group['frames'][:5]}...")
                print(f"      Group size: {len(dup_group['frames'])}")
                if 'hash' in dup_group:
                    print(f"      Hash: {dup_group['hash'][:16]}...")
    
    # Clustering analysis
    if 'clusters' in hashes:
        clusters = hashes['clusters']
        print(f"\n  Visual Clustering:")
        print(f"    Number of clusters: {len(clusters)}")
        
        for i, cluster in enumerate(clusters[:3]):
            print(f"\n    Cluster {i+1}:")
            print(f"      Size: {cluster.get('size', 0)} frames")
            if 'representative_frame' in cluster:
                print(f"      Representative: Frame {cluster['representative_frame']}")
            if 'avg_similarity' in cluster:
                print(f"      Internal similarity: {cluster['avg_similarity']:.3f}")

# Analyze hash patterns
print("\n" + "="*50)
print("Visual Pattern Analysis:")

if isinstance(hashes, dict):
    if 'similarities' in hashes:
        similarities = hashes['similarities']
        
        if len(similarities) > 0:
            avg_sim = np.mean(similarities)
            
            if avg_sim > 0.95:
                print("  Pattern: Nearly static content")
                print("  Interpretation: Minimal visual changes")
            elif avg_sim > 0.85:
                print("  Pattern: Slowly changing content")
                print("  Interpretation: Gradual visual transitions")
            elif avg_sim > 0.70:
                print("  Pattern: Moderate visual changes")
                print("  Interpretation: Regular content updates")
            else:
                print("  Pattern: Rapid visual changes")
                print("  Interpretation: Dynamic or fast-changing content")
            
            # Check for periodic patterns
            if len(similarities) > 20:
                # Simple periodicity check
                diffs = np.diff(similarities)
                zero_crossings = np.where(np.diff(np.sign(diffs)))[0]
                
                if len(zero_crossings) > 5:
                    periods = np.diff(zero_crossings)
                    if np.std(periods) < np.mean(periods) * 0.3:
                        print(f"\n  Periodic pattern detected")
                        print(f"  Approximate period: {np.mean(periods):.1f} frames")

# Combine with scene boundary detection
print("\n" + "="*50)
print("Scene Boundary Detection via Hashes:")

vk2 = VideoKurt()
vk2.configure(frame_step=2, resolution_scale=0.5)

# Add perceptual hashes with different settings
vk2.add_feature('perceptual_hashes',
               hamming_threshold=5)  # Stricter threshold

print("\nProcessing with higher resolution hashes...")
results2 = vk2.analyze('sample_recording.MP4')

hashes_hr = results2.features['perceptual_hashes'].data

if 'similarities' in hashes and 'similarities' in hashes_hr:
    sim_low = hashes['similarities']
    sim_high = hashes_hr['similarities']
    
    # Detect scene boundaries (low similarity)
    scene_boundaries_low = [i for i, s in enumerate(sim_low) if s < 0.7]
    scene_boundaries_high = [i for i, s in enumerate(sim_high) if s < 0.7]
    
    print(f"\n  Scene Boundary Comparison:")
    print(f"    Low-res hash boundaries: {len(scene_boundaries_low)}")
    print(f"    High-res hash boundaries: {len(scene_boundaries_high)}")
    
    if scene_boundaries_low:
        print(f"    First boundaries (low-res): {scene_boundaries_low[:5]}")
    if scene_boundaries_high:
        print(f"    First boundaries (high-res): {scene_boundaries_high[:5]}")

# Content fingerprinting
print("\n" + "="*50)
print("Content Fingerprinting:")

if isinstance(hashes, dict) and 'hashes' in hashes:
    hash_list = hashes['hashes']
    
    if hash_list:
        # Create a simple fingerprint
        unique_sequence = []
        prev_hash = None
        
        for h in hash_list:
            if h != prev_hash:
                unique_sequence.append(h)
                prev_hash = h
        
        print(f"\n  Video Fingerprint:")
        print(f"    Original frames: {len(hash_list)}")
        print(f"    Unique transitions: {len(unique_sequence)}")
        print(f"    Compression ratio: {len(unique_sequence)/len(hash_list):.1%}")
        
        if len(unique_sequence) < len(hash_list) * 0.1:
            print("    Highly repetitive content")
        elif len(unique_sequence) < len(hash_list) * 0.5:
            print("    Moderately repetitive content")
        else:
            print("    Highly varied content")
        
        # Check for loops
        if len(unique_sequence) > 10:
            # Simple loop detection
            for window_size in [3, 5, 10]:
                if window_size < len(unique_sequence) // 2:
                    window = unique_sequence[:window_size]
                    matches = 0
                    for i in range(window_size, len(unique_sequence) - window_size):
                        if unique_sequence[i:i+window_size] == window:
                            matches += 1
                    
                    if matches > 0:
                        print(f"    Potential {window_size}-frame loop: {matches} repetitions")

print("\n✓ Perceptual hash analysis complete")