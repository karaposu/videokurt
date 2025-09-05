"""Structural similarity example - measures visual structure similarity between frames."""

# to run python example_structural_similarity.py
from videokurt import VideoKurt
import numpy as np

# Create VideoKurt instance
vk = VideoKurt()

# Configure for structural similarity analysis
vk.configure(frame_step=3, resolution_scale=0.5)

# Add structural_similarity feature
vk.add_feature('structural_similarity',
               win_size=7,  # SSIM window size
               multichannel=False)  # Grayscale comparison

print("Computing structural similarity between frames...")
print("This measures how similar the visual structure is across time")
print()

results = vk.analyze('sample_recording.MP4')

# Get structural similarity results
ssim = results.features['structural_similarity'].data

print("\nStructural Similarity Results:")
print(f"  Data type: {type(ssim)}")

if isinstance(ssim, dict):
    # SSIM scores
    if 'ssim_scores' in ssim:
        scores = ssim['ssim_scores']
        if isinstance(scores, (list, np.ndarray)) and len(scores) > 0:
            print(f"\n  SSIM Statistics:")
            print(f"    Number of comparisons: {len(scores)}")
            print(f"    Average SSIM: {np.mean(scores):.3f}")
            print(f"    Min SSIM: {min(scores):.3f}")
            print(f"    Max SSIM: {max(scores):.3f}")
            print(f"    Std deviation: {np.std(scores):.3f}")
            
            # Categorize similarity levels
            very_similar = sum(1 for s in scores if s > 0.9)
            similar = sum(1 for s in scores if 0.7 < s <= 0.9)
            different = sum(1 for s in scores if 0.5 < s <= 0.7)
            very_different = sum(1 for s in scores if s <= 0.5)
            
            print(f"\n  Similarity Distribution:")
            print(f"    Very similar (>0.9): {very_similar} ({100*very_similar/len(scores):.1f}%)")
            print(f"    Similar (0.7-0.9): {similar} ({100*similar/len(scores):.1f}%)")
            print(f"    Different (0.5-0.7): {different} ({100*different/len(scores):.1f}%)")
            print(f"    Very different (≤0.5): {very_different} ({100*very_different/len(scores):.1f}%)")
    
    # Component-wise similarity
    if 'component_scores' in ssim:
        components = ssim['component_scores']
        if components:
            print(f"\n  Component Analysis:")
            if 'luminance' in components:
                print(f"    Luminance similarity: {np.mean(components['luminance']):.3f}")
            if 'contrast' in components:
                print(f"    Contrast similarity: {np.mean(components['contrast']):.3f}")
            if 'structure' in components:
                print(f"    Structure similarity: {np.mean(components['structure']):.3f}")
    
    # Change points
    if 'change_points' in ssim:
        changes = ssim['change_points']
        if changes:
            print(f"\n  Structural Change Points:")
            print(f"    Total changes detected: {len(changes)}")
            
            for i, change in enumerate(changes[:5]):
                if isinstance(change, dict):
                    print(f"\n    Change {i+1}:")
                    print(f"      Frame: {change.get('frame', 0)}")
                    print(f"      SSIM drop: {change.get('ssim_drop', 0):.3f}")
                    print(f"      Type: {change.get('change_type', 'unknown')}")
                else:
                    # Simple frame index
                    print(f"\n    Change at frame: {change}")
    
    # Stability regions
    if 'stable_regions' in ssim:
        regions = ssim['stable_regions']
        if regions:
            print(f"\n  Stable Regions: {len(regions)} found")
            
            for i, region in enumerate(regions[:3]):
                print(f"\n    Region {i+1}:")
                print(f"      Start: Frame {region.get('start', 0)}")
                print(f"      End: Frame {region.get('end', 0)}")
                duration = region.get('end', 0) - region.get('start', 0)
                print(f"      Duration: {duration} frames")
                print(f"      Avg SSIM: {region.get('avg_ssim', 0):.3f}")

# Analyze structural patterns
print("\n" + "="*50)
print("Structural Pattern Analysis:")

if isinstance(ssim, dict) and 'ssim_scores' in ssim:
    scores = ssim['ssim_scores']
    
    if len(scores) > 0:
        avg_ssim = np.mean(scores)
        
        if avg_ssim > 0.95:
            print("  Pattern: Highly stable structure")
            print("  Interpretation: Minimal structural changes, possibly static UI")
        elif avg_ssim > 0.85:
            print("  Pattern: Mostly stable structure")
            print("  Interpretation: Minor changes, consistent layout")
        elif avg_ssim > 0.70:
            print("  Pattern: Moderate structural changes")
            print("  Interpretation: Regular updates with consistent elements")
        elif avg_ssim > 0.50:
            print("  Pattern: Significant structural variation")
            print("  Interpretation: Dynamic content with changing layouts")
        else:
            print("  Pattern: Highly variable structure")
            print("  Interpretation: Rapid changes or scene transitions")
        
        # Check for gradual vs sudden changes
        if len(scores) > 10:
            diffs = np.diff(scores)
            sudden_changes = sum(1 for d in diffs if abs(d) > 0.3)
            gradual_changes = sum(1 for d in diffs if 0.05 < abs(d) <= 0.3)
            
            print(f"\n  Change Characteristics:")
            print(f"    Sudden changes: {sudden_changes}")
            print(f"    Gradual changes: {gradual_changes}")
            
            if sudden_changes > gradual_changes:
                print("    Pattern: Primarily sudden transitions")
            else:
                print("    Pattern: Primarily gradual transitions")

# Multi-scale SSIM analysis
print("\n" + "="*50)
print("Multi-scale Structural Analysis:")

vk2 = VideoKurt()
vk2.configure(frame_step=3, resolution_scale=0.5)

# Add SSIM with different window size
vk2.add_feature('structural_similarity',
               win_size=11)  # Larger window for coarser structure

print("\nProcessing with larger SSIM window...")
results2 = vk2.analyze('sample_recording.MP4')

ssim_coarse = results2.features['structural_similarity'].data

if 'ssim_scores' in ssim and 'ssim_scores' in ssim_coarse:
    fine_scores = ssim['ssim_scores']
    coarse_scores = ssim_coarse['ssim_scores']
    
    if len(fine_scores) > 0 and len(coarse_scores) > 0:
        min_len = min(len(fine_scores), len(coarse_scores))
        
        print(f"\n  Scale Comparison:")
        print(f"    Fine structure SSIM: {np.mean(fine_scores[:min_len]):.3f}")
        print(f"    Coarse structure SSIM: {np.mean(coarse_scores[:min_len]):.3f}")
        
        # Compare scale sensitivity
        fine_var = np.var(fine_scores[:min_len])
        coarse_var = np.var(coarse_scores[:min_len])
        
        if fine_var > coarse_var * 1.5:
            print("    Fine details changing more than overall structure")
        elif coarse_var > fine_var * 1.5:
            print("    Overall structure changing more than fine details")
        else:
            print("    Changes occur at multiple scales equally")

# Content type detection
print("\n" + "="*50)
print("Content Type Detection via Structure:")

if isinstance(ssim, dict):
    avg_ssim = np.mean(ssim.get('ssim_scores', [0]))
    change_points = len(ssim.get('change_points', []))
    
    # Heuristic content detection
    if avg_ssim > 0.95 and change_points < 5:
        print("  Likely content: Static document or presentation")
    elif avg_ssim > 0.85 and change_points < 20:
        print("  Likely content: UI application with minimal animation")
    elif avg_ssim > 0.70:
        if change_points > 50:
            print("  Likely content: Dynamic UI with frequent updates")
        else:
            print("  Likely content: Slowly changing content or slideshow")
    elif avg_ssim > 0.50:
        print("  Likely content: Video playback or animation")
    else:
        print("  Likely content: Rapid scene changes or transitions")
    
    # Check for UI patterns
    if 'stable_regions' in ssim:
        regions = ssim['stable_regions']
        if regions:
            avg_region_length = np.mean([r.get('end', 0) - r.get('start', 0) for r in regions])
            
            if avg_region_length > 30:
                print("\n  UI Pattern: Long stable periods")
                print("  Suggests: Page/screen dwelling")
            elif avg_region_length > 10:
                print("\n  UI Pattern: Moderate stability")
                print("  Suggests: Regular interaction")
            else:
                print("\n  UI Pattern: Brief stable periods")
                print("  Suggests: Continuous interaction or scrolling")

print("\n✓ Structural similarity analysis complete")