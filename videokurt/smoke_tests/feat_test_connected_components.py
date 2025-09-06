"""Connected components example - identifies distinct connected regions."""

# to run python videokurt/smoke_tests/feat_test_connected_components.py
from videokurt import VideoKurt
import numpy as np

# Create VideoKurt instance
vk = VideoKurt()

# Configure for connected components analysis
vk.configure(frame_step=3, resolution_scale=0.5)

# Add connected_components feature
vk.add_feature('connected_components',
               min_area=50,  # Minimum component area in pixels
               max_components=100)  # Maximum components to track

print("Analyzing connected components in video...")
print("This identifies distinct connected regions of activity")
print()

results = vk.analyze('sample_recording.MP4')

# Get connected components results (if available)
if 'connected_components' not in results.features:
    print("\nError: Connected components feature failed to compute")
    print("This feature requires background subtraction which may not work well with screen recordings")
    exit(1)
    
components = results.features['connected_components'].data

print("\nConnected Components Results:")
print(f"  Data type: {type(components)}")

if isinstance(components, dict):
    # Component counts
    if 'num_components_timeline' in components:
        counts = components['num_components_timeline']
        print(f"\n  Component Statistics:")
        print(f"    Total frames analyzed: {len(counts)}")
        print(f"    Average components per frame: {np.mean(counts):.1f}")
        print(f"    Max components in a frame: {max(counts) if len(counts) > 0 else 0}")
        print(f"    Min components in a frame: {min(counts) if len(counts) > 0 else 0}")
        
        # Frames with components
        frames_with_components = sum(1 for c in counts if c > 0)
        if len(counts) > 0:
            print(f"    Frames with components: {frames_with_components} ({100*frames_with_components/len(counts):.1f}%)")
    
    # Component timeline data
    if 'component_timeline' in components:
        timeline = components['component_timeline']
        
        # Extract all sizes
        all_sizes = []
        for frame_data in timeline:
            for comp in frame_data.get('components', []):
                if 'area' in comp:
                    all_sizes.append(comp['area'])
        
        if all_sizes:
            print(f"\n  Size Distribution:")
            print(f"    Total components: {len(all_sizes)}")
            print(f"    Average size: {np.mean(all_sizes):.0f} pixels")
            print(f"    Size range: {min(all_sizes):.0f} - {max(all_sizes):.0f} pixels")
            
            # Size categories
            tiny = sum(1 for s in all_sizes if s < 100)
            small = sum(1 for s in all_sizes if 100 <= s < 500)
            medium = sum(1 for s in all_sizes if 500 <= s < 2000)
            large = sum(1 for s in all_sizes if s >= 2000)
            
            print(f"\n  Component Categories:")
            print(f"    Tiny (<100px): {tiny}")
            print(f"    Small (100-500px): {small}")
            print(f"    Medium (500-2000px): {medium}")
            print(f"    Large (>2000px): {large}")
    
    # Component properties from timeline
    if 'component_timeline' in components and len(components['component_timeline']) > 0:
        # Show properties from first frame with components
        for frame_data in components['component_timeline']:
            if frame_data['num_components'] > 0:
                print(f"\n  Component Properties (Frame {frame_data['frame']}):")
                for i, comp in enumerate(frame_data['components'][:3]):
                    print(f"\n    Component {i+1}:")
                    if 'area' in comp:
                        print(f"      Area: {comp['area']} pixels")
                    if 'centroid' in comp:
                        print(f"      Centroid: {comp['centroid']}")
                    if 'bbox' in comp:
                        x, y, w, h = comp['bbox']
                        print(f"      Bounding box: ({x},{y}) {w}x{h}")
                    if 'aspect_ratio' in comp:
                        print(f"      Aspect ratio: {comp['aspect_ratio']:.2f}")
                break
    
    # Connectivity statistics
    if 'connectivity_stats' in components:
        conn_stats = components['connectivity_stats']
        print(f"\n  Connectivity Analysis:")
        
        if 'avg_connectivity' in conn_stats:
            print(f"    Average connectivity: {conn_stats['avg_connectivity']:.2f}")
        
        if 'fragmentation_score' in conn_stats:
            score = conn_stats['fragmentation_score']
            print(f"    Fragmentation score: {score:.2f}")
            
            if score < 0.3:
                print("    Pattern: Low fragmentation (large connected regions)")
            elif score < 0.7:
                print("    Pattern: Moderate fragmentation")
            else:
                print("    Pattern: High fragmentation (many small regions)")

# Analyze temporal evolution
print("\n" + "="*50)
print("Temporal Component Evolution:")

if isinstance(components, dict) and 'num_components_timeline' in components:
    counts = components['num_components_timeline']
    
    if len(counts) > 10:
        # Check for trends
        first_half = np.mean(counts[:len(counts)//2])
        second_half = np.mean(counts[len(counts)//2:])
        
        print(f"\n  Temporal Trends:")
        print(f"    First half average: {first_half:.1f} components")
        print(f"    Second half average: {second_half:.1f} components")
        
        if abs(first_half - second_half) < 1:
            print("    Pattern: Stable component count")
        elif first_half > second_half:
            print("    Pattern: Decreasing components (consolidation)")
        else:
            print("    Pattern: Increasing components (fragmentation)")
        
        # Check variability
        std_count = np.std(counts)
        cv = std_count / np.mean(counts) if np.mean(counts) > 0 else 0
        
        if cv < 0.2:
            print(f"    Variability: Low (CV={cv:.2f})")
        elif cv < 0.5:
            print(f"    Variability: Moderate (CV={cv:.2f})")
        else:
            print(f"    Variability: High (CV={cv:.2f})")

# Combine with morphological analysis
print("\n" + "="*50)
print("Morphological Analysis:")

vk2 = VideoKurt()
vk2.configure(frame_step=3, resolution_scale=0.5)

# Add connected components with different thresholds
vk2.add_feature('connected_components', min_size=20)  # Lower threshold

print("\nProcessing with lower size threshold...")
results2 = vk2.analyze('sample_recording.MP4')

components_low = results2.features['connected_components'].data

if 'num_components_timeline' in components and 'num_components_timeline' in components_low:
    regular_count = np.mean(components['num_components_timeline'])
    low_threshold_count = np.mean(components_low['num_components_timeline'])
    
    print(f"\n  Threshold Comparison:")
    print(f"    Regular threshold: {regular_count:.1f} components/frame")
    print(f"    Low threshold: {low_threshold_count:.1f} components/frame")
    
    noise_ratio = (low_threshold_count - regular_count) / low_threshold_count if low_threshold_count > 0 else 0
    print(f"    Noise components: {noise_ratio:.1%}")
    
    if noise_ratio > 0.5:
        print("    High noise level detected")
    elif noise_ratio > 0.2:
        print("    Moderate noise level")
    else:
        print("    Low noise level")

# Interpret component patterns
print("\n" + "="*50)
print("Component Pattern Interpretation:")

if isinstance(components, dict):
    if 'num_components_timeline' in components:
        avg_components = np.mean(components['num_components_timeline'])
        
        if avg_components == 0:
            print("  No distinct components detected")
            print("  Interpretation: Uniform or static content")
        elif avg_components < 2:
            print("  Pattern: Single dominant component")
            print("  Interpretation: One main active region")
        elif avg_components < 5:
            print("  Pattern: Few distinct components")
            print("  Interpretation: Several separate active areas")
        elif avg_components < 10:
            print("  Pattern: Multiple components")
            print("  Interpretation: Complex scene with many distinct regions")
        else:
            print("  Pattern: Many components")
            print("  Interpretation: Highly fragmented or noisy activity")
    
    # Size distribution interpretation
    if 'component_timeline' in components:
        all_sizes = []
        for frame_data in components['component_timeline']:
            for comp in frame_data.get('components', []):
                if 'area' in comp:
                    all_sizes.append(comp['area'])
        
        if all_sizes:
            size_cv = np.std(all_sizes) / np.mean(all_sizes) if np.mean(all_sizes) > 0 else 0
            
            if size_cv < 0.3:
                print("\n  Component sizes are uniform")
            elif size_cv < 0.7:
                print("\n  Component sizes vary moderately")
            else:
                print("\n  Component sizes vary significantly")

print("\n✓ Connected components analysis complete")