"""Dwell time maps example - shows how long activity persists in different regions."""

# to run python videokurt/smoke_tests/feat_test_dwell_time.py
from videokurt import VideoKurt
import numpy as np

# Create VideoKurt instance
vk = VideoKurt()

# Configure for dwell time analysis
vk.configure(frame_step=2, resolution_scale=0.5)

# Add dwell_time_maps feature (tracks activity persistence)
vk.add_feature('dwell_time_maps', 
               activity_threshold=10.0,  # Minimum change to consider active
               decay_factor=0.95)  # How quickly activity fades (1.0 = no decay)

print("Computing dwell time maps...")
print("This shows how long activity persists in different regions of the video")
print()

results = vk.analyze('sample_recording.MP4')

# Get dwell time data
dwell_maps = results.features['dwell_time_maps'].data

print("\nDwell Time Maps Results:")
print(f"  Data type: {type(dwell_maps)}")

if isinstance(dwell_maps, dict):
    # Dwell map
    if 'dwell_map' in dwell_maps:
        dwell_map = dwell_maps['dwell_map']
        if hasattr(dwell_map, 'shape'):
            print(f"\n  Dwell map shape: {dwell_map.shape}")
            print(f"  Max dwell time: {dwell_maps.get('max_dwell', 0):.1f} frames")
            print(f"  Mean dwell time: {dwell_maps.get('mean_dwell', 0):.1f} frames")
            print(f"  Active pixels: {dwell_maps.get('active_pixels', 0)}")
            
            # Find hotspots (areas with high dwell time)
            threshold = np.percentile(dwell_map[dwell_map > 0], 90) if np.any(dwell_map > 0) else 0
            hotspot_pixels = np.sum(dwell_map > threshold)
            total_pixels = dwell_map.size
            print(f"  Hotspot coverage: {100*hotspot_pixels/total_pixels:.1f}% of frame")
    
    # Per-frame dwell maps
    if 'dwell_maps' in dwell_maps:
        frame_dwells = dwell_maps['dwell_maps']
        if isinstance(frame_dwells, list) and len(frame_dwells) > 0:
            print(f"\n  Frame-by-frame dwell maps: {len(frame_dwells)} frames")
            
            # Analyze temporal evolution
            max_dwells = [np.max(d) if hasattr(d, 'max') else 0 for d in frame_dwells]
            mean_dwells = [np.mean(d) if hasattr(d, 'mean') else 0 for d in frame_dwells]
            
            if max_dwells:
                print(f"  Peak dwell across frames: {max(max_dwells):.1f}")
                print(f"  Average dwell across frames: {np.mean(mean_dwells):.2f}")
    
    # Dwell zones
    if 'dwell_zones' in dwell_maps:
        zones = dwell_maps['dwell_zones']
        print(f"\n  Identified {len(zones)} dwell zones")
        
        for i, zone in enumerate(zones[:5]):  # Show first 5 zones
            print(f"\n  Zone {i+1}:")
            if isinstance(zone, dict):
                for key, val in zone.items():
                    print(f"    {key}: {val}")
    
    # Statistics
    if 'statistics' in dwell_maps:
        stats = dwell_maps['statistics']
        print(f"\n  Overall Statistics:")
        if 'total_active_pixels' in stats:
            print(f"    Total active pixels: {stats['total_active_pixels']}")
        if 'avg_dwell_time' in stats:
            print(f"    Average dwell time: {stats['avg_dwell_time']:.2f} frames")
        if 'max_dwell_time' in stats:
            print(f"    Maximum dwell time: {stats['max_dwell_time']:.1f} frames")
        if 'persistence_ratio' in stats:
            print(f"    Persistence ratio: {stats['persistence_ratio']:.2%}")

# Analyze dwell patterns
print("\n" + "="*50)
print("Dwell Pattern Analysis:")

if isinstance(dwell_maps, dict):
    if 'dwell_map' in dwell_maps:
        dwell_map = dwell_maps['dwell_map']
        
        # Identify zones by activity level
        if hasattr(dwell_map, 'shape') and len(dwell_map.shape) >= 2:
            h, w = dwell_map.shape[:2]
            
            # Divide into quadrants
            mid_h, mid_w = h // 2, w // 2
            
            top_left = dwell_map[:mid_h, :mid_w]
            top_right = dwell_map[:mid_h, mid_w:]
            bottom_left = dwell_map[mid_h:, :mid_w]
            bottom_right = dwell_map[mid_h:, mid_w:]
            
            quadrant_activity = {
                'Top-left': np.mean(top_left),
                'Top-right': np.mean(top_right),
                'Bottom-left': np.mean(bottom_left),
                'Bottom-right': np.mean(bottom_right)
            }
            
            print("\n  Quadrant Activity (avg dwell time):")
            for quad, activity in sorted(quadrant_activity.items(), 
                                        key=lambda x: x[1], reverse=True):
                print(f"    {quad}: {activity:.2f} frames")
            
            # Identify most active quadrant
            most_active = max(quadrant_activity.items(), key=lambda x: x[1])
            print(f"\n  Most active region: {most_active[0]}")

# Combine with other features for context
print("\n" + "="*50)
print("Combined analysis with motion patterns:")

vk2 = VideoKurt()
vk2.configure(frame_step=2, resolution_scale=0.5)

# Add dwell time with complementary features
vk2.add_feature('dwell_time_maps')
vk2.add_feature('spatial_occupancy_grid', grid_size=(4, 4))  # Spatial activity grid
vk2.add_feature('zone_based_activity', grid_size=(3, 3))  # Zone activity levels

print("\nProcessing with spatial analysis features...")
results2 = vk2.analyze('sample_recording.MP4')

dwell = results2.features['dwell_time_maps'].data
occupancy = results2.features['spatial_occupancy_grid'].data
zones = results2.features.get('zone_based_activity', {}).data if 'zone_based_activity' in results2.features else {}

# Compare spatial patterns
print(f"\nSpatial Activity Comparison:")

if isinstance(occupancy, dict) and 'occupancy_grid' in occupancy:
    grid = occupancy['occupancy_grid']
    if hasattr(grid, 'shape'):
        print(f"  Occupancy grid shape: {grid.shape}")
        print(f"  Most occupied cell: {np.unravel_index(np.argmax(grid), grid.shape)}")
        print(f"  Occupancy range: {np.min(grid):.1f} to {np.max(grid):.1f}")

if isinstance(zones, dict) and 'zone_activities' in zones:
    zone_data = zones['zone_activities']
    print(f"  Zone activity data available: {len(zone_data)} zones")
    
    # Find consistently active zones
    if isinstance(zone_data, dict):
        # zone_data is a dict with zone IDs as keys
        for i, (zone_id, zone_activity) in enumerate(list(zone_data.items())[:3]):
            print(f"\n  Zone {zone_id}:")
            if isinstance(zone_activity, (list, np.ndarray)) and len(zone_activity) > 0:
                print(f"    Activity timeline length: {len(zone_activity)}")
                print(f"    Mean activity: {np.mean(zone_activity):.2f}")
                print(f"    Max activity: {np.max(zone_activity):.2f}")

# Interpret dwell patterns
print("\n" + "="*50)
print("Dwell Pattern Interpretation:")

if isinstance(dwell_maps, dict):
    avg_dwell = dwell_maps.get('mean_dwell', 0)
    max_dwell = dwell_maps.get('max_dwell', 0)
    
    if max_dwell > 100:
        print("  Pattern: Persistent activity regions detected")
        print("  Interpretation: UI elements or content areas with sustained interaction")
    elif avg_dwell > 10:
        print("  Pattern: Moderate activity persistence")
        print("  Interpretation: Regular interaction with some focus areas")
    else:
        print("  Pattern: Transient activity")
        print("  Interpretation: Rapid changes with little persistence")
    
    # Calculate persistence based on active pixels
    active_pixels = dwell_maps.get('active_pixels', 0)
    if 'dwell_map' in dwell_maps:
        total_pixels = dwell_maps['dwell_map'].size
        persistence_ratio = active_pixels / total_pixels if total_pixels > 0 else 0
        
        if persistence_ratio > 0.3:
            print(f"  High activity coverage ({persistence_ratio:.1%}): Widespread interaction")
        elif persistence_ratio > 0.1:
            print(f"  Moderate activity coverage ({persistence_ratio:.1%}): Focused interaction")
        else:
            print(f"  Low activity coverage ({persistence_ratio:.1%}): Limited interaction areas")

print("\n✓ Dwell time analysis complete")