"""Spatial occupancy grid example - shows activity distribution across frame regions."""

# to run python example_spatial_occupancy.py
from videokurt import VideoKurt
import numpy as np

# Create VideoKurt instance
vk = VideoKurt()

# Configure for spatial analysis
vk.configure(frame_step=2, resolution_scale=0.5)

# Add spatial_occupancy_grid feature
vk.add_feature('spatial_occupancy_grid', 
               grid_size=(4, 4),  # Divide frame into 4x4 grid
               threshold=10)  # Activity threshold

print("Computing spatial occupancy grid...")
print("This shows how activity is distributed across different regions of the frame")
print()

results = vk.analyze('sample_recording.MP4')

# Get occupancy grid results
occupancy = results.features['spatial_occupancy_grid'].data

print("\nSpatial Occupancy Grid Results:")
print(f"  Data type: {type(occupancy)}")

if isinstance(occupancy, dict):
    # Occupancy grid
    if 'occupancy_grid' in occupancy:
        grid = occupancy['occupancy_grid']
        print(f"\n  Grid shape: {grid.shape}")
        print(f"  Occupancy range: {grid.min():.2f} to {grid.max():.2f}")
        
        # Display grid as text visualization
        print("\n  Occupancy Grid Visualization:")
        print("  (Higher values = more activity)")
        print()
        
        # Normalize for display
        norm_grid = (grid - grid.min()) / (grid.max() - grid.min() + 1e-8)
        
        for i in range(grid.shape[0]):
            row_str = "    "
            for j in range(grid.shape[1]):
                val = norm_grid[i, j]
                if val < 0.2:
                    symbol = "░"  # Low activity
                elif val < 0.4:
                    symbol = "▒"  # Low-medium
                elif val < 0.6:
                    symbol = "▓"  # Medium
                elif val < 0.8:
                    symbol = "█"  # High-medium
                else:
                    symbol = "●"  # High activity
                row_str += f" {symbol} "
            print(row_str)
        
        # Find hotspots
        flat_idx = np.argmax(grid)
        max_cell = np.unravel_index(flat_idx, grid.shape)
        print(f"\n  Hottest cell: {max_cell} (row {max_cell[0]}, col {max_cell[1]})")
        
        flat_idx_min = np.argmin(grid)
        min_cell = np.unravel_index(flat_idx_min, grid.shape)
        print(f"  Coldest cell: {min_cell} (row {min_cell[0]}, col {min_cell[1]})")
    
    # Cell activities
    if 'cell_activities' in occupancy:
        activities = occupancy['cell_activities']
        if isinstance(activities, dict):
            print(f"\n  Cell-by-cell activity levels:")
            sorted_cells = sorted(activities.items(), key=lambda x: np.mean(x[1]), reverse=True)
            for i, (cell, activity) in enumerate(sorted_cells[:5]):
                if isinstance(activity, (list, np.ndarray)) and len(activity) > 0:
                    print(f"    Cell {cell}: mean={np.mean(activity):.2f}, max={np.max(activity):.2f}")
    
    # Statistics
    if 'statistics' in occupancy:
        stats = occupancy['statistics']
        print(f"\n  Grid Statistics:")
        if 'total_active_cells' in stats:
            print(f"    Active cells: {stats['total_active_cells']}")
        if 'activity_variance' in stats:
            print(f"    Activity variance: {stats['activity_variance']:.2f}")
        if 'spatial_entropy' in stats:
            print(f"    Spatial entropy: {stats['spatial_entropy']:.2f}")

# Analyze spatial patterns
print("\n" + "="*50)
print("Spatial Activity Pattern Analysis:")

if isinstance(occupancy, dict) and 'occupancy_grid' in occupancy:
    grid = occupancy['occupancy_grid']
    
    # Analyze quadrants
    h, w = grid.shape
    mid_h, mid_w = h // 2, w // 2
    
    quadrants = {
        'Top-left': grid[:mid_h, :mid_w],
        'Top-right': grid[:mid_h, mid_w:],
        'Bottom-left': grid[mid_h:, :mid_w],
        'Bottom-right': grid[mid_h:, mid_w:]
    }
    
    print("\n  Quadrant Activity:")
    quad_means = {}
    for name, quad in quadrants.items():
        mean_activity = np.mean(quad)
        quad_means[name] = mean_activity
        print(f"    {name}: {mean_activity:.3f}")
    
    # Find dominant quadrant
    dominant = max(quad_means.items(), key=lambda x: x[1])
    print(f"\n  Most active quadrant: {dominant[0]}")
    
    # Check for patterns
    vertical_balance = abs(np.mean(grid[:mid_h, :]) - np.mean(grid[mid_h:, :]))
    horizontal_balance = abs(np.mean(grid[:, :mid_w]) - np.mean(grid[:, mid_w:]))
    
    print(f"\n  Balance Analysis:")
    if vertical_balance < 0.05:
        print("    Vertical: Balanced (top/bottom similar)")
    else:
        if np.mean(grid[:mid_h, :]) > np.mean(grid[mid_h:, :]):
            print("    Vertical: Top-heavy activity")
        else:
            print("    Vertical: Bottom-heavy activity")
    
    if horizontal_balance < 0.05:
        print("    Horizontal: Balanced (left/right similar)")
    else:
        if np.mean(grid[:, :mid_w]) > np.mean(grid[:, mid_w:]):
            print("    Horizontal: Left-heavy activity")
        else:
            print("    Horizontal: Right-heavy activity")

# Combine with zone-based analysis
print("\n" + "="*50)
print("Combined spatial analysis:")

vk2 = VideoKurt()
vk2.configure(frame_step=2, resolution_scale=0.5)

# Add spatial features
vk2.add_feature('spatial_occupancy_grid', grid_size=(3, 3))
vk2.add_feature('zone_based_activity', grid_size=(3, 3))

print("\nProcessing with zone activity features...")
results2 = vk2.analyze('sample_recording.MP4')

occupancy = results2.features['spatial_occupancy_grid'].data
zones = results2.features['zone_based_activity'].data

print(f"\nSpatial Analysis Summary:")

if isinstance(occupancy, dict) and 'occupancy_grid' in occupancy:
    grid = occupancy['occupancy_grid']
    # Calculate concentration
    flat_grid = grid.flatten()
    sorted_values = np.sort(flat_grid)[::-1]
    top_20_percent = int(len(sorted_values) * 0.2)
    concentration = np.sum(sorted_values[:top_20_percent]) / np.sum(sorted_values)
    
    print(f"  Activity concentration: {concentration:.1%} in top 20% of cells")
    
    if concentration > 0.6:
        print("  Pattern: Highly concentrated activity")
    elif concentration > 0.4:
        print("  Pattern: Moderately concentrated activity")
    else:
        print("  Pattern: Dispersed activity")

if isinstance(zones, dict):
    if 'most_active_zone' in zones:
        print(f"  Most active zone: {zones['most_active_zone']}")
    
    if 'zone_statistics' in zones:
        zone_stats = zones['zone_statistics']
        if zone_stats:
            active_zones = sum(1 for zone_name, stats in zone_stats.items() if stats.get('mean_activity', 0) > 0.1)
            print(f"  Active zones: {active_zones} out of {len(zone_stats)}")

# Interpret spatial patterns
print("\n" + "="*50)
print("Spatial Pattern Interpretation:")

if isinstance(occupancy, dict) and 'occupancy_grid' in occupancy:
    grid = occupancy['occupancy_grid']
    
    # Check for common UI patterns
    top_row_activity = np.mean(grid[0, :])
    bottom_row_activity = np.mean(grid[-1, :])
    center_activity = np.mean(grid[1:-1, 1:-1]) if grid.shape[0] > 2 else 0
    
    if top_row_activity > center_activity * 1.5:
        print("  High top activity: Possible header/navigation area")
    if bottom_row_activity > center_activity * 1.5:
        print("  High bottom activity: Possible footer/control area")
    if center_activity > (top_row_activity + bottom_row_activity) / 2:
        print("  High center activity: Main content area")
    
    # Overall interpretation
    std_activity = np.std(grid)
    mean_activity = np.mean(grid)
    cv = std_activity / mean_activity if mean_activity > 0 else 0
    
    if cv < 0.3:
        print("\n  Overall: Uniform activity distribution")
        print("  Interpretation: Activity spread across entire frame")
    elif cv < 0.7:
        print("\n  Overall: Moderate spatial variation")
        print("  Interpretation: Some areas more active than others")
    else:
        print("\n  Overall: High spatial variation")
        print("  Interpretation: Activity concentrated in specific regions")

print("\n✓ Spatial occupancy analysis complete")