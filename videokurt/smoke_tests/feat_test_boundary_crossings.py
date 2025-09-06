"""Boundary crossings example - detects objects crossing defined boundaries."""

# to run python videokurt/smoke_tests/feat_test_boundary_crossings.py
from videokurt import VideoKurt
import numpy as np

# Create VideoKurt instance
vk = VideoKurt()

# Configure for boundary crossing detection
vk.configure(frame_step=2, resolution_scale=0.6)

# Add boundary_crossings feature
# Default boundaries are frame edges if not specified
vk.add_feature('boundary_crossings',
               boundaries=[
                   {'type': 'horizontal', 'y': 240, 'name': 'mid_horizontal'},
                   {'type': 'vertical', 'x': 320, 'name': 'mid_vertical'}
               ])

print("Detecting boundary crossings in video...")
print("This tracks when objects cross defined lines in the frame")
print()

results = vk.analyze('sample_recording.MP4')

# Get boundary crossing results
crossings = results.features['boundary_crossings'].data

print("\nBoundary Crossing Results:")
print(f"  Data type: {type(crossings)}")

if isinstance(crossings, dict):
    # Crossing events
    if 'crossing_events' in crossings:
        events = crossings['crossing_events']
        print(f"\n  Total crossing events: {len(events)}")
        
        # Analyze events by boundary
        boundary_counts = {}
        for event in events:
            boundary = event.get('boundary', 'unknown')
            if boundary not in boundary_counts:
                boundary_counts[boundary] = 0
            boundary_counts[boundary] += 1
        
        if boundary_counts:
            print(f"\n  Crossings by boundary:")
            for boundary, count in boundary_counts.items():
                print(f"    {boundary}: {count} crossings")
        
        # Show first few events
        for i, event in enumerate(events[:5]):
            print(f"\n  Event {i+1}:")
            print(f"    Frame: {event.get('frame', 0)}")
            print(f"    Boundary: {event.get('boundary', 'unknown')}")
            print(f"    Direction: {event.get('direction', 'unknown')}")
            if 'position' in event:
                print(f"    Position: ({event['position'][0]:.0f}, {event['position'][1]:.0f})")
            if 'velocity' in event:
                print(f"    Velocity: {event['velocity']:.2f}")
    
    # Crossing statistics
    if 'statistics' in crossings:
        stats = crossings['statistics']
        print(f"\n  Crossing Statistics:")
        
        if 'total_crossings' in stats:
            print(f"    Total crossings: {stats['total_crossings']}")
        
        if 'crossings_per_boundary' in stats:
            print(f"    Crossings per boundary:")
            for boundary, data in stats['crossings_per_boundary'].items():
                print(f"      {boundary}: {data.get('count', 0)} ({data.get('percentage', 0):.1f}%)")
        
        if 'crossing_rate' in stats:
            print(f"    Crossing rate: {stats['crossing_rate']:.2f} per frame")
        
        if 'peak_activity_frame' in stats:
            print(f"    Peak activity frame: {stats['peak_activity_frame']}")
    
    # Direction analysis
    if 'direction_analysis' in crossings:
        directions = crossings['direction_analysis']
        print(f"\n  Direction Analysis:")
        
        if 'inbound' in directions:
            print(f"    Inbound crossings: {directions['inbound']}")
        if 'outbound' in directions:
            print(f"    Outbound crossings: {directions['outbound']}")
        if 'lateral' in directions:
            print(f"    Lateral crossings: {directions['lateral']}")
        
        # Flow pattern
        if 'flow_pattern' in directions:
            print(f"    Dominant flow: {directions['flow_pattern']}")

# Analyze crossing patterns
print("\n" + "="*50)
print("Crossing Pattern Analysis:")

if isinstance(crossings, dict) and 'crossing_events' in crossings:
    events = crossings['crossing_events']
    
    if len(events) == 0:
        print("  No boundary crossings detected")
    elif len(events) < 5:
        print("  Pattern: Rare crossings")
        print("  Interpretation: Minimal movement across boundaries")
    elif len(events) < 20:
        print("  Pattern: Moderate crossing activity")
        print("  Interpretation: Regular movement across boundaries")
    else:
        print("  Pattern: Frequent crossings")
        print("  Interpretation: High activity with many boundary transitions")
    
    # Temporal distribution
    if events:
        frames = [e.get('frame', 0) for e in events]
        if frames:
            frame_range = max(frames) - min(frames)
            if frame_range > 0:
                density = len(events) / frame_range
                print(f"\n  Temporal density: {density:.2f} crossings per frame")
                
                if density < 0.1:
                    print("  Crossings are sparse")
                elif density < 0.5:
                    print("  Crossings are moderately distributed")
                else:
                    print("  Crossings are densely packed")

# Combine with entry/exit detection
print("\n" + "="*50)
print("Entry/Exit Detection:")

vk2 = VideoKurt()
vk2.configure(frame_step=2, resolution_scale=0.6)

# Add boundary crossing with frame edge detection
vk2.add_feature('boundary_crossings',
               boundaries=[
                   {'type': 'frame_edge', 'side': 'top'},
                   {'type': 'frame_edge', 'side': 'bottom'},
                   {'type': 'frame_edge', 'side': 'left'},
                   {'type': 'frame_edge', 'side': 'right'}
               ])

print("\nProcessing with frame edge boundaries...")
results2 = vk2.analyze('sample_recording.MP4')

edge_crossings = results2.features['boundary_crossings'].data

if isinstance(edge_crossings, dict):
    if 'crossing_events' in edge_crossings:
        events = edge_crossings['crossing_events']
        
        # Count entries and exits
        entries = 0
        exits = 0
        
        for event in events:
            direction = event.get('direction', '')
            if 'enter' in direction.lower() or 'in' in direction.lower():
                entries += 1
            elif 'exit' in direction.lower() or 'out' in direction.lower():
                exits += 1
        
        print(f"\n  Frame Entry/Exit Summary:")
        print(f"    Objects entering frame: {entries}")
        print(f"    Objects exiting frame: {exits}")
        print(f"    Net flow: {entries - exits}")
        
        if entries > exits:
            print("    Pattern: More objects entering than leaving")
        elif exits > entries:
            print("    Pattern: More objects leaving than entering")
        else:
            print("    Pattern: Balanced entry/exit flow")
    
    # Edge activity distribution
    if 'statistics' in edge_crossings:
        stats = edge_crossings['statistics']
        if 'crossings_per_boundary' in stats:
            print(f"\n  Edge Activity Distribution:")
            for edge, data in stats['crossings_per_boundary'].items():
                count = data.get('count', 0)
                percentage = data.get('percentage', 0)
                print(f"    {edge}: {count} crossings ({percentage:.1f}%)")

# Interpret boundary crossing patterns
print("\n" + "="*50)
print("Boundary Crossing Interpretation:")

if isinstance(crossings, dict):
    total_crossings = len(crossings.get('crossing_events', []))
    
    if total_crossings == 0:
        print("  No movement across boundaries detected")
        print("  Interpretation: Static content or movement within zones")
    else:
        # Check for patterns
        if 'direction_analysis' in crossings:
            directions = crossings['direction_analysis']
            inbound = directions.get('inbound', 0)
            outbound = directions.get('outbound', 0)
            
            if inbound > outbound * 1.5:
                print("  Pattern: Convergent flow")
                print("  Interpretation: Objects moving toward center")
            elif outbound > inbound * 1.5:
                print("  Pattern: Divergent flow")
                print("  Interpretation: Objects moving away from center")
            else:
                print("  Pattern: Bidirectional flow")
                print("  Interpretation: Objects moving in multiple directions")

print("\n✓ Boundary crossing analysis complete")