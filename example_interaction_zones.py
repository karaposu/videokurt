"""Interaction zones example - detects areas where objects interact."""

# to run python example_interaction_zones.py
from videokurt import VideoKurt
import numpy as np

# Create VideoKurt instance
vk = VideoKurt()

# Configure for interaction zone detection
vk.configure(frame_step=2, resolution_scale=0.6)

# Add interaction_zones feature
vk.add_feature('interaction_zones',
               min_overlap=0.3,  # 30% overlap to consider interaction
               min_area=100)  # Minimum area for blobs

print("Detecting interaction zones in video...")
print("This identifies regions where multiple objects overlap or interact")
print()

results = vk.analyze('sample_recording.MP4')

# Get interaction zone results
interactions = results.features['interaction_zones'].data

print("\nInteraction Zone Results:")
print(f"  Data type: {type(interactions)}")

if isinstance(interactions, dict):
    # Interaction events
    if 'interaction_events' in interactions:
        events = interactions['interaction_events']
        print(f"\n  Total interaction events: {len(events)}")
        
        # Show first few events
        for i, event in enumerate(events[:5]):
            print(f"\n  Interaction {i+1}:")
            print(f"    Start frame: {event.get('start_frame', 0)}")
            print(f"    End frame: {event.get('end_frame', 0)}")
            duration = event.get('end_frame', 0) - event.get('start_frame', 0) + 1
            print(f"    Duration: {duration} frames")
            
            if 'location' in event:
                x, y = event['location']
                print(f"    Location: ({x:.0f}, {y:.0f})")
            
            if 'overlap_ratio' in event:
                print(f"    Overlap ratio: {event['overlap_ratio']:.1%}")
            
            if 'intensity' in event:
                print(f"    Intensity: {event['intensity']:.2f}")
            
            if 'participants' in event:
                print(f"    Participants: {event['participants']}")
    
    # Hotspot analysis
    if 'hotspots' in interactions:
        hotspots = interactions['hotspots']
        print(f"\n  Interaction Hotspots: {len(hotspots)} found")
        
        for i, hotspot in enumerate(hotspots[:3]):
            print(f"\n  Hotspot {i+1}:")
            if 'center' in hotspot:
                print(f"    Center: ({hotspot['center'][0]:.0f}, {hotspot['center'][1]:.0f})")
            if 'radius' in hotspot:
                print(f"    Radius: {hotspot['radius']:.0f} pixels")
            if 'frequency' in hotspot:
                print(f"    Interaction frequency: {hotspot['frequency']}")
            if 'avg_duration' in hotspot:
                print(f"    Average duration: {hotspot['avg_duration']:.1f} frames")
    
    # Statistics
    if 'statistics' in interactions:
        stats = interactions['statistics']
        print(f"\n  Interaction Statistics:")
        
        if 'total_interactions' in stats:
            print(f"    Total interactions: {stats['total_interactions']}")
        
        if 'avg_duration' in stats:
            print(f"    Average duration: {stats['avg_duration']:.1f} frames")
        
        if 'avg_overlap' in stats:
            print(f"    Average overlap: {stats['avg_overlap']:.1%}")
        
        if 'interaction_density' in stats:
            print(f"    Interaction density: {stats['interaction_density']:.3f} per frame")
        
        if 'max_simultaneous' in stats:
            print(f"    Max simultaneous interactions: {stats['max_simultaneous']}")
    
    # Temporal distribution
    if 'temporal_distribution' in interactions:
        temporal = interactions['temporal_distribution']
        if temporal:
            print(f"\n  Temporal Distribution:")
            
            # Find peaks
            peak_frames = [f for f, count in temporal.items() if count > np.mean(list(temporal.values()))]
            if peak_frames:
                print(f"    Peak activity frames: {peak_frames[:5]}")
            
            # Activity phases
            total_frames = len(temporal)
            active_frames = sum(1 for count in temporal.values() if count > 0)
            print(f"    Active frames: {active_frames}/{total_frames} ({100*active_frames/total_frames:.1f}%)")

# Analyze interaction patterns
print("\n" + "="*50)
print("Interaction Pattern Analysis:")

if isinstance(interactions, dict) and 'interaction_events' in interactions:
    events = interactions['interaction_events']
    
    if len(events) == 0:
        print("  No interactions detected")
        print("  Interpretation: Objects move independently")
    elif len(events) < 5:
        print("  Pattern: Rare interactions")
        print("  Interpretation: Mostly independent movement with occasional overlap")
    elif len(events) < 20:
        print("  Pattern: Moderate interaction frequency")
        print("  Interpretation: Regular object interactions")
    else:
        print("  Pattern: Frequent interactions")
        print("  Interpretation: High level of object overlap and interaction")
    
    # Analyze interaction types
    if events:
        durations = [e.get('end_frame', 0) - e.get('start_frame', 0) + 1 for e in events]
        avg_duration = np.mean(durations)
        
        if avg_duration < 5:
            print("\n  Interaction type: Brief encounters")
        elif avg_duration < 15:
            print("\n  Interaction type: Moderate interactions")
        else:
            print("\n  Interaction type: Extended interactions")

# Combine with collision detection
print("\n" + "="*50)
print("Collision and Merge Analysis:")

vk2 = VideoKurt()
vk2.configure(frame_step=2, resolution_scale=0.6)

# Add interaction zones with higher overlap threshold
vk2.add_feature('interaction_zones', 
               min_overlap=0.7,  # Higher threshold for collision detection
               min_area=50)  # Smaller minimum area

print("\nProcessing with collision detection settings...")
results2 = vk2.analyze('sample_recording.MP4')

collisions = results2.features['interaction_zones'].data

if isinstance(collisions, dict):
    collision_events = collisions.get('interaction_events', [])
    regular_events = interactions.get('interaction_events', [])
    
    print(f"\n  Collision Analysis:")
    print(f"    Regular interactions: {len(regular_events)}")
    print(f"    High-overlap collisions: {len(collision_events)}")
    
    if len(collision_events) > 0:
        collision_ratio = len(collision_events) / len(regular_events) if regular_events else 0
        print(f"    Collision ratio: {collision_ratio:.1%} of interactions are collisions")
        
        if collision_ratio > 0.5:
            print("    Pattern: Frequent collisions/merges")
        elif collision_ratio > 0.2:
            print("    Pattern: Occasional collisions")
        else:
            print("    Pattern: Mostly near-misses")

# Spatial clustering of interactions
print("\n" + "="*50)
print("Spatial Interaction Patterns:")

if isinstance(interactions, dict) and 'hotspots' in interactions:
    hotspots = interactions['hotspots']
    
    if len(hotspots) == 0:
        print("  No interaction hotspots identified")
    elif len(hotspots) == 1:
        print("  Single interaction zone")
        print("  Interpretation: Interactions concentrated in one area")
    elif len(hotspots) < 4:
        print("  Few interaction zones")
        print("  Interpretation: Interactions occur in specific regions")
    else:
        print("  Multiple interaction zones")
        print("  Interpretation: Distributed interaction patterns across frame")
    
    # Check for central vs peripheral interactions
    if hotspots and 'center' in hotspots[0]:
        # Assume frame dimensions (would be from actual video metadata)
        frame_center_x, frame_center_y = 320, 240
        
        central_interactions = 0
        for hotspot in hotspots:
            if 'center' in hotspot:
                x, y = hotspot['center']
                dist_from_center = np.sqrt((x - frame_center_x)**2 + (y - frame_center_y)**2)
                if dist_from_center < 150:  # Within central region
                    central_interactions += 1
        
        if central_interactions > len(hotspots) * 0.7:
            print("\n  Location pattern: Central interactions")
        elif central_interactions < len(hotspots) * 0.3:
            print("\n  Location pattern: Peripheral interactions")
        else:
            print("\n  Location pattern: Mixed central/peripheral")

print("\n✓ Interaction zone analysis complete")