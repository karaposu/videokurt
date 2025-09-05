"""Motion trajectories example - tracks movement paths through video."""

# to run python example_motion_trajectories.py
from videokurt import VideoKurt
import numpy as np

# Create VideoKurt instance
vk = VideoKurt()

# Configure for trajectory tracking
vk.configure(frame_step=2, resolution_scale=0.6)

# Add motion_trajectories feature (can use sparse or dense optical flow)
vk.add_feature('motion_trajectories', 
               min_trajectory_length=10,  # Minimum 10 frames for valid trajectory
               max_trajectories=50)  # Track up to 50 trajectories

print("Extracting motion trajectories from video...")
print("This uses optical flow to track movement paths")
print()

results = vk.analyze('sample_recording.MP4')

# Get trajectory data
trajectories = results.features['motion_trajectories'].data

print("\nMotion Trajectory Results:")
print(f"  Data type: {type(trajectories)}")

if isinstance(trajectories, dict):
    # Trajectory paths
    if 'trajectories' in trajectories:
        traj_list = trajectories['trajectories']
        print(f"\n  Found {len(traj_list)} trajectories")
        
        # Analyze first few trajectories
        for i, traj in enumerate(traj_list[:5]):
            print(f"\n  Trajectory {i+1}:")
            if 'start_frame' in traj:
                print(f"    Start frame: {traj['start_frame']}")
            else:
                print(f"    Start frame: N/A")
            if 'end_frame' in traj:
                print(f"    End frame: {traj['end_frame']}")
            if 'length' in traj:
                print(f"    Length: {traj['length']} frames")
            if 'path' in traj:
                path = traj['path']
                if isinstance(path, (list, np.ndarray)) and len(path) > 0:
                    # Calculate total distance traveled
                    if len(path) > 1:
                        distances = [np.linalg.norm(np.array(path[i+1]) - np.array(path[i])) 
                                   for i in range(len(path)-1)]
                        total_distance = sum(distances)
                        print(f"    Total distance: {total_distance:.1f} pixels")
                        print(f"    Average speed: {total_distance/len(path):.2f} pixels/frame")
            if 'type' in traj:
                print(f"    Type: {traj['type']}")
    
    # Statistics
    if 'statistics' in trajectories:
        stats = trajectories['statistics']
        print(f"\n  Overall Statistics:")
        print(f"    Total trajectories: {stats.get('total_trajectories', 0)}")
        print(f"    Average length: {stats.get('avg_length', 0):.1f} frames")
        print(f"    Longest trajectory: {stats.get('max_length', 0)} frames")
        
        if 'dominant_direction' in stats:
            print(f"    Dominant direction: {stats['dominant_direction']}")

# Analyze trajectory patterns
print("\n" + "="*50)
print("Trajectory Pattern Analysis:")

if isinstance(trajectories, dict) and 'trajectories' in trajectories:
    traj_list = trajectories['trajectories']
    
    if len(traj_list) > 0:
        # Classify trajectories by length
        short_traj = sum(1 for t in traj_list if t.get('length', 0) < 20)
        medium_traj = sum(1 for t in traj_list if 20 <= t.get('length', 0) < 50)
        long_traj = sum(1 for t in traj_list if t.get('length', 0) >= 50)
        
        print(f"\nTrajectory Length Distribution:")
        print(f"  Short (<20 frames): {short_traj}")
        print(f"  Medium (20-50 frames): {medium_traj}")
        print(f"  Long (>50 frames): {long_traj}")
        
        # Analyze trajectory shapes
        trajectory_types = {}
        for traj in traj_list:
            traj_type = traj.get('type', 'unknown')
            trajectory_types[traj_type] = trajectory_types.get(traj_type, 0) + 1
        
        if trajectory_types:
            print(f"\nTrajectory Types:")
            for traj_type, count in sorted(trajectory_types.items(), key=lambda x: x[1], reverse=True):
                print(f"  {traj_type}: {count}")

# Combine with other motion features
print("\n" + "="*50)
print("Combined motion analysis:")

vk2 = VideoKurt()
vk2.configure(frame_step=2, resolution_scale=0.6)

# Add multiple motion tracking features
vk2.add_feature('motion_trajectories')
vk2.add_feature('dominant_flow_vector')  # Overall motion direction
vk2.add_feature('motion_magnitude')  # Motion intensity

print("\nProcessing with multiple motion features...")
results2 = vk2.analyze('sample_recording.MP4')

trajectories = results2.features['motion_trajectories'].data
flow_vectors = results2.features['dominant_flow_vector'].data
motion_mag = results2.features['motion_magnitude'].data

# Compare trajectory motion with overall flow
print(f"\nMotion Comparison:")
if isinstance(trajectories, dict):
    num_traj = len(trajectories.get('trajectories', []))
    print(f"  Tracked trajectories: {num_traj}")

if isinstance(flow_vectors, np.ndarray) and len(flow_vectors) > 0:
    # Analyze dominant motion direction
    avg_flow_x = np.mean(flow_vectors[:, 0])
    avg_flow_y = np.mean(flow_vectors[:, 1])
    
    print(f"  Average flow direction: ({avg_flow_x:.2f}, {avg_flow_y:.2f})")
    
    if abs(avg_flow_y) > abs(avg_flow_x):
        if avg_flow_y > 0:
            print(f"  Primary motion: Downward")
        else:
            print(f"  Primary motion: Upward")
    elif abs(avg_flow_x) > 0.1:
        if avg_flow_x > 0:
            print(f"  Primary motion: Rightward")
        else:
            print(f"  Primary motion: Leftward")
    else:
        print(f"  Primary motion: Minimal/Mixed")

if isinstance(motion_mag, np.ndarray):
    print(f"  Average motion magnitude: {motion_mag.mean():.3f}")
    high_motion_frames = np.sum(motion_mag > motion_mag.mean() + motion_mag.std())
    print(f"  High motion frames: {high_motion_frames}")

print("\n✓ Motion trajectory analysis complete")