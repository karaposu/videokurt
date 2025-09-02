"""
Exploration 1: Farneback Optical Flow
This detects dense optical flow - motion vectors for every pixel.
Useful for detecting scrolling, sliding, and general motion patterns.


python -m explorations.01_optical_flow_farneback

"""

import cv2
import numpy as np

def analyze_optical_flow_farneback(video_path, max_frames=None, max_seconds=None):
    """Analyze video using Farneback optical flow.
    
    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to process (optional)
        max_seconds: Maximum seconds of video to process (optional)
    """
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate time per frame
    time_per_frame = 1.0 / fps if fps > 0 else 0
    
    # Calculate frame limit based on parameters
    if max_seconds is not None:
        max_frames = int(fps * max_seconds)
        print(f"Processing {max_seconds} seconds = {max_frames} frames at {fps:.1f} fps")
    elif max_frames is None:
        max_frames = total_frames
    
    max_frames = min(max_frames, total_frames)
    
    # Read first frame
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Cannot read first frame")
        return
    
    # Convert to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    # Create HSV image for visualization (Hue for direction, Value for magnitude)
    h, w = prev_gray.shape
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 1] = 255  # Full saturation
    
    frame_count = 0
    motion_stats = []
    
    print(f"Analyzing {video_path}...")
    print(f"Frame size: {w}x{h}")
    print("-" * 50)
    
    # Track frames with low motion
    low_motion_threshold = 0.5
    first_motion_frame = None
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow using Farneback method
        # Parameters: prev, next, flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None, 
            pyr_scale=0.5,  # Image pyramid scale
            levels=3,       # Number of pyramid levels
            winsize=15,     # Window size
            iterations=3,   # Iterations at each level
            poly_n=5,       # Polynomial expansion size
            poly_sigma=1.2, # Gaussian standard deviation
            flags=0
        )
        
        # Calculate motion statistics
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Average motion magnitude
        avg_magnitude = np.mean(mag)
        max_magnitude = np.max(mag)
        
        # Dominant direction (weighted by magnitude)
        avg_angle = np.average(ang.flatten(), weights=mag.flatten())
        
        # Detect motion patterns
        vertical_flow = np.mean(flow[..., 1])  # Positive = down, Negative = up
        horizontal_flow = np.mean(flow[..., 0])  # Positive = right, Negative = left
        
        # Calculate percentage of moving pixels (threshold at 1.0 pixel movement)
        moving_pixels = np.sum(mag > 1.0)
        moving_percentage = (moving_pixels / mag.size) * 100
        
        # Store stats
        motion_stats.append({
            'frame': frame_count,
            'avg_magnitude': avg_magnitude,
            'max_magnitude': max_magnitude,
            'vertical_flow': vertical_flow,
            'horizontal_flow': horizontal_flow,
            'moving_percentage': moving_percentage
        })
        
        # Track first frame with motion
        if first_motion_frame is None and avg_magnitude > low_motion_threshold:
            first_motion_frame = frame_count
            print(f"First significant motion detected at frame {frame_count} ({frame_count/fps:.2f}s)")
            print("-" * 50)
        
        # Print significant motion
        if avg_magnitude > 0.5:
            direction = ""
            if abs(vertical_flow) > abs(horizontal_flow):
                direction = "UP" if vertical_flow < -0.5 else "DOWN" if vertical_flow > 0.5 else ""
            else:
                direction = "LEFT" if horizontal_flow < -0.5 else "RIGHT" if horizontal_flow > 0.5 else ""
            
            # Calculate timestamp
            timestamp = frame_count * time_per_frame
            mins = int(timestamp // 60)
            secs = timestamp % 60
            
            print(f"Frame {frame_count:3d} ({mins:02d}:{secs:05.2f}): Magnitude={avg_magnitude:.2f}, "
                  f"Moving={moving_percentage:.1f}%, Direction={direction}")
        
        # Visualize optical flow (optional - comment out for faster processing)
        if frame_count % 10 == 0:  # Every 10th frame
            # Convert angle to hue (0-179 for OpenCV)
            hsv[..., 0] = ang * 180 / np.pi / 2
            # Convert magnitude to value (brightness)
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            
            # Save visualization
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            # cv2.imwrite(f'flow_frame_{frame_count:03d}.png', bgr)
        
        prev_gray = gray
        frame_count += 1
    
    cap.release()
    
    # Analyze overall patterns
    print("\n" + "=" * 50)
    print("MOTION ANALYSIS SUMMARY")
    print("=" * 50)
    
    if motion_stats:
        avg_stats = {
            'avg_magnitude': np.mean([s['avg_magnitude'] for s in motion_stats]),
            'avg_vertical': np.mean([s['vertical_flow'] for s in motion_stats]),
            'avg_horizontal': np.mean([s['horizontal_flow'] for s in motion_stats]),
            'avg_moving': np.mean([s['moving_percentage'] for s in motion_stats])
        }
        
        print(f"Average motion magnitude: {avg_stats['avg_magnitude']:.2f} pixels/frame")
        print(f"Average moving pixels: {avg_stats['avg_moving']:.1f}%")
        
        # Detect dominant patterns
        if avg_stats['avg_magnitude'] < 0.1:
            print("Pattern: MOSTLY IDLE")
        elif abs(avg_stats['avg_vertical']) > abs(avg_stats['avg_horizontal']) * 2:
            if avg_stats['avg_vertical'] < -0.5:
                print("Pattern: VERTICAL SLIDING UP (scrolling down)")
            elif avg_stats['avg_vertical'] > 0.5:
                print("Pattern: VERTICAL SLIDING DOWN (scrolling up)")
        elif abs(avg_stats['avg_horizontal']) > abs(avg_stats['avg_vertical']) * 2:
            if avg_stats['avg_horizontal'] < -0.5:
                print("Pattern: HORIZONTAL SLIDING LEFT")
            elif avg_stats['avg_horizontal'] > 0.5:
                print("Pattern: HORIZONTAL SLIDING RIGHT")
        else:
            print("Pattern: MIXED MOTION")
        
        # Detect if motion is localized or full screen
        if avg_stats['avg_moving'] < 20:
            print("Scope: LOCALIZED (mini) - motion in small regions")
        elif avg_stats['avg_moving'] > 50:
            print("Scope: FULL SCREEN - motion across entire frame")
        else:
            print("Scope: PARTIAL - motion in medium regions")
    
    return motion_stats

if __name__ == "__main__":
    # Test with the screen recording
    video_path = "ScreenRecording_08-27-2025 16-38-43_1.MP4"
    
    
    stats = analyze_optical_flow_farneback(video_path, max_seconds=30)
    
    print("\n💡 This method is good for:")
    print("  - Detecting scrolling direction and speed")
    print("  - Identifying full vs partial screen motion")
    print("  - Finding smooth continuous movements")
    print("  - Detecting sliding UI elements")