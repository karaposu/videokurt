"""
Exploration 2: Lucas-Kanade Optical Flow
This detects sparse optical flow - tracks specific feature points.
Useful for tracking UI elements, detecting clicks, and following moving objects.


python -m explorations.02_optical_flow_lucas_kanade



"""

import cv2
import numpy as np

def analyze_optical_flow_lucas_kanade(video_path, max_frames=None, max_seconds=None):
    """Analyze video using Lucas-Kanade optical flow.
    
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
    
    # Parameters for ShiTomasi corner detection (good features to track)
    feature_params = dict(
        maxCorners=100,
        qualityLevel=0.3,
        minDistance=7,
        blockSize=7
    )
    
    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )
    
    # Read first frame
    ret, old_frame = cap.read()
    if not ret:
        print("Error: Cannot read first frame")
        return
    
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    
    # Detect initial features to track
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    
    if p0 is None:
        print("No features found in first frame")
        return
    
    # Create a mask for drawing tracks
    mask = np.zeros_like(old_frame)
    
    # Colors for visualization
    colors = np.random.randint(0, 255, (100, 3))
    
    frame_count = 0
    motion_patterns = []
    
    print(f"Analyzing {video_path}...")
    print(f"Initial features detected: {len(p0)}")
    print("-" * 50)
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow
        if p0 is not None and len(p0) > 0:
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            
            # Select only good points
            if p1 is not None:
                good_new = p1[st == 1]
                good_old = p0[st == 1]
                
                # Calculate motion vectors
                if len(good_new) > 0:
                    motion_vectors = good_new - good_old
                    
                    # Calculate average motion
                    avg_motion_x = np.mean(motion_vectors[:, 0])
                    avg_motion_y = np.mean(motion_vectors[:, 1])
                    avg_magnitude = np.mean(np.sqrt(motion_vectors[:, 0]**2 + motion_vectors[:, 1]**2))
                    
                    # Detect motion patterns
                    if avg_magnitude > 0.5:
                        direction = ""
                        if abs(avg_motion_y) > abs(avg_motion_x):
                            direction = "VERTICAL_SLIDE_UP" if avg_motion_y < -1 else "VERTICAL_SLIDE_DOWN" if avg_motion_y > 1 else ""
                        else:
                            direction = "HORIZONTAL_SLIDE_LEFT" if avg_motion_x < -1 else "HORIZONTAL_SLIDE_RIGHT" if avg_motion_x > 1 else ""
                        
                        # Check if motion is uniform (scrolling) or scattered (changes)
                        motion_std_x = np.std(motion_vectors[:, 0])
                        motion_std_y = np.std(motion_vectors[:, 1])
                        
                        if motion_std_x < 2 and motion_std_y < 2:
                            pattern = "UNIFORM_MOTION"  # All points moving similarly
                        else:
                            pattern = "SCATTERED_MOTION"  # Points moving differently
                        
                        # Calculate timestamp
                        timestamp = frame_count * time_per_frame
                        mins = int(timestamp // 60)
                        secs = timestamp % 60
                        
                        print(f"Frame {frame_count:3d} ({mins:02d}:{secs:05.2f}): Tracked={len(good_new)}, "
                              f"Magnitude={avg_magnitude:.2f}, {pattern}, {direction}")
                        
                        motion_patterns.append({
                            'frame': frame_count,
                            'num_points': len(good_new),
                            'avg_motion_x': avg_motion_x,
                            'avg_motion_y': avg_motion_y,
                            'magnitude': avg_magnitude,
                            'pattern': pattern,
                            'direction': direction
                        })
                    
                    # Update points for next iteration
                    p0 = good_new.reshape(-1, 1, 2)
                else:
                    p0 = None
        
        # Re-detect features every 30 frames or if too few points
        if frame_count % 30 == 0 or p0 is None or len(p0) < 10:
            p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
            if p0 is not None:
                mask = np.zeros_like(old_frame)  # Reset tracks
                if frame_count % 30 == 0:
                    print(f"Frame {frame_count}: Re-detected {len(p0)} features")
        
        old_gray = frame_gray.copy()
        frame_count += 1
    
    cap.release()
    
    # Analyze patterns
    print("\n" + "=" * 50)
    print("LUCAS-KANADE MOTION ANALYSIS SUMMARY")
    print("=" * 50)
    
    if motion_patterns:
        # Count pattern types
        uniform_count = sum(1 for p in motion_patterns if p['pattern'] == 'UNIFORM_MOTION')
        scattered_count = sum(1 for p in motion_patterns if p['pattern'] == 'SCATTERED_MOTION')
        
        print(f"Total motion events: {len(motion_patterns)}")
        print(f"Uniform motion (scrolling/sliding): {uniform_count}")
        print(f"Scattered motion (UI changes): {scattered_count}")
        
        # Detect dominant directions
        directions = [p['direction'] for p in motion_patterns if p['direction']]
        if directions:
            from collections import Counter
            dir_counts = Counter(directions)
            print(f"\nDominant directions:")
            for direction, count in dir_counts.most_common(3):
                print(f"  - {direction}: {count} frames")
        
        # Analyze motion magnitude
        avg_magnitude = np.mean([p['magnitude'] for p in motion_patterns])
        max_magnitude = np.max([p['magnitude'] for p in motion_patterns])
        print(f"\nMotion intensity:")
        print(f"  Average: {avg_magnitude:.2f} pixels/frame")
        print(f"  Maximum: {max_magnitude:.2f} pixels/frame")
        
        # Classify overall activity
        if avg_magnitude < 2:
            print("\nOverall: LOW ACTIVITY (subtle movements)")
        elif avg_magnitude < 10:
            print("\nOverall: MODERATE ACTIVITY (normal scrolling/interactions)")
        else:
            print("\nOverall: HIGH ACTIVITY (rapid scrolling/transitions)")
    else:
        print("No significant motion detected")
    
    return motion_patterns

if __name__ == "__main__":
    # Test with the screen recording
    video_path = "ScreenRecording_08-27-2025 16-38-43_1.MP4"
    
    patterns = analyze_optical_flow_lucas_kanade(video_path, max_seconds=30)
    
    print("\n💡 Lucas-Kanade is good for:")
    print("  - Tracking specific UI elements")
    print("  - Detecting when elements appear/disappear")
    print("  - Distinguishing uniform motion (scrolling) from scattered changes")
    print("  - Following moving objects or regions")