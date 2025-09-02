"""
Exploration 5: Advanced Frame Differencing
This implements various frame differencing techniques for detecting changes.
Includes running average, accumulated differences, and triple frame differencing.


python -m explorations.05_advanced_frame_differencing
"""

import cv2
import numpy as np

def analyze_advanced_frame_differencing(video_path, max_frames=None, max_seconds=None):
    """Analyze video using advanced frame differencing techniques.
    
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
    
    # Calculate frame limit based on parameters
    if max_seconds is not None:
        max_frames = int(fps * max_seconds)
        print(f"Processing {max_seconds} seconds = {max_frames} frames at {fps:.1f} fps")
    elif max_frames is None:
        max_frames = total_frames
    
    max_frames = min(max_frames, total_frames)
    
    # Read first frames
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()
    if not ret:
        print("Error: Cannot read initial frames")
        return
    
    # Convert to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    h, w = gray1.shape
    total_pixels = h * w
    
    # Initialize running average for background
    running_avg = np.float32(gray1)
    
    # Initialize accumulated difference
    accumulated_diff = np.zeros((h, w), dtype=np.float32)
    
    frame_count = 2
    change_events = []
    
    print(f"Analyzing {video_path}...")
    print(f"Frame size: {w}x{h}")
    print("-" * 50)
    
    while frame_count < max_frames:
        ret, frame3 = cap.read()
        if not ret:
            break
        
        gray3 = cv2.cvtColor(frame3, cv2.COLOR_BGR2GRAY)
        
        # 1. Simple frame differencing (current vs previous)
        diff_simple = cv2.absdiff(gray2, gray3)
        
        # 2. Triple frame differencing (reduces noise)
        diff1 = cv2.absdiff(gray1, gray2)
        diff2 = cv2.absdiff(gray2, gray3)
        triple_diff = cv2.bitwise_and(diff1, diff2)
        
        # 3. Running average background subtraction
        cv2.accumulateWeighted(gray3, running_avg, 0.02)  # Learning rate 0.02
        background = np.uint8(running_avg)
        diff_background = cv2.absdiff(gray3, background)
        
        # 4. Accumulated differences (motion history)
        accumulated_diff = accumulated_diff * 0.95  # Decay factor
        accumulated_diff += diff_simple.astype(np.float32) / 255.0
        accumulated_norm = np.uint8(np.clip(accumulated_diff * 255, 0, 255))
        
        # Apply thresholds to get binary masks
        _, mask_simple = cv2.threshold(diff_simple, 25, 255, cv2.THRESH_BINARY)
        _, mask_triple = cv2.threshold(triple_diff, 20, 255, cv2.THRESH_BINARY)
        _, mask_background = cv2.threshold(diff_background, 30, 255, cv2.THRESH_BINARY)
        _, mask_accumulated = cv2.threshold(accumulated_norm, 50, 255, cv2.THRESH_BINARY)
        
        # Calculate change percentages
        change_simple = (cv2.countNonZero(mask_simple) / total_pixels) * 100
        change_triple = (cv2.countNonZero(mask_triple) / total_pixels) * 100
        change_background = (cv2.countNonZero(mask_background) / total_pixels) * 100
        change_accumulated = (cv2.countNonZero(mask_accumulated) / total_pixels) * 100
        
        # Detect patterns based on different methods
        pattern = ""
        
        # If accumulated shows high activity but simple is low = repeated small changes
        if change_accumulated > 10 and change_simple < 5:
            pattern = "FLICKER/ANIMATION"
        
        # If triple diff is high = consistent motion
        elif change_triple > 5:
            pattern = "CONSISTENT_MOTION"
        
        # If background diff is high but simple is moderate = new content appearing
        elif change_background > change_simple * 1.5 and change_background > 10:
            pattern = "CONTENT_APPEARING"
        
        # If simple is very high = sudden change
        elif change_simple > 50:
            pattern = "SUDDEN_CHANGE"
        elif change_simple > 20:
            pattern = "SIGNIFICANT_CHANGE"
        elif change_simple > 5:
            pattern = "MODERATE_CHANGE"
        elif change_simple > 1:
            pattern = "MINOR_CHANGE"
        
        # Detect fade/gradual changes
        if frame_count > 5:
            # Check if change is uniform across frame (fade)
            diff_std = np.std(diff_simple)
            if diff_std < 10 and change_simple > 1:
                pattern += "_UNIFORM_FADE"
        
        if change_simple > 1.0 or pattern:
            print(f"Frame {frame_count:3d}: Simple={change_simple:.1f}%, "
                  f"Triple={change_triple:.1f}%, BG={change_background:.1f}%, "
                  f"Accum={change_accumulated:.1f}% | {pattern}")
            
            change_events.append({
                'frame': frame_count,
                'simple': change_simple,
                'triple': change_triple,
                'background': change_background,
                'accumulated': change_accumulated,
                'pattern': pattern
            })
        
        # Shift frames for next iteration
        gray1 = gray2
        gray2 = gray3
        frame_count += 1
    
    cap.release()
    
    # Analyze patterns
    print("\n" + "=" * 50)
    print("ADVANCED FRAME DIFFERENCING SUMMARY")
    print("=" * 50)
    
    if change_events:
        # Statistics
        avg_simple = np.mean([e['simple'] for e in change_events])
        avg_triple = np.mean([e['triple'] for e in change_events])
        avg_background = np.mean([e['background'] for e in change_events])
        
        print(f"Average change detection:")
        print(f"  Simple diff: {avg_simple:.1f}%")
        print(f"  Triple diff: {avg_triple:.1f}% (noise reduced)")
        print(f"  Background diff: {avg_background:.1f}% (vs running average)")
        
        # Count patterns
        from collections import Counter
        patterns = Counter([e['pattern'] for e in change_events if e['pattern']])
        print("\nDetected patterns:")
        for pattern, count in patterns.most_common():
            print(f"  - {pattern}: {count} frames")
        
        # Detect sequences
        print("\nSequence analysis:")
        
        # Find longest consistent motion
        consistent_frames = [e for e in change_events if 'CONSISTENT_MOTION' in e['pattern']]
        if len(consistent_frames) > 5:
            print(f"  Consistent motion detected in {len(consistent_frames)} frames")
        
        # Find flicker/animation regions
        flicker_frames = [e for e in change_events if 'FLICKER' in e['pattern']]
        if flicker_frames:
            print(f"  Flicker/animation detected in {len(flicker_frames)} frames")
        
        # Find fades
        fade_frames = [e for e in change_events if 'FADE' in e['pattern']]
        if fade_frames:
            print(f"  Uniform fading detected in {len(fade_frames)} frames")
    
    return change_events

if __name__ == "__main__":
    # Test with the screen recording
    video_path = "ScreenRecording_08-27-2025 16-38-43_1.MP4"
    
    events = analyze_advanced_frame_differencing(video_path, max_seconds=30)
    
    print("\n💡 Advanced Frame Differencing techniques:")
    print("  - Simple diff: Basic frame-to-frame changes")
    print("  - Triple diff: Reduces noise, finds consistent motion")
    print("  - Background diff: Detects new content against running average")
    print("  - Accumulated diff: Shows motion history, finds repeated changes")
    print("\n📝 Useful for detecting:")
    print("  - Fades and gradual changes")
    print("  - Flicker and repeated animations")
    print("  - Content appearing/disappearing")
    print("  - Distinguishing sudden vs gradual changes")