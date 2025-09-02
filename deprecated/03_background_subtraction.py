"""
Exploration 3: Background Subtraction (MOG2 and KNN)
These algorithms learn what's "background" and detect "foreground" changes.
Useful for detecting moving regions, animations, and UI changes.


python -m explorations.03_background_subtraction
"""

import cv2
import numpy as np

def analyze_background_subtraction(video_path, max_frames=None, max_seconds=None):
    """Analyze video using background subtraction algorithms.
    
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
    
    # Create background subtractors with reduced history to save memory
    mog2 = cv2.createBackgroundSubtractorMOG2(
        detectShadows=True,  # Detect shadows
        varThreshold=16,     # Variance threshold
        history=120          # Reduced from 500 to save memory
    )
    
    knn = cv2.createBackgroundSubtractorKNN(
        detectShadows=True,  # Detect shadows
        dist2Threshold=400.0, # Distance threshold
        history=120          # Reduced from 500 to save memory
    )
    
    frame_count = 0
    change_regions = []
    
    print(f"Analyzing {video_path}...")
    print("Learning background for first 30 frames...")
    print("-" * 50)
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Apply background subtraction
        mask_mog2 = mog2.apply(frame)
        mask_knn = knn.apply(frame)
        
        # Remove shadows (shadows are marked as 127, foreground as 255)
        mask_mog2_clean = cv2.threshold(mask_mog2, 200, 255, cv2.THRESH_BINARY)[1]
        mask_knn_clean = cv2.threshold(mask_knn, 200, 255, cv2.THRESH_BINARY)[1]
        
        # Apply morphological operations to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_mog2_clean = cv2.morphologyEx(mask_mog2_clean, cv2.MORPH_OPEN, kernel)
        mask_knn_clean = cv2.morphologyEx(mask_knn_clean, cv2.MORPH_OPEN, kernel)
        
        # Free memory from intermediate masks
        del mask_mog2
        del mask_knn
        
        # Find contours (regions of change)
        contours_mog2, _ = cv2.findContours(mask_mog2_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # contours_knn not used to save processing - just counting pixels is enough
        # contours_knn, _ = cv2.findContours(mask_knn_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Analyze change regions
        if frame_count > 30:  # After learning phase
            # Calculate percentage of frame that changed
            h, w = frame.shape[:2]
            total_pixels = h * w
            changed_pixels_mog2 = cv2.countNonZero(mask_mog2_clean)
            changed_pixels_knn = cv2.countNonZero(mask_knn_clean)
            
            change_percent_mog2 = (changed_pixels_mog2 / total_pixels) * 100
            change_percent_knn = (changed_pixels_knn / total_pixels) * 100
            
            # Find significant contours (LIMIT to prevent memory issues)
            significant_contours = []
            # Sort contours by area and take only top 10 largest
            sorted_contours = sorted(contours_mog2, key=cv2.contourArea, reverse=True)
            for contour in sorted_contours[:10]:  # LIMIT TO 10 CONTOURS
                area = cv2.contourArea(contour)
                if area > 100:  # Minimum area threshold
                    x, y, w, h = cv2.boundingRect(contour)
                    significant_contours.append({
                        'area': area,
                        'bbox': (x, y, w, h),
                        'area_percent': (area / total_pixels) * 100
                    })
            
            if change_percent_mog2 > 1.0 or len(significant_contours) > 0:
                # Classify the type of change
                change_type = ""
                if change_percent_mog2 > 80:
                    change_type = "FULL_CHANGE"
                elif change_percent_mog2 > 20:
                    change_type = "PARTIAL_CHANGE"
                elif change_percent_mog2 > 5:
                    change_type = "LOCALIZED_CHANGE"
                else:
                    change_type = "MINI_CHANGE"
                
                # Check if changes are in a specific region (animation)
                if len(significant_contours) == 1 and change_percent_mog2 < 30:
                    bbox = significant_contours[0]['bbox']
                    if frame_count > 0 and change_regions:
                        # Check if this region was active before
                        last_regions = change_regions[-1]['regions']
                        for last_region in last_regions:
                            last_bbox = last_region['bbox']
                            # Check overlap
                            if (abs(bbox[0] - last_bbox[0]) < 50 and 
                                abs(bbox[1] - last_bbox[1]) < 50):
                                change_type = "REGION_ANIMATION"
                                break
                
                # Calculate timestamp
                timestamp = frame_count * time_per_frame
                mins = int(timestamp // 60)
                secs = timestamp % 60
                
                print(f"Frame {frame_count:3d} ({mins:02d}:{secs:05.2f}): Change={change_percent_mog2:.1f}% (MOG2), "
                      f"{change_percent_knn:.1f}% (KNN), Regions={len(significant_contours)}, "
                      f"Type={change_type}")
                
                # Only store summary data, not full contour data to prevent memory issues
                change_regions.append({
                    'frame': frame_count,
                    'change_percent_mog2': change_percent_mog2,
                    'change_percent_knn': change_percent_knn,
                    'num_regions': len(significant_contours),
                    'regions': significant_contours[:3],  # Only keep top 3 regions
                    'change_type': change_type
                })
                
                # Limit history to prevent memory issues
                if len(change_regions) > 500:  # Keep only last 500 frames
                    change_regions.pop(0)
        
        frame_count += 1
    
    cap.release()
    
    # Analyze patterns
    print("\n" + "=" * 50)
    print("BACKGROUND SUBTRACTION ANALYSIS SUMMARY")
    print("=" * 50)
    
    if change_regions:
        # Statistics
        avg_change = np.mean([r['change_percent_mog2'] for r in change_regions])
        max_change = np.max([r['change_percent_mog2'] for r in change_regions])
        
        print(f"Total frames with changes: {len(change_regions)}/{frame_count}")
        print(f"Average change: {avg_change:.1f}%")
        print(f"Maximum change: {max_change:.1f}%")
        
        # Count change types
        from collections import Counter
        change_types = Counter([r['change_type'] for r in change_regions])
        print("\nChange patterns detected:")
        for change_type, count in change_types.most_common():
            print(f"  - {change_type}: {count} frames")
        
        # Detect animation regions
        animation_frames = [r for r in change_regions if r['change_type'] == 'REGION_ANIMATION']
        if animation_frames:
            print(f"\nAnimation detected in {len(animation_frames)} frames")
            # Find common animation bbox
            if animation_frames[0]['regions']:
                bbox = animation_frames[0]['regions'][0]['bbox']
                print(f"  Region: x={bbox[0]}, y={bbox[1]}, w={bbox[2]}, h={bbox[3]}")
        
        # Overall classification
        if avg_change < 5:
            print("\nOverall: MOSTLY STATIC with small changes")
        elif avg_change < 20:
            print("\nOverall: MODERATE ACTIVITY (UI interactions)")
        elif avg_change < 50:
            print("\nOverall: HIGH ACTIVITY (scrolling/transitions)")
        else:
            print("\nOverall: VERY HIGH ACTIVITY (full screen changes)")
    else:
        print("No significant changes detected after background learning")
    
    return change_regions

if __name__ == "__main__":
    # Test with the screen recording


    video_path = "ScreenRecording_08-27-2025 16-38-43_1.MP4"
    
    regions = analyze_background_subtraction(video_path, max_seconds=30)
    
    print("\n💡 Background Subtraction is good for:")
    print("  - Detecting which parts of screen are changing")
    print("  - Finding animated regions (GIFs, videos, spinners)")
    print("  - Measuring change magnitude (full/partial/localized)")
    print("  - Identifying static vs dynamic areas")
    print("\n📝 Note: MOG2 and KNN need time to learn the background")
    print("     First 20-30 frames are for learning")