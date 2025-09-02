"""
Exploration 4: Contour Detection and Tracking
This detects shapes and regions that are changing between frames.
Useful for finding UI elements, detecting click areas, and tracking moving regions.

python -m explorations.04_contour_detection

"""

import cv2
import numpy as np

def analyze_contour_changes(video_path, max_frames=None, max_seconds=None):
    """Analyze video by detecting and tracking contours of changing regions.
    
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
    
    # Read first frame
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Cannot read first frame")
        return
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    h, w = prev_gray.shape
    total_pixels = h * w
    
    frame_count = 0
    contour_events = []
    prev_contours = []
    
    print(f"Analyzing {video_path}...")
    print(f"Frame size: {w}x{h}")
    print("-" * 50)
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate frame difference
        diff = cv2.absdiff(prev_gray, gray)
        
        # Apply threshold to get binary mask of changes
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)  # Close gaps
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)   # Remove noise
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and analyze significant contours
        significant_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Minimum area threshold
                x, y, w_box, h_box = cv2.boundingRect(contour)
                
                # Calculate contour properties
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                aspect_ratio = w_box / h_box if h_box > 0 else 0
                extent = area / (w_box * h_box) if (w_box * h_box) > 0 else 0
                
                significant_contours.append({
                    'area': area,
                    'bbox': (x, y, w_box, h_box),
                    'center': (x + w_box//2, y + h_box//2),
                    'area_percent': (area / total_pixels) * 100,
                    'circularity': circularity,
                    'aspect_ratio': aspect_ratio,
                    'extent': extent
                })
        
        # Analyze contour patterns
        if significant_contours:
            total_change_area = sum(c['area'] for c in significant_contours)
            change_percent = (total_change_area / total_pixels) * 100
            
            # Classify change pattern
            pattern = ""
            if len(significant_contours) == 1:
                c = significant_contours[0]
                if c['area_percent'] > 50:
                    pattern = "FULL_SCREEN_MOTION"
                elif c['area_percent'] > 10:
                    pattern = "LARGE_REGION_CHANGE"
                elif c['circularity'] > 0.7:
                    pattern = "CIRCULAR_ELEMENT" # Possible button
                elif c['aspect_ratio'] > 5 or c['aspect_ratio'] < 0.2:
                    pattern = "LINEAR_ELEMENT"  # Possible scrollbar or line
                else:
                    pattern = "LOCALIZED_ELEMENT"
            elif len(significant_contours) > 10:
                pattern = "SCATTERED_CHANGES"  # Many small changes
            elif len(significant_contours) > 1:
                # Check if contours are aligned (possible list/grid)
                centers_y = [c['center'][1] for c in significant_contours]
                centers_x = [c['center'][0] for c in significant_contours]
                std_y = np.std(centers_y)
                std_x = np.std(centers_x)
                
                if std_y < 50:  # Horizontally aligned
                    pattern = "HORIZONTAL_ELEMENTS"
                elif std_x < 50:  # Vertically aligned
                    pattern = "VERTICAL_ELEMENTS"
                else:
                    pattern = "MULTIPLE_REGIONS"
            
            if change_percent > 1.0 or pattern:
                print(f"Frame {frame_count:3d}: Contours={len(significant_contours)}, "
                      f"Change={change_percent:.1f}%, Pattern={pattern}")
                
                # Track movement of contours
                movement = ""
                if prev_contours and significant_contours:
                    # Compare center positions with previous frame
                    curr_center_y = np.mean([c['center'][1] for c in significant_contours])
                    prev_center_y = np.mean([c['center'][1] for c in prev_contours])
                    curr_center_x = np.mean([c['center'][0] for c in significant_contours])
                    prev_center_x = np.mean([c['center'][0] for c in prev_contours])
                    
                    dy = curr_center_y - prev_center_y
                    dx = curr_center_x - prev_center_x
                    
                    if abs(dy) > 5:
                        movement = f"VERTICAL_MOVE({'DOWN' if dy > 0 else 'UP'})"
                    elif abs(dx) > 5:
                        movement = f"HORIZONTAL_MOVE({'RIGHT' if dx > 0 else 'LEFT'})"
                    
                    if movement:
                        print(f"         Movement: {movement} (dy={dy:.1f}, dx={dx:.1f})")
                
                contour_events.append({
                    'frame': frame_count,
                    'num_contours': len(significant_contours),
                    'change_percent': change_percent,
                    'pattern': pattern,
                    'movement': movement,
                    'contours': significant_contours
                })
        
        prev_contours = significant_contours
        prev_gray = gray
        frame_count += 1
    
    cap.release()
    
    # Analyze patterns
    print("\n" + "=" * 50)
    print("CONTOUR ANALYSIS SUMMARY")
    print("=" * 50)
    
    if contour_events:
        print(f"Total frames with contours: {len(contour_events)}/{frame_count}")
        
        # Count patterns
        from collections import Counter
        patterns = Counter([e['pattern'] for e in contour_events if e['pattern']])
        print("\nDetected patterns:")
        for pattern, count in patterns.most_common():
            print(f"  - {pattern}: {count} frames")
        
        # Count movements
        movements = [e['movement'] for e in contour_events if e['movement']]
        if movements:
            movement_counts = Counter(movements)
            print("\nDetected movements:")
            for movement, count in movement_counts.most_common():
                print(f"  - {movement}: {count} frames")
        
        # Find persistent regions (possible animations)
        persistent_regions = []
        for event in contour_events:
            if event['num_contours'] == 1 and event['change_percent'] < 10:
                bbox = event['contours'][0]['bbox']
                # Check if this region persists
                found = False
                for region in persistent_regions:
                    if (abs(region['bbox'][0] - bbox[0]) < 20 and 
                        abs(region['bbox'][1] - bbox[1]) < 20):
                        region['count'] += 1
                        found = True
                        break
                if not found:
                    persistent_regions.append({'bbox': bbox, 'count': 1})
        
        # Report persistent regions
        persistent_regions = [r for r in persistent_regions if r['count'] > 5]
        if persistent_regions:
            print(f"\nPersistent changing regions (possible animations):")
            for region in persistent_regions:
                bbox = region['bbox']
                print(f"  - Region at ({bbox[0]}, {bbox[1]}), size {bbox[2]}x{bbox[3]}, "
                      f"active for {region['count']} frames")
    else:
        print("No significant contours detected")
    
    return contour_events

if __name__ == "__main__":
    # Test with the screen recording
    video_path = "ScreenRecording_08-27-2025 16-38-43_1.MP4"

    
    events = analyze_contour_changes(video_path, max_seconds=30)
    
    print("\n💡 Contour Detection is good for:")
    print("  - Finding UI elements that are changing")
    print("  - Detecting click/tap locations")
    print("  - Tracking moving regions")
    print("  - Identifying shapes (buttons, cards, lists)")
    print("  - Detecting aligned elements (grids, lists)")