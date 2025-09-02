"""
Exploration 7: Motion Heatmap Generation
This creates cumulative motion heatmaps showing which areas of the video
have the most activity over time. Useful for identifying UI hotspots,
frequent interaction areas, and motion patterns.
"""

import cv2
import numpy as np
from collections import deque

def create_motion_heatmap(video_path, max_frames=None, max_seconds=None, decay_factor=0.98, method='flow'):
    """
    Create a motion heatmap from a video.
    
    Args:
        video_path: Path to video file
        max_frames: Maximum frames to process (None for all)
        max_seconds: Maximum seconds of video to process (optional)
        decay_factor: How quickly old motion fades (0-1, lower = faster fade)
        method: 'flow', 'diff', or 'combined' for detection method
    """
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return None
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame limit based on parameters
    if max_seconds is not None:
        max_frames = int(fps * max_seconds)
        print(f"Processing {max_seconds} seconds = {max_frames} frames at {fps:.1f} fps")
    elif max_frames is None:
        max_frames = total_frames
    
    max_frames = min(max_frames, total_frames)
    
    print(f"Creating motion heatmap for {video_path}...")
    print(f"Video: {width}x{height}, {total_frames} frames @ {fps:.1f} fps")
    print(f"Processing up to {max_frames} frames")
    print(f"Method: {method}, Decay factor: {decay_factor}")
    print("-" * 50)
    
    # Initialize heatmaps
    instant_heatmap = np.zeros((height, width), dtype=np.float32)
    cumulative_heatmap = np.zeros((height, width), dtype=np.float32)
    weighted_heatmap = np.zeros((height, width), dtype=np.float32)
    
    # For temporal analysis
    temporal_buffer = deque(maxlen=30)  # Keep last 30 frames of motion
    
    # Read first frame
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Cannot read first frame")
        return None
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    # Background subtractor for combined method
    if method == 'combined':
        bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
    
    frame_count = 0
    motion_regions = []
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate motion based on method
        if method == 'flow':
            # Optical flow method
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            mag, _ = cv2.cartToPolar(flow[...,0], flow[...,1])
            motion_mask = mag
            
        elif method == 'diff':
            # Frame difference method
            diff = cv2.absdiff(prev_gray, gray)
            motion_mask = diff.astype(np.float32) / 255.0
            
        elif method == 'combined':
            # Combined method: flow + background subtraction
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None,
                pyr_scale=0.5, levels=2, winsize=15,
                iterations=2, poly_n=5, poly_sigma=1.1, flags=0
            )
            mag, _ = cv2.cartToPolar(flow[...,0], flow[...,1])
            
            # Background subtraction
            fg_mask = bg_subtractor.apply(frame)
            fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)[1]
            fg_normalized = fg_mask.astype(np.float32) / 255.0
            
            # Combine both methods
            motion_mask = (mag * 0.7 + fg_normalized * 0.3)
        
        # Update heatmaps
        instant_heatmap = motion_mask
        cumulative_heatmap += motion_mask
        weighted_heatmap = weighted_heatmap * decay_factor + motion_mask
        
        # Add to temporal buffer
        temporal_buffer.append(motion_mask)
        
        # Analyze motion regions every 10 frames
        if frame_count % 10 == 0 and frame_count > 0:
            # Find regions with consistent motion
            if len(temporal_buffer) > 5:
                temporal_avg = np.mean(temporal_buffer, axis=0)
                temporal_std = np.std(temporal_buffer, axis=0)
                
                # High average + low std = consistent motion (e.g., animation)
                # High average + high std = sporadic motion (e.g., interaction)
                consistent_motion = (temporal_avg > 0.5) & (temporal_std < 0.3)
                sporadic_motion = (temporal_avg > 0.3) & (temporal_std > 0.3)
                
                # Find connected components
                consistent_labels = cv2.connectedComponents(consistent_motion.astype(np.uint8))[1]
                sporadic_labels = cv2.connectedComponents(sporadic_motion.astype(np.uint8))[1]
                
                num_consistent = np.max(consistent_labels)
                num_sporadic = np.max(sporadic_labels)
                
                if num_consistent > 0 or num_sporadic > 0:
                    motion_regions.append({
                        'frame': frame_count,
                        'consistent_regions': num_consistent,
                        'sporadic_regions': num_sporadic,
                        'avg_motion': np.mean(temporal_avg),
                        'max_motion': np.max(temporal_avg)
                    })
                    
                    print(f"Frame {frame_count:4d}: "
                          f"Consistent regions: {num_consistent}, "
                          f"Sporadic regions: {num_sporadic}, "
                          f"Avg motion: {np.mean(temporal_avg):.2f}")
        
        prev_gray = gray
        frame_count += 1
        
        # Progress indicator
        if frame_count % 30 == 0:
            progress = (frame_count / max_frames) * 100
            print(f"  Progress: {progress:.1f}%")
    
    cap.release()
    
    # Normalize heatmaps
    cumulative_normalized = cumulative_heatmap / frame_count if frame_count > 0 else cumulative_heatmap
    
    # Create visualizations
    print("\n" + "=" * 50)
    print("MOTION HEATMAP ANALYSIS")
    print("=" * 50)
    
    # 1. Cumulative heatmap (average motion per pixel)
    cumulative_vis = np.uint8(np.clip(cumulative_normalized * 255, 0, 255))
    cumulative_colored = cv2.applyColorMap(cumulative_vis, cv2.COLORMAP_JET)
    
    # 2. Weighted heatmap (recent motion weighted more)
    weighted_max = np.max(weighted_heatmap)
    weighted_normalized = weighted_heatmap / weighted_max if weighted_max > 0 else weighted_heatmap
    weighted_vis = np.uint8(np.clip(weighted_normalized * 255, 0, 255))
    weighted_colored = cv2.applyColorMap(weighted_vis, cv2.COLORMAP_HOT)
    
    # 3. Activity zones (threshold-based regions)
    zones = {
        'high_activity': cumulative_normalized > np.percentile(cumulative_normalized, 90),
        'medium_activity': (cumulative_normalized > np.percentile(cumulative_normalized, 70)) & 
                          (cumulative_normalized <= np.percentile(cumulative_normalized, 90)),
        'low_activity': (cumulative_normalized > np.percentile(cumulative_normalized, 30)) & 
                       (cumulative_normalized <= np.percentile(cumulative_normalized, 70))
    }
    
    # Create zone visualization
    zone_vis = np.zeros((height, width, 3), dtype=np.uint8)
    zone_vis[zones['high_activity']] = [0, 0, 255]  # Red for high
    zone_vis[zones['medium_activity']] = [0, 255, 255]  # Yellow for medium
    zone_vis[zones['low_activity']] = [0, 255, 0]  # Green for low
    
    # Analyze zones
    print("\nActivity zone analysis:")
    for zone_name, zone_mask in zones.items():
        zone_pixels = np.sum(zone_mask)
        zone_percent = (zone_pixels / (width * height)) * 100
        print(f"  {zone_name}: {zone_percent:.1f}% of frame")
        
        # Find connected components in each zone
        if zone_pixels > 0:
            num_components = cv2.connectedComponents(zone_mask.astype(np.uint8))[0] - 1
            print(f"    - {num_components} distinct regions")
    
    # Find motion hotspots
    print("\nMotion hotspots (top 5):")
    hotspot_threshold = np.percentile(cumulative_normalized, 95)
    hotspot_mask = cumulative_normalized > hotspot_threshold
    
    # Find contours of hotspots
    contours, _ = cv2.findContours(hotspot_mask.astype(np.uint8), 
                                  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    hotspots = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if area > 100:  # Minimum area threshold
            # Calculate average intensity in this region
            region_intensity = np.mean(cumulative_normalized[y:y+h, x:x+w])
            hotspots.append({
                'bbox': (x, y, w, h),
                'area': area,
                'intensity': region_intensity,
                'center': (x + w//2, y + h//2)
            })
    
    # Sort by intensity and show top 5
    hotspots.sort(key=lambda h: h['intensity'], reverse=True)
    for i, hotspot in enumerate(hotspots[:5]):
        bbox = hotspot['bbox']
        print(f"  {i+1}. Region at ({bbox[0]}, {bbox[1]}), "
              f"size {bbox[2]}x{bbox[3]}, "
              f"intensity: {hotspot['intensity']:.2f}")
        
        # Draw on visualization
        cv2.rectangle(cumulative_colored, 
                     (bbox[0], bbox[1]), 
                     (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                     (255, 255, 255), 2)
        cv2.putText(cumulative_colored, f"#{i+1}", 
                   (bbox[0], bbox[1]-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Temporal analysis summary
    if motion_regions:
        avg_consistent = np.mean([r['consistent_regions'] for r in motion_regions])
        avg_sporadic = np.mean([r['sporadic_regions'] for r in motion_regions])
        print(f"\nTemporal motion patterns:")
        print(f"  Average consistent regions: {avg_consistent:.1f}")
        print(f"  Average sporadic regions: {avg_sporadic:.1f}")
        
        # Classify overall motion pattern
        if avg_consistent > avg_sporadic:
            print("  Pattern: ANIMATION_HEAVY (consistent repeated motion)")
        elif avg_sporadic > avg_consistent:
            print("  Pattern: INTERACTION_HEAVY (sporadic user-driven motion)")
        else:
            print("  Pattern: MIXED (both animations and interactions)")
    
    # Save visualizations
    cv2.imwrite('heatmap_cumulative.png', cumulative_colored)
    cv2.imwrite('heatmap_weighted.png', weighted_colored)
    cv2.imwrite('heatmap_zones.png', zone_vis)
    
    print("\n📊 Saved heatmap visualizations:")
    print("  - heatmap_cumulative.png: Average motion over time")
    print("  - heatmap_weighted.png: Recent motion weighted more heavily")
    print("  - heatmap_zones.png: Activity zones (red=high, yellow=medium, green=low)")
    
    return {
        'cumulative': cumulative_normalized,
        'weighted': weighted_normalized,
        'zones': zones,
        'hotspots': hotspots,
        'motion_regions': motion_regions
    }

if __name__ == "__main__":
    # Test with the screen recording
    video_path = "ScreenRecording_08-26-2025 16-28-41_1.MP4"
    
    method = 'flow'  # Default method
    print(f"Using {method} method for motion detection\n")
    
    # Create heatmap
    heatmap_data = create_motion_heatmap(
        video_path, 
        max_seconds=30, 
        decay_factor=0.98,
        method=method
    )
    
    if heatmap_data:
        print("\n💡 Motion Heatmap Applications:")
        print("  - Identify UI interaction hotspots")
        print("  - Find regions with animations/videos")
        print("  - Detect scrollable areas")
        print("  - Analyze user attention patterns")
        print("  - Optimize UI element placement")
        print("\n📝 Usage: python 07_motion_heatmap.py [video_path] [method]")
        print("  Methods: 'flow' (optical flow), 'diff' (frame difference), 'combined'")