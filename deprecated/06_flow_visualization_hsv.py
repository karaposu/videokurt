"""
Exploration 6: Optical Flow Visualization with HSV
This converts optical flow vectors to HSV color representation where:
- Hue represents direction of motion (0-360 degrees mapped to colors)
- Saturation represents confidence/consistency 
- Value represents magnitude/speed of motion
Creates a beautiful, intuitive visualization of motion patterns.


python -m explorations.06_flow_visualization_hsv




"""

import cv2
import numpy as np

def flow_to_hsv(flow):
    """Convert optical flow to HSV color representation."""
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    
    # Calculate magnitude and angle
    mag, ang = cv2.cartToPolar(fx, fy)
    
    # Create HSV image
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Better color mapping for common motions:
    # Right = Red (0°), Down = Green (90°), Left = Cyan (180°), Up = Blue/Magenta (270°)
    # This makes vertical scrolling more visually distinct
    hsv[:,:,0] = (ang * 180 / np.pi / 2 + 90) % 180  # Shift angle for better colors
    
    # Saturation based on magnitude (more motion = more saturated)
    # This helps distinguish between slow and fast motion
    sat_normalized = np.minimum(mag * 20, 255).astype(np.uint8)
    hsv[:,:,1] = sat_normalized
    
    # Value = magnitude (normalized to 0-255)
    # Use higher sensitivity for better visibility
    normalized_mag = np.minimum(mag * 15, 255).astype(np.uint8)
    hsv[:,:,2] = normalized_mag
    
    # Convert HSV to BGR for display
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr, mag, ang

def create_flow_legend():
    """Create a color wheel legend showing motion directions."""
    size = 120
    center = size // 2
    legend = np.zeros((size, size, 3), dtype=np.uint8)
    
    for y in range(size):
        for x in range(size):
            dx = x - center
            dy = y - center
            dist = np.sqrt(dx**2 + dy**2)
            
            if dist < center:
                # Calculate angle
                angle = np.arctan2(dy, dx)
                # Convert to degrees and map to HSV hue
                hue = ((angle + np.pi) * 180 / np.pi / 2).astype(np.uint8)
                # Saturation based on distance from center
                sat = min(255, int(dist * 255 / center))
                # Full brightness
                val = 255
                
                legend[y, x] = [hue, sat, val]
    
    legend = cv2.cvtColor(legend, cv2.COLOR_HSV2BGR)
    
    # Add labels
    cv2.putText(legend, "R", (size-15, center+3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
    cv2.putText(legend, "L", (5, center+3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
    cv2.putText(legend, "U", (center-5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
    cv2.putText(legend, "D", (center-5, size-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
    
    return legend

def analyze_flow_visualization(video_path, max_frames=None, max_seconds=None, save_output=False):
    """Analyze video with HSV flow visualization.
    
    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to process (optional)
        max_seconds: Maximum seconds of video to process (optional)
        save_output: Whether to save visualization video and images
    """
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return
    
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
    
    print(f"Analyzing {video_path}...")
    print(f"Video: {width}x{height} @ {fps:.1f} fps")
    print("-" * 50)
    
    # Read first frame
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Cannot read first frame")
        return
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    # Create flow legend
    legend = create_flow_legend()
    
    # Video writer for output if requested
    out = None
    if save_output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # Double width for side-by-side view
        out = cv2.VideoWriter('flow_visualization.mp4', fourcc, fps, (width*2, height))
    
    frame_count = 0
    motion_events = []
    
    # Create accumulator for total flow (not a true heatmap)
    flow_accumulator = np.zeros((height, width), dtype=np.float32)
    
    print("\nColor Legend:")
    print("  Red/Orange: Rightward motion")
    print("  Cyan/Blue: Leftward motion") 
    print("  Green/Yellow: Up/Down motion")
    print("  Brightness: Speed of motion")
    print("-" * 50)
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate dense optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        
        # Convert to HSV visualization
        flow_vis, mag, ang = flow_to_hsv(flow)
        
        # Update flow accumulator (simple sum of magnitudes)
        flow_accumulator += mag
        
        # Calculate motion statistics
        avg_magnitude = np.mean(mag)
        max_magnitude = np.max(mag)
        moving_pixels = np.sum(mag > 1.0)
        moving_percent = (moving_pixels / (width * height)) * 100
        
        # Detect dominant motion direction
        if avg_magnitude > 0.5:
            # Weight angles by magnitude to find dominant direction
            significant_mask = mag > 1.0
            if np.any(significant_mask):
                weighted_ang = ang[significant_mask]
                weighted_mag = mag[significant_mask]
                avg_angle = np.average(weighted_ang, weights=weighted_mag)
                
                # Convert angle to direction
                direction = ""
                angle_deg = avg_angle * 180 / np.pi
                if 45 <= angle_deg < 135:
                    direction = "DOWN"
                elif 135 <= angle_deg < 225:
                    direction = "LEFT"
                elif 225 <= angle_deg < 315:
                    direction = "UP"
                else:
                    direction = "RIGHT"
                
                # Detect motion patterns
                pattern = ""
                if moving_percent > 60:
                    pattern = "FULL_SCREEN_MOTION"
                elif moving_percent > 20:
                    pattern = "LARGE_AREA_MOTION"
                elif moving_percent > 5:
                    pattern = "MODERATE_MOTION"
                elif moving_percent > 1:
                    pattern = "LOCALIZED_MOTION"
                
                if avg_magnitude > 1.0 or pattern:
                    print(f"Frame {frame_count:3d}: Magnitude={avg_magnitude:.2f}, "
                          f"Max={max_magnitude:.1f}, Moving={moving_percent:.1f}%, "
                          f"Direction={direction}, Pattern={pattern}")
                    
                    motion_events.append({
                        'frame': frame_count,
                        'avg_magnitude': avg_magnitude,
                        'max_magnitude': max_magnitude,
                        'moving_percent': moving_percent,
                        'direction': direction,
                        'pattern': pattern
                    })
        
        # Save visualization if requested
        if out:
            # Create side-by-side view for better understanding
            # Left: Original frame, Right: Flow visualization
            side_by_side = np.zeros((height, width*2, 3), dtype=np.uint8)
            side_by_side[:, :width] = frame
            side_by_side[:, width:] = flow_vis
            
            # Add legend to flow side
            legend_small = cv2.resize(legend, (80, 80))
            side_by_side[10:90, width*2-90:width*2-10] = legend_small
            
            # Add labels
            cv2.putText(side_by_side, "Original", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(side_by_side, "Optical Flow", (width+10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Add motion stats on flow side
            if avg_magnitude > 0.5:
                cv2.putText(side_by_side, f"Mag: {avg_magnitude:.1f}", 
                           (width+10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(side_by_side, f"Moving: {moving_percent:.0f}%", 
                           (width+10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            out.write(side_by_side)
        
        prev_gray = gray
        frame_count += 1
    
    cap.release()
    if out:
        out.release()
    
    # Analyze results
    print("\n" + "=" * 50)
    print("FLOW VISUALIZATION ANALYSIS SUMMARY")
    print("=" * 50)
    
    if motion_events:
        # Direction statistics
        from collections import Counter
        directions = Counter([e['direction'] for e in motion_events])
        print("\nDominant motion directions:")
        for direction, count in directions.most_common():
            print(f"  - {direction}: {count} frames")
        
        # Pattern statistics
        patterns = Counter([e['pattern'] for e in motion_events])
        print("\nMotion patterns detected:")
        for pattern, count in patterns.most_common():
            print(f"  - {pattern}: {count} frames")
        
        # Overall statistics
        avg_moving = np.mean([e['moving_percent'] for e in motion_events])
        max_moving = np.max([e['moving_percent'] for e in motion_events])
        print(f"\nMotion coverage:")
        print(f"  Average: {avg_moving:.1f}% of frame")
        print(f"  Maximum: {max_moving:.1f}% of frame")
        
        # Create and save flow accumulation visualization (not a true heatmap)
        flow_avg = flow_accumulator / frame_count  # Average flow per pixel
        
        # Use percentile-based normalization for better contrast
        # This ensures we use the full color range regardless of motion scale
        vmin = np.percentile(flow_avg[flow_avg > 0], 5) if np.any(flow_avg > 0) else 0
        vmax = np.percentile(flow_avg[flow_avg > 0], 95) if np.any(flow_avg > 0) else 1
        
        # Normalize to 0-255 range using percentiles
        flow_normalized = np.zeros_like(flow_avg, dtype=np.uint8)
        mask = flow_avg > vmin
        flow_normalized[mask] = np.clip(
            ((flow_avg[mask] - vmin) / (vmax - vmin) * 255), 0, 255
        ).astype(np.uint8)
        
        # Apply different colormaps and save multiple versions
        flow_jet = cv2.applyColorMap(flow_normalized, cv2.COLORMAP_JET)
        flow_hot = cv2.applyColorMap(flow_normalized, cv2.COLORMAP_HOT)
        flow_turbo = cv2.applyColorMap(flow_normalized, cv2.COLORMAP_TURBO)
        
        # Create a custom visualization with clear zones
        flow_zones = np.zeros((height, width, 3), dtype=np.uint8)
        # Define clear activity levels
        no_motion = flow_avg == 0
        low_motion = (flow_avg > 0) & (flow_avg < np.percentile(flow_avg[flow_avg > 0], 33))
        med_motion = (flow_avg >= np.percentile(flow_avg[flow_avg > 0], 33)) & \
                     (flow_avg < np.percentile(flow_avg[flow_avg > 0], 66))
        high_motion = flow_avg >= np.percentile(flow_avg[flow_avg > 0], 66)
        
        # Color code the zones
        flow_zones[no_motion] = [0, 0, 0]      # Black for no motion
        flow_zones[low_motion] = [128, 0, 0]   # Dark blue for low
        flow_zones[med_motion] = [0, 128, 128] # Yellow for medium  
        flow_zones[high_motion] = [0, 0, 255]  # Red for high
        
        # Add contours to highlight boundaries
        contour_overlay = flow_jet.copy()
        # Find contours at different activity levels
        for threshold in [30, 60, 90]:
            thresh_mask = (flow_normalized > threshold * 255 / 100).astype(np.uint8)
            contours, _ = cv2.findContours(thresh_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(contour_overlay, contours, -1, (255, 255, 255), 2)
        
        # Find flow hotspots (areas with most optical flow)
        threshold = np.percentile(flow_avg, 95)
        hotspots = flow_avg > threshold
        num_hotspots = cv2.connectedComponents(hotspots.astype(np.uint8))[0] - 1
        
        print(f"\nFlow accumulation analysis:")
        print(f"  Identified {num_hotspots} high-flow regions")
        print(f"  Highest flow average: {np.max(flow_avg):.2f} pixels/frame")
        
        if save_output:
            # Save multiple versions for comparison
            cv2.imwrite('flow_accumulation_jet.png', flow_jet)
            cv2.imwrite('flow_accumulation_hot.png', flow_hot) 
            cv2.imwrite('flow_accumulation_turbo.png', flow_turbo)
            cv2.imwrite('flow_accumulation_zones.png', flow_zones)
            cv2.imwrite('flow_accumulation_contours.png', contour_overlay)
            
            # Create a comparison grid
            grid = np.zeros((height*2, width*2, 3), dtype=np.uint8)
            grid[:height, :width] = flow_jet
            grid[:height, width:] = flow_hot
            grid[height:, :width] = flow_zones
            grid[height:, width:] = contour_overlay
            
            # Add labels
            label_color = (255, 255, 255)
            cv2.putText(grid, "JET", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, label_color, 2)
            cv2.putText(grid, "HOT", (width+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, label_color, 2)
            cv2.putText(grid, "ZONES", (10, height+30), cv2.FONT_HERSHEY_SIMPLEX, 1, label_color, 2)
            cv2.putText(grid, "CONTOURS", (width+10, height+30), cv2.FONT_HERSHEY_SIMPLEX, 1, label_color, 2)
            
            cv2.imwrite('flow_accumulation_comparison.png', grid)
            
            print("\n  Saved flow accumulation visualizations:")
            print("    - flow_accumulation_comparison.png (4 styles in grid)")
            print("    - flow_accumulation_zones.png (clear flow zones)")
            print("    - flow_accumulation_contours.png (with boundary lines)")
    
    if save_output and out:
        print(f"  Saved flow visualization video to 'flow_visualization.mp4'")
    
    return motion_events, flow_accumulator

if __name__ == "__main__":
    # Test with the screen recording
    video_path = "ScreenRecording_08-27-2025 16-38-43_1.MP4"
    
    events, flow_total = analyze_flow_visualization(video_path, max_seconds=30, save_output=True)
    
    print("\n💡 HSV Flow Visualization benefits:")
    print("  - Intuitive color coding for motion direction")
    print("  - Brightness shows motion speed")
    print("  - Easy to spot motion patterns at a glance")
    print("  - Can identify scrolling, panning, zooming patterns")
    print("\n📝 Saved: flow_visualization.mp4 and flow_accumulation_*.png files")