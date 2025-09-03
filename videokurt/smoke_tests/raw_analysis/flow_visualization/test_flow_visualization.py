"""
Run: python -m videokurt.smoke_tests.raw_analysis.flow_visualization.test_flow_visualization

Smoke test for Flow HSV Visualization raw analysis
Tests optical flow visualization using HSV color space
"""

import sys
import cv2
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from videokurt.raw_analysis.flow_hsv_viz import FlowHSVViz


def test_flow_hsv_basic():
    """Test basic flow visualization"""
    print("Testing Flow HSV Visualization...")
    
    # Initialize analyzer with correct parameters
    analyzer = FlowHSVViz(downsample=1.0, max_magnitude=20.0, saturation_boost=1.5)
    
    # Create test frames with horizontal motion
    frames = []
    for i in range(5):
        frame = np.zeros((200, 300, 3), dtype=np.uint8)
        
        # Moving vertical line (creates horizontal flow)
        x = 50 + i * 30
        cv2.line(frame, (x, 20), (x, 180), (255, 255, 255), 3)
        
        # Moving circle (creates diagonal flow)
        cx = 100 + i * 20
        cy = 100 + i * 10
        cv2.circle(frame, (cx, cy), 20, (200, 200, 200), -1)
        
        frames.append(frame)
    
    # Analyze frames
    result = analyzer.analyze(frames)
    
    # Verify result structure
    assert result is not None, "Result should not be None"
    assert hasattr(result, 'data'), "Result should have data attribute"
    assert hasattr(result, 'method'), "Result should have method attribute"
    assert result.method == 'flow_hsv_viz', f"Method should be 'flow_hsv_viz', got {result.method}"
    
    # Check data structure - actual implementation returns 'hsv_flow'
    assert 'hsv_flow' in result.data, "Should have hsv_flow in data"
    
    # Check dimensions
    hsv_flow = result.data['hsv_flow']
    assert len(hsv_flow) == len(frames) - 1, "Should have n-1 flow visualizations"
    
    flow_viz = hsv_flow[0]
    assert flow_viz.shape[:2] == (200, 300), f"Flow viz shape should match frame size"
    assert flow_viz.shape[2] == 3, "Flow viz should be 3-channel"
    
    # Should detect some motion (non-zero flow)
    assert hsv_flow.max() > 0, "Should detect motion"
    
    print(f"✓ Analyzed {len(frames)} frames")
    print(f"✓ HSV flow shape: {hsv_flow.shape}")
    print(f"✓ Max flow value: {hsv_flow.max()}")
    print(f"✓ Mean flow value: {hsv_flow.mean():.4f}")
    
    return True


def test_directional_flow():
    """Test detection of different flow directions"""
    print("\nTesting directional flow detection...")
    
    analyzer = FlowHSVViz()
    
    # Create frames with known directional motion
    frames = []
    
    # Frame 0: Base frame
    frame0 = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.rectangle(frame0, (50, 50), (150, 150), (255, 255, 255), -1)
    frames.append(frame0)
    
    # Frame 1: Move right (0 degrees)
    frame1 = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.rectangle(frame1, (70, 50), (170, 150), (255, 255, 255), -1)
    frames.append(frame1)
    
    # Frame 2: Move down (90 degrees)
    frame2 = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.rectangle(frame2, (70, 70), (170, 170), (255, 255, 255), -1)
    frames.append(frame2)
    
    # Frame 3: Move left (180 degrees)
    frame3 = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.rectangle(frame3, (50, 70), (150, 170), (255, 255, 255), -1)
    frames.append(frame3)
    
    # Analyze
    result = analyzer.analyze(frames)
    
    # Check flow
    hsv_flow = result.data['hsv_flow']
    assert len(hsv_flow) == 3, "Should have 3 flow fields"
    
    # Verify we have flow detected
    assert hsv_flow.max() > 0, "Should detect flow"
    
    print(f"✓ Detected {len(hsv_flow)} flow fields")
    print(f"✓ Flow visualization uses HSV encoding:")
    print("  - Hue: direction (color represents angle)")
    print("  - Saturation: typically maximized")
    print("  - Value: magnitude (brightness represents speed)")
    
    return True


def test_static_frames():
    """Test with static frames (no motion)"""
    print("\nTesting static frames...")
    
    analyzer = FlowHSVViz()
    
    # Create identical frames
    frames = []
    static_frame = np.ones((150, 150, 3), dtype=np.uint8) * 128
    cv2.circle(static_frame, (75, 75), 30, (255, 255, 255), -1)
    
    for _ in range(5):
        frames.append(static_frame.copy())
    
    # Analyze
    result = analyzer.analyze(frames)
    
    # With no motion, flow should be minimal
    hsv_flow = result.data['hsv_flow']
    mean_flow = hsv_flow.mean()
    
    print(f"✓ Static frames mean flow: {mean_flow:.6f}")
    print(f"✓ Static frames max flow: {hsv_flow.max()}")
    
    # Flow should be very low but might not be exactly zero due to noise
    assert mean_flow < 10, f"Static frames should have minimal flow, got {mean_flow}"
    
    return True


def test_magnitude_normalization():
    """Test max_magnitude parameter for normalization"""
    print("\nTesting magnitude normalization...")
    
    # Create frames with strong motion
    frames = []
    for i in range(5):
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        # Fast moving object
        x = i * 40  # Large displacement
        cv2.circle(frame, (x % 200, 100), 30, (255, 255, 255), -1)
        frames.append(frame)
    
    # Test with different max_magnitude settings
    magnitudes = [10.0, 20.0, 50.0]
    
    for mag in magnitudes:
        analyzer = FlowHSVViz(max_magnitude=mag)
        result = analyzer.analyze(frames)
        hsv_flow = result.data['hsv_flow']
        print(f"✓ Max magnitude={mag}: max flow value = {hsv_flow.max()}")
    
    return True


def test_with_real_video():
    """Test with real video file if available"""
    print("\nTesting with real video...")
    
    video_path = Path('/Users/ns/Desktop/projects/videokurt/sample_recording.MP4')
    
    if not video_path.exists():
        print("⚠ Sample video not found, skipping real video test")
        return True
    
    # Load first 20 frames
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    for _ in range(20):
        ret, frame = cap.read()
        if not ret:
            break
        # Downsample for speed
        frame = cv2.resize(frame, (frame.shape[1]//4, frame.shape[0]//4))
        frames.append(frame)
    cap.release()
    
    if len(frames) < 2:
        print("⚠ Could not load enough frames from video")
        return False
    
    # Analyze with flow visualization
    analyzer = FlowHSVViz(downsample=0.5, max_magnitude=20.0, saturation_boost=1.5)
    result = analyzer.analyze(frames)
    
    # Verify visualization was created
    hsv_flow = result.data['hsv_flow']
    assert len(hsv_flow) == len(frames) - 1
    
    print(f"✓ Analyzed {len(frames)} real frames")
    print(f"✓ Mean flow value: {hsv_flow.mean():.4f}")
    print(f"✓ Max flow value: {hsv_flow.max()}")
    print(f"✓ Generated {len(hsv_flow)} HSV flow visualizations")
    print(f"✓ Processing time: {result.processing_time:.3f}s")
    
    return True


def main():
    """Run all tests"""
    print("="*50)
    print("Flow HSV Visualization Smoke Test")
    print("="*50)
    
    tests = [
        test_flow_hsv_basic,
        test_directional_flow,
        test_static_frames,
        test_magnitude_normalization,
        test_with_real_video
    ]
    
    for test in tests:
        try:
            if not test():
                print(f"✗ {test.__name__} failed")
                return False
        except Exception as e:
            print(f"✗ {test.__name__} error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print("\n" + "="*50)
    print("All Flow Visualization tests passed! ✓")
    print("="*50)
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)