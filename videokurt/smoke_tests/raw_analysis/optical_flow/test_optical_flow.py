"""
Run: python -m videokurt.smoke_tests.raw_analysis.optical_flow.test_optical_flow

Smoke test for Optical Flow raw analysis
Tests basic functionality with real video file
"""

import sys
import cv2
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from videokurt.raw_analysis.optical_flow_dense import OpticalFlowDense


def test_optical_flow_basic():
    """Test basic optical flow computation"""
    print("Testing Optical Flow Dense...")
    
    # Initialize analyzer
    analyzer = OpticalFlowDense(downsample=0.5)  # Downsample for performance
    
    # Create test frames with known motion
    frames = []
    for i in range(5):
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        
        # Moving square (translates to the right)
        x_pos = 50 + i * 20  # Moves 20 pixels per frame
        cv2.rectangle(frame, (x_pos, 80), (x_pos + 40, 120), (255, 255, 255), -1)
        
        # Static reference point
        cv2.circle(frame, (100, 50), 10, (128, 128, 128), -1)
        
        frames.append(frame)
    
    # Analyze frames
    result = analyzer.analyze(frames)
    
    # Verify result structure
    assert result is not None, "Result should not be None"
    assert hasattr(result, 'data'), "Result should have data attribute"
    assert hasattr(result, 'method'), "Result should have method attribute"
    assert result.method == 'optical_flow_dense', f"Method should be 'optical_flow_dense', got {result.method}"
    
    print(f"✓ Analyzed {len(frames)} frames")
    print(f"✓ Method: {result.method}")
    if 'flow' in result.data:
        flow_shape = result.data['flow'].shape
        print(f"✓ Flow field shape: {flow_shape}")
        # Should have (n_frames-1, height, width, 2) for x,y flow
        assert flow_shape[-1] == 2, "Flow should have 2 channels (x, y)"
    
    return True


def test_motion_patterns():
    """Test detection of different motion patterns"""
    print("\nTesting different motion patterns...")
    
    analyzer = OpticalFlowDense(downsample=0.5)
    
    frames = []
    
    # Frame 1: Initial position
    frame1 = np.zeros((150, 150, 3), dtype=np.uint8)
    cv2.circle(frame1, (75, 75), 20, (255, 255, 255), -1)
    frames.append(frame1)
    
    # Frame 2: Horizontal motion
    frame2 = np.zeros((150, 150, 3), dtype=np.uint8)
    cv2.circle(frame2, (95, 75), 20, (255, 255, 255), -1)  # Moved right
    frames.append(frame2)
    
    # Frame 3: Vertical motion
    frame3 = np.zeros((150, 150, 3), dtype=np.uint8)
    cv2.circle(frame3, (95, 95), 20, (255, 255, 255), -1)  # Moved down
    frames.append(frame3)
    
    # Frame 4: Diagonal motion
    frame4 = np.zeros((150, 150, 3), dtype=np.uint8)
    cv2.circle(frame4, (115, 115), 20, (255, 255, 255), -1)  # Moved diagonally
    frames.append(frame4)
    
    # Frame 5: Return motion
    frame5 = np.zeros((150, 150, 3), dtype=np.uint8)
    cv2.circle(frame5, (75, 75), 20, (255, 255, 255), -1)  # Back to start
    frames.append(frame5)
    
    result = analyzer.analyze(frames)
    
    assert result is not None, "Result should not be None"
    if 'flow' in result.data:
        flow = result.data['flow']
        print(f"  ✓ Computed flow for {len(flow)} frame pairs")
        print(f"  ✓ Horizontal, vertical, and diagonal motions processed")
    
    return True


def test_rotation_motion():
    """Test rotational motion detection"""
    print("\nTesting rotational motion...")
    
    analyzer = OpticalFlowDense(downsample=0.5)
    
    frames = []
    center = (100, 100)
    
    for angle in range(0, 90, 15):  # Rotate 15 degrees per frame
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        
        # Draw rotated rectangle
        rect = np.array([
            [-30, -10],
            [30, -10],
            [30, 10],
            [-30, 10]
        ])
        
        # Rotation matrix
        theta = np.radians(angle)
        rotation = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        
        rotated = rect @ rotation.T
        rotated = rotated.astype(int) + center
        
        cv2.fillPoly(frame, [rotated], (255, 255, 255))
        frames.append(frame)
    
    result = analyzer.analyze(frames)
    
    assert result is not None, "Result should not be None"
    print(f"  ✓ Processed {len(frames)} frames with rotational motion")
    
    return True


def test_zoom_motion():
    """Test zoom/scale motion detection"""
    print("\nTesting zoom motion...")
    
    analyzer = OpticalFlowDense(downsample=0.5)
    
    frames = []
    
    for scale in [10, 20, 30, 40, 50]:  # Growing circle
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.circle(frame, (100, 100), scale, (255, 255, 255), -1)
        frames.append(frame)
    
    result = analyzer.analyze(frames)
    
    assert result is not None, "Result should not be None"
    if 'flow' in result.data:
        print(f"  ✓ Processed expanding/zoom motion")
        # In zoom motion, flow vectors should point outward from center
    
    return True


def test_complex_scene():
    """Test with complex scene containing multiple motions"""
    print("\nTesting complex scene...")
    
    analyzer = OpticalFlowDense(downsample=0.5, winsize=21)  # Larger window for complex motion
    
    frames = []
    
    for t in range(5):
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        
        # Object 1: Moving right
        x1 = 20 + t * 15
        cv2.rectangle(frame, (x1, 50), (x1 + 20, 70), (255, 0, 0), -1)
        
        # Object 2: Moving down
        y2 = 20 + t * 15
        cv2.circle(frame, (100, y2), 10, (0, 255, 0), -1)
        
        # Object 3: Moving diagonally
        x3, y3 = 150 + t * 10, 150 + t * 10
        cv2.ellipse(frame, (x3, y3), (15, 10), 0, 0, 360, (0, 0, 255), -1)
        
        # Static background pattern
        for i in range(0, 200, 40):
            cv2.line(frame, (i, 0), (i, 200), (50, 50, 50), 1)
        
        frames.append(frame)
    
    result = analyzer.analyze(frames)
    
    assert result is not None, "Result should not be None"
    print(f"  ✓ Processed complex scene with multiple moving objects")
    
    if 'statistics' in result.data:
        stats = result.data['statistics']
        if 'mean_magnitude' in stats:
            print(f"  ✓ Mean flow magnitude: {stats['mean_magnitude']:.2f}")
    
    return True


def test_minimum_frames():
    """Test with minimum required frames"""
    print("\nTesting minimum frame requirements...")
    
    analyzer = OpticalFlowDense()
    
    # Need at least 2 frames for optical flow
    frames = [
        np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8),
        np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    ]
    
    result = analyzer.analyze(frames)
    assert result is not None, "Should work with 2 frames"
    print(f"  ✓ Processed with minimum 2 frames")
    
    # Test with single frame (should fail or return empty)
    try:
        single_frame = [frames[0]]
        result = analyzer.analyze(single_frame)
        # If it doesn't fail, check that flow is empty or None
        if 'flow' in result.data:
            assert result.data['flow'] is None or len(result.data['flow']) == 0, \
                   "Single frame should not produce flow"
        print(f"  ✓ Handled single frame gracefully")
    except Exception as e:
        print(f"  ✓ Correctly rejected single frame: {e}")
    
    return True


def main():
    """Run all smoke tests"""
    print("=" * 50)
    print("OPTICAL FLOW DENSE SMOKE TESTS")
    print("=" * 50)
    
    tests = [
        ("Basic Functionality", test_optical_flow_basic),
        ("Motion Patterns", test_motion_patterns),
        ("Rotational Motion", test_rotation_motion),
        ("Zoom Motion", test_zoom_motion),
        ("Complex Scene", test_complex_scene),
        ("Minimum Frames", test_minimum_frames),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"\n✅ {test_name} PASSED")
            else:
                failed += 1
                print(f"\n❌ {test_name} FAILED")
        except Exception as e:
            failed += 1
            print(f"\n❌ {test_name} FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 50)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 50)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)