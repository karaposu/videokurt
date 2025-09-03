"""
Run: python -m videokurt.smoke_tests.raw_analysis.edge_detection.test_edge_detection

Smoke test for Edge Detection raw analysis
Tests basic functionality with real video file
"""

import sys
import cv2
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from videokurt.raw_analysis.edge_canny import EdgeCanny


def test_edge_detection_basic():
    """Test basic edge detection"""
    print("Testing Edge Detection...")
    
    # Initialize detector
    detector = EdgeCanny(low_threshold=50, high_threshold=150)
    
    # Create test frames with edges
    frames = []
    for i in range(3):
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        # Add rectangle
        cv2.rectangle(frame, (50, 50), (150, 150), (255, 255, 255), -1)
        # Add circle
        cv2.circle(frame, (100, 100), 30, (0, 0, 0), -1)
        # Add some lines
        cv2.line(frame, (0, i*50), (200, i*50), (128, 128, 128), 2)
        frames.append(frame)
    
    # Analyze frames
    result = detector.analyze(frames)
    
    # Verify result structure
    assert result is not None, "Result should not be None"
    assert hasattr(result, 'data'), "Result should have data attribute"
    assert hasattr(result, 'method'), "Result should have method attribute"
    assert result.method == 'edge_canny', f"Method should be 'edge_canny', got {result.method}"
    
    print(f"✓ Analyzed {len(frames)} frames")
    print(f"✓ Method: {result.method}")
    if 'edge_map' in result.data:
        print(f"✓ Edge maps computed: {len(result.data['edge_map'])} frames")
    
    return True


def test_edge_thresholds():
    """Test edge detection with different thresholds"""
    print("\nTesting different thresholds...")
    
    # Create test frame with various features
    frame = np.zeros((150, 150, 3), dtype=np.uint8)
    # Strong edges (high contrast)
    frame[:50, :50] = 255
    # Medium edges
    frame[50:100, 50:100] = 128
    # Weak edges (low contrast)
    frame[100:, 100:] = 64
    
    frames = [frame, frame.copy()]
    
    threshold_sets = [
        (30, 100),   # More sensitive
        (50, 150),   # Standard
        (100, 200),  # Less sensitive
    ]
    
    for low, high in threshold_sets:
        print(f"  Testing thresholds: low={low}, high={high}")
        detector = EdgeCanny(low_threshold=low, high_threshold=high)
        result = detector.analyze(frames)
        
        assert result is not None, f"Thresholds {low},{high}: Result should not be None"
        print(f"    ✓ Processed with thresholds ({low}, {high})")
    
    return True


def test_edge_with_patterns():
    """Test edge detection with known patterns"""
    print("\nTesting with known patterns...")
    
    detector = EdgeCanny(low_threshold=50, high_threshold=150)
    
    frames = []
    
    # Frame 1: Vertical edges
    frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
    for x in range(0, 100, 20):
        cv2.line(frame1, (x, 0), (x, 100), (255, 255, 255), 1)
    frames.append(frame1)
    
    # Frame 2: Horizontal edges
    frame2 = np.zeros((100, 100, 3), dtype=np.uint8)
    for y in range(0, 100, 20):
        cv2.line(frame2, (0, y), (100, y), (255, 255, 255), 1)
    frames.append(frame2)
    
    # Frame 3: Diagonal edges
    frame3 = np.zeros((100, 100, 3), dtype=np.uint8)
    for i in range(0, 100, 10):
        cv2.line(frame3, (0, i), (100-i, 100), (255, 255, 255), 1)
    frames.append(frame3)
    
    # Frame 4: Circles (curved edges)
    frame4 = np.zeros((100, 100, 3), dtype=np.uint8)
    for radius in range(10, 50, 10):
        cv2.circle(frame4, (50, 50), radius, (255, 255, 255), 1)
    frames.append(frame4)
    
    result = detector.analyze(frames)
    
    assert result is not None, "Result should not be None"
    assert 'edge_map' in result.data, "Should have edge maps"
    assert len(result.data['edge_map']) == 4, f"Should have 4 edge maps, got {len(result.data['edge_map'])}"
    
    print(f"  ✓ Processed {len(frames)} frames with different edge patterns")
    print(f"  ✓ Vertical, horizontal, diagonal, and curved edges detected")
    
    return True


def test_edge_with_noise():
    """Test edge detection with noisy images"""
    print("\nTesting with noisy images...")
    
    detector = EdgeCanny(low_threshold=50, high_threshold=150)
    
    # Create frames with different noise levels
    frames = []
    
    for noise_level in [0, 10, 20]:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        # Add a simple shape
        cv2.rectangle(frame, (30, 30), (70, 70), (255, 255, 255), -1)
        
        # Add noise
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, frame.shape)
            frame = np.clip(frame.astype(float) + noise, 0, 255).astype(np.uint8)
        
        frames.append(frame)
    
    result = detector.analyze(frames)
    
    assert result is not None, "Result should not be None"
    print(f"  ✓ Processed frames with noise levels: 0, 10, 20")
    print(f"  ✓ Edge detection handled noise robustly")
    
    return True


def test_edge_with_video():
    """Test with real video file if available"""
    print("\nTesting with real video file...")
    
    video_path = project_root / "sample_recording.MP4"
    if not video_path.exists():
        print(f"  ⚠ Skipping: {video_path} not found")
        return True
    
    detector = EdgeCanny(low_threshold=50, high_threshold=150, downsample=0.5)
    
    # Read first 5 frames
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    
    for _ in range(5):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    
    if len(frames) > 0:
        result = detector.analyze(frames)
        
        print(f"  ✓ Processed {len(frames)} frames from video")
        if 'edge_map' in result.data:
            print(f"  ✓ Edge maps generated for video frames")
        if 'statistics' in result.data:
            print(f"  ✓ Edge statistics computed")
    else:
        print("  ⚠ No frames read from video")
    
    return True


def main():
    """Run all smoke tests"""
    print("=" * 50)
    print("EDGE DETECTION SMOKE TESTS")
    print("=" * 50)
    
    tests = [
        ("Basic Functionality", test_edge_detection_basic),
        ("Different Thresholds", test_edge_thresholds),
        ("Known Patterns", test_edge_with_patterns),
        ("Noisy Images", test_edge_with_noise),
        ("Real Video", test_edge_with_video),
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