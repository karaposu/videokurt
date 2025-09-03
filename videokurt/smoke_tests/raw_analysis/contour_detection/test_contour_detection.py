"""
Run: python -m videokurt.smoke_tests.raw_analysis.contour_detection.test_contour_detection

Smoke test for Contour Detection raw analysis
Tests contour detection and hierarchy extraction
"""

import sys
import cv2
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from videokurt.raw_analysis.contour_detection import ContourDetection


def test_contour_detection_basic():
    """Test basic contour detection"""
    print("Testing Contour Detection...")
    
    # Initialize analyzer with correct parameters
    analyzer = ContourDetection(downsample=1.0, threshold=127, max_contours=100)
    
    # Create test frames with shapes
    frames = []
    for i in range(5):
        frame = np.zeros((400, 400, 3), dtype=np.uint8)
        
        # Add different shapes in each frame
        if i == 0:
            # Circle
            cv2.circle(frame, (200, 200), 50, (255, 255, 255), -1)
        elif i == 1:
            # Rectangle
            cv2.rectangle(frame, (150, 150), (250, 250), (255, 255, 255), -1)
        elif i == 2:
            # Multiple small circles
            cv2.circle(frame, (100, 100), 30, (255, 255, 255), -1)
            cv2.circle(frame, (300, 100), 30, (255, 255, 255), -1)
            cv2.circle(frame, (200, 300), 30, (255, 255, 255), -1)
        elif i == 3:
            # Triangle
            pts = np.array([[200, 100], [150, 250], [250, 250]], np.int32)
            cv2.fillPoly(frame, [pts], (255, 255, 255))
        else:
            # Complex shape (star-like)
            pts = []
            for angle in range(0, 360, 72):
                r = 80 if angle % 144 == 0 else 40
                x = int(200 + r * np.cos(np.radians(angle)))
                y = int(200 + r * np.sin(np.radians(angle)))
                pts.append([x, y])
            pts = np.array(pts, np.int32)
            cv2.fillPoly(frame, [pts], (255, 255, 255))
        
        frames.append(frame)
    
    # Analyze frames
    result = analyzer.analyze(frames)
    
    # Verify result structure
    assert result is not None, "Result should not be None"
    assert hasattr(result, 'data'), "Result should have data attribute"
    assert hasattr(result, 'method'), "Result should have method attribute"
    assert result.method == 'contour_detection', f"Method should be 'contour_detection', got {result.method}"
    
    # Check data structure - actual implementation returns 'contours' and 'hierarchy'
    assert 'contours' in result.data, "Should have contours in data"
    assert 'hierarchy' in result.data, "Should have hierarchy in data"
    
    contours = result.data['contours']
    hierarchy = result.data['hierarchy']
    
    # Contour detection works on frame differences, so n-1 outputs for n frames
    assert len(contours) == len(frames) - 1, "Should have n-1 contour sets for n frames"
    assert len(hierarchy) == len(frames) - 1, "Should have n-1 hierarchy sets for n frames"
    
    # Count contours per frame
    num_contours = [len(frame_contours) for frame_contours in contours]
    
    print(f"✓ Analyzed {len(frames)} frames")
    print(f"✓ Total contours detected: {sum(num_contours)}")
    print(f"✓ Contours per frame: {num_contours}")
    
    # Verify we detected some contours (shapes are different between frames)
    assert sum(num_contours) > 0, "Should detect some contours from frame differences"
    
    return True


def test_threshold_values():
    """Test different threshold values"""
    print("\nTesting threshold values...")
    
    # Create two frames with gradual intensity changes
    frame1 = np.zeros((200, 200, 3), dtype=np.uint8)
    # Add shapes with different intensities
    cv2.circle(frame1, (50, 50), 20, (100, 100, 100), -1)   # Low intensity
    cv2.circle(frame1, (150, 50), 20, (150, 150, 150), -1)  # Medium intensity
    cv2.circle(frame1, (100, 150), 20, (255, 255, 255), -1) # High intensity
    
    # Second frame with slight changes
    frame2 = frame1.copy()
    cv2.circle(frame2, (50, 50), 22, (100, 100, 100), -1)   # Slightly bigger
    
    frames = [frame1, frame2]
    
    # Test with different thresholds
    thresholds = [50, 120, 200]
    contour_counts = []
    
    for thresh in thresholds:
        analyzer = ContourDetection(threshold=thresh)
        result = analyzer.analyze(frames)
        num_contours = len(result.data['contours'][0])
        contour_counts.append(num_contours)
        print(f"✓ Threshold={thresh}: {num_contours} contours detected")
    
    # Higher thresholds should detect fewer contours
    assert contour_counts[0] >= contour_counts[2], "Lower threshold should detect more contours"
    
    return True


def test_max_contours_limit():
    """Test max_contours parameter"""
    print("\nTesting max_contours limit...")
    
    # Create two frames with many small objects (with motion)
    frame1 = np.zeros((300, 300, 3), dtype=np.uint8)
    frame2 = np.zeros((300, 300, 3), dtype=np.uint8)
    
    for i in range(10):
        for j in range(10):
            x, y = 30 + i * 25, 30 + j * 25
            cv2.circle(frame1, (x, y), 8, (255, 255, 255), -1)
            # Slightly offset in second frame to create motion
            cv2.circle(frame2, (x + 2, y), 8, (255, 255, 255), -1)
    
    frames = [frame1, frame2]
    
    # Test with different max_contours limits
    limits = [5, 20, 100]
    
    for limit in limits:
        analyzer = ContourDetection(max_contours=limit)
        result = analyzer.analyze(frames)
        num_contours = len(result.data['contours'][0])
        print(f"✓ Max contours={limit}: {num_contours} contours returned")
        assert num_contours <= limit, f"Should not exceed max_contours limit of {limit}"
    
    return True


def test_with_real_video():
    """Test with real video file if available"""
    print("\nTesting with real video...")
    
    video_path = Path('/Users/ns/Desktop/projects/videokurt/sample_recording.MP4')
    
    if not video_path.exists():
        print("⚠ Sample video not found, skipping real video test")
        return True
    
    # Load first 10 frames
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    for _ in range(10):
        ret, frame = cap.read()
        if not ret:
            break
        # Downsample significantly for speed
        frame = cv2.resize(frame, (frame.shape[1]//8, frame.shape[0]//8))
        frames.append(frame)
    cap.release()
    
    if len(frames) < 2:
        print("⚠ Could not load enough frames from video")
        return False
    
    # Analyze with reasonable parameters
    analyzer = ContourDetection(downsample=1.0, threshold=100, max_contours=50)
    result = analyzer.analyze(frames)
    
    contours = result.data['contours']
    total_contours = sum(len(frame_contours) for frame_contours in contours)
    avg_contours = total_contours / len(contours)
    
    print(f"✓ Analyzed {len(frames)} real frames")
    print(f"✓ Total contours: {total_contours}")
    print(f"✓ Average contours per frame: {avg_contours:.1f}")
    print(f"✓ Processing time: {result.processing_time:.3f}s")
    
    return True


def main():
    """Run all tests"""
    print("="*50)
    print("Contour Detection Smoke Test")
    print("="*50)
    
    tests = [
        test_contour_detection_basic,
        test_threshold_values,
        test_max_contours_limit,
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
    print("All Contour Detection tests passed! ✓")
    print("="*50)
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)