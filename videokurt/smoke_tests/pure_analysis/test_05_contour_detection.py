"""
Test 05: Contour Detection Analysis
Tests the ContourDetection analysis class for shape detection.

Run: python -m videokurt.smoke_tests.pure_analysis.test_05_contour_detection
"""

import numpy as np
from videokurt.raw_analysis.contour_detection import ContourDetection


def create_frames_with_shapes():
    """Create frames with moving shapes for contour detection."""
    frames = []
    for i in range(10):
        frame = np.zeros((150, 150, 3), dtype=np.uint8)
        
        # Moving rectangle
        x_rect = 20 + i * 3
        frame[20:50, x_rect:x_rect+30] = 255
        
        # Moving circle (approximate)
        center = (100 - i * 2, 40 + i * 2)
        radius = 15
        y, x = np.ogrid[:150, :150]
        mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
        frame[mask] = 200
        
        # Moving triangle (approximate)
        offset = i * 4
        triangle = np.array([[50 + offset, 100], [70 + offset, 130], [30 + offset, 130]], np.int32)
        for pt in triangle:
            if 0 <= pt[1] < 150 and 0 <= pt[0] < 150:
                frame[pt[1]-2:pt[1]+2, pt[0]-2:pt[0]+2] = 150
        
        # Fast moving small square
        x = 60 + (i * 10) % 60
        frame[80:90, x:x+10] = 180
        
        frames.append(frame)
    return frames


def test_contour_detection():
    """Test basic contour detection."""
    print("\nTest 1: Basic Contour Detection")
    print("-" * 40)
    
    frames = create_frames_with_shapes()
    
    # Run analysis
    analyzer = ContourDetection(
        downsample=1.0,  # No downsampling to preserve details
        threshold=20,  # Lower threshold for motion detection
        max_contours=50
    )
    result = analyzer.analyze(frames)
    
    # Check results
    assert result.method == 'contour_detection'
    assert 'contours' in result.data
    assert 'hierarchy' in result.data
    
    contours = result.data['contours']
    hierarchy = result.data['hierarchy']
    
    # Contours are detected from frame differences, so len is frames-1
    assert len(contours) == len(frames) - 1
    assert len(hierarchy) == len(frames) - 1
    
    # Should detect multiple shapes
    total_contours = sum(len(c) for c in contours)
    assert total_contours > 0, "No contours detected"
    
    print(f"✓ Method: {result.method}")
    print(f"✓ Frame pairs processed: {len(contours)}")
    print(f"✓ Total contours detected: {total_contours}")
    print(f"✓ Processing time: {result.processing_time:.3f}s")


def test_threshold_effect():
    """Test threshold effect on contour detection."""
    print("\nTest 2: Threshold Effect")
    print("-" * 40)
    
    frames = create_frames_with_shapes()
    
    # Very low threshold - more sensitive
    analyzer_low = ContourDetection(
        downsample=1.0,
        threshold=10,
        max_contours=100
    )
    result_low = analyzer_low.analyze(frames)
    count_low = sum(len(c) for c in result_low.data['contours'])
    
    # Medium threshold
    analyzer_med = ContourDetection(
        downsample=1.0,
        threshold=50,
        max_contours=100
    )
    result_med = analyzer_med.analyze(frames)
    count_med = sum(len(c) for c in result_med.data['contours'])
    
    print(f"✓ Contours with threshold=10: {count_low}")
    print(f"✓ Contours with threshold=50: {count_med}")
    
    # Lower threshold should detect more or equal contours
    assert count_low >= count_med, "Lower threshold should detect more contours"
    print(f"✓ Threshold affects detection sensitivity")


def test_max_contours_limit():
    """Test max contours limitation."""
    print("\nTest 3: Max Contours Limit")
    print("-" * 40)
    
    frames = create_frames_with_shapes()
    
    # Small limit
    analyzer_small = ContourDetection(
        downsample=0.5,
        threshold=100,
        max_contours=2
    )
    result_small = analyzer_small.analyze(frames)
    
    # Large limit
    analyzer_large = ContourDetection(
        downsample=0.5,
        threshold=100,
        max_contours=50
    )
    result_large = analyzer_large.analyze(frames)
    
    # Check limits are respected
    for frame_contours in result_small.data['contours']:
        assert len(frame_contours) <= 2, "Should respect max_contours=2"
    
    for frame_contours in result_large.data['contours']:
        assert len(frame_contours) <= 50, "Should respect max_contours=50"
    
    total_small = sum(len(c) for c in result_small.data['contours'])
    total_large = sum(len(c) for c in result_large.data['contours'])
    
    print(f"✓ Contours with max=2: {total_small}")
    print(f"✓ Contours with max=50: {total_large}")
    print(f"✓ Limits properly enforced")


def test_contour_properties():
    """Test contour properties computation."""
    print("\nTest 4: Contour Properties")
    print("-" * 40)
    
    # Create frames with motion
    frames = []
    for i in range(5):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        # Moving rectangle
        x = 30 + i * 5
        frame[30:60, x:x+30] = 255
        frames.append(frame)
    
    analyzer = ContourDetection(
        downsample=1.0,  # No downsampling
        threshold=30,
        max_contours=10
    )
    result = analyzer.analyze(frames)
    
    contours = result.data['contours']
    
    # Compute properties from contours
    for frame_idx, frame_contours in enumerate(contours):
        if len(frame_contours) > 0:
            # We can compute area and centroid from the contours
            import cv2
            areas = [cv2.contourArea(c) for c in frame_contours]
            
            if areas:
                max_area = max(areas)
                print(f"✓ Frame {frame_idx}: max contour area = {max_area:.0f} pixels")
            
            # Compute centroid of largest contour
            if frame_contours:
                largest_contour = max(frame_contours, key=cv2.contourArea)
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]
                    print(f"✓ Frame {frame_idx}: centroid = ({cx:.1f}, {cy:.1f})")
    
    print(f"✓ Contour properties can be computed from raw contours")


if __name__ == "__main__":
    print("="*50)
    print("Contour Detection Analysis Tests")
    print("="*50)
    
    try:
        test_contour_detection()
        test_threshold_effect()
        test_max_contours_limit()
        test_contour_properties()
        
        print("\n" + "="*50)
        print("ALL TESTS PASSED ✓")
        print("="*50)
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        exit(1)
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        exit(1)