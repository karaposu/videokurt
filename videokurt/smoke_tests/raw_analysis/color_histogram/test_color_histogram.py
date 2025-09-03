"""
Run: python -m videokurt.smoke_tests.raw_analysis.color_histogram.test_color_histogram

Smoke test for Color Histogram raw analysis
Tests basic functionality with real video file
"""

import sys
import cv2
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from videokurt.raw_analysis.color_histogram import ColorHistogram


def test_color_histogram_basic():
    """Test basic color histogram computation"""
    print("Testing Color Histogram...")
    
    # Initialize analyzer
    analyzer = ColorHistogram(bins=256, channels='gray')
    
    # Create test frames
    frames = []
    for i in range(5):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add gradient that changes over time
        frame[:, :] = int(50 + i * 40)
        frames.append(frame)
    
    # Analyze frames
    result = analyzer.analyze(frames)
    
    # Verify result structure
    assert result is not None, "Result should not be None"
    assert hasattr(result, 'data'), "Result should have data attribute"
    assert hasattr(result, 'method'), "Result should have method attribute"
    assert result.method == 'color_histogram', f"Method should be 'color_histogram', got {result.method}"
    
    # Check data structure
    assert 'histograms' in result.data, "Should have histograms in data"
    assert len(result.data['histograms']) == 5, f"Should have 5 histograms, got {len(result.data['histograms'])}"
    
    print(f"✓ Analyzed {len(frames)} frames")
    print(f"✓ Histogram shape: {result.data['histograms'][0].shape}")
    print(f"✓ Method: {result.method}")
    
    return True


def test_color_histogram_rgb():
    """Test RGB color histogram"""
    print("\nTesting RGB histogram...")
    
    analyzer = ColorHistogram(bins=32, channels='rgb')
    
    # Create colorful test frames
    frames = []
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # R, G, B
    
    for color in colors:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:, :] = color
        frames.append(frame)
    
    result = analyzer.analyze(frames)
    
    assert result is not None, "Result should not be None"
    assert 'histograms' in result.data, "Should have histograms"
    assert len(result.data['histograms']) == 3, "Should have 3 histograms"
    
    # Each RGB histogram should have 3 channels
    for hist in result.data['histograms']:
        assert len(hist) == 3, "RGB histogram should have 3 channels"
    
    print(f"✓ RGB histograms computed")
    print(f"✓ Channels per histogram: {len(result.data['histograms'][0])}")
    
    return True


def test_color_histogram_with_video():
    """Test with real video file if available"""
    print("\nTesting with real video file...")
    
    video_path = project_root / "sample_recording.MP4"
    if not video_path.exists():
        print(f"  ⚠ Skipping: {video_path} not found")
        return True
    
    analyzer = ColorHistogram(bins=256, channels='gray', downsample=0.5)
    
    # Read first 10 frames
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    
    for _ in range(10):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    
    if len(frames) > 0:
        result = analyzer.analyze(frames)
        
        print(f"  ✓ Processed {len(frames)} frames from video")
        print(f"  ✓ Histograms shape: {result.data['histograms'][0].shape}")
        
        # Check temporal statistics if available
        if 'temporal_stats' in result.data:
            print(f"  ✓ Temporal stats available")
    else:
        print("  ⚠ No frames read from video")
    
    return True


def test_color_histogram_hsv():
    """Test HSV color histogram"""
    print("\nTesting HSV histogram...")
    
    analyzer = ColorHistogram(bins=30, channels='hsv')
    
    # Create frames with different hues
    frames = []
    for hue in [0, 60, 120]:  # Red, Yellow, Green hues
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv[:, :, 0] = hue  # Set hue
        hsv[:, :, 1] = 255   # Full saturation
        hsv[:, :, 2] = 255   # Full value
        frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        frames.append(frame)
    
    result = analyzer.analyze(frames)
    
    assert result is not None, "Result should not be None"
    assert 'histograms' in result.data, "Should have histograms"
    
    print(f"✓ HSV histograms computed")
    print(f"✓ Number of frames: {len(result.data['histograms'])}")
    
    return True


def test_normalized_histogram():
    """Test normalized histograms"""
    print("\nTesting normalized histograms...")
    
    analyzer = ColorHistogram(bins=256, channels='gray', normalize=True)
    
    # Create frames with different intensities
    frames = []
    for intensity in [50, 100, 150, 200]:
        frame = np.full((100, 100, 3), intensity, dtype=np.uint8)
        frames.append(frame)
    
    result = analyzer.analyze(frames)
    
    # Check if histograms are normalized
    for hist in result.data['histograms']:
        hist_sum = np.sum(hist)
        # Normalized histograms should sum to 1 (or close to it)
        assert 0.99 < hist_sum < 1.01, f"Normalized histogram should sum to ~1, got {hist_sum}"
    
    print(f"✓ Histograms are properly normalized")
    
    return True


def main():
    """Run all smoke tests"""
    print("=" * 50)
    print("COLOR HISTOGRAM SMOKE TESTS")
    print("=" * 50)
    
    tests = [
        ("Basic Functionality", test_color_histogram_basic),
        ("RGB Histogram", test_color_histogram_rgb),
        ("HSV Histogram", test_color_histogram_hsv),
        ("Normalized Histogram", test_normalized_histogram),
        ("Real Video", test_color_histogram_with_video),
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