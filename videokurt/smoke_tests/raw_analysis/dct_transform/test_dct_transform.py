"""
Run: python -m videokurt.smoke_tests.raw_analysis.dct_transform.test_dct_transform

Smoke test for DCT Transform raw analysis
Tests basic functionality with real video file
"""

import sys
import cv2
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from videokurt.raw_analysis.dct_transform import DCTTransform


def test_dct_basic():
    """Test basic DCT computation"""
    print("Testing DCT Transform...")
    
    # Initialize analyzer
    analyzer = DCTTransform(block_size=8)
    
    # Create test frames with real patterns
    frames = []
    for i in range(3):
        frame = np.zeros((256, 256, 3), dtype=np.uint8)
        # Add low frequency pattern (gradient)
        for y in range(256):
            frame[y, :] = int(y * 255 / 256)
        # Add high frequency pattern (checkerboard) in part of image
        checker_size = 4
        for y in range(128, 256, checker_size * 2):
            for x in range(128, 256, checker_size * 2):
                frame[y:y+checker_size, x:x+checker_size] = 255
                if y+checker_size < 256 and x+checker_size < 256:
                    frame[y+checker_size:y+2*checker_size, x+checker_size:x+2*checker_size] = 255
        frames.append(frame)
    
    # Analyze frames
    result = analyzer.analyze(frames)
    
    # Verify result structure
    assert result is not None, "Result should not be None"
    assert hasattr(result, 'data'), "Result should have data attribute"
    assert hasattr(result, 'method'), "Result should have method attribute"
    assert result.method == 'dct_transform', f"Method should be 'dct_transform', got {result.method}"
    
    print(f"✓ Analyzed {len(frames)} frames")
    print(f"✓ Method: {result.method}")
    if 'dct_coefficients' in result.data:
        print(f"✓ DCT coefficients computed")
    
    return True


def test_dct_block_sizes():
    """Test DCT with different block sizes"""
    print("\nTesting different block sizes...")
    
    block_sizes = [4, 8, 16]
    
    # Create test frame with diagonal pattern
    frame = np.zeros((128, 128, 3), dtype=np.uint8)
    for i in range(128):
        if i < 127:
            frame[i, i] = 255
            frame[i, i+1] = 128 if i+1 < 128 else 0
    
    frames = [frame, frame.copy()]  # Need at least 2 frames for some analyses
    
    for block_size in block_sizes:
        print(f"  Testing block size {block_size}...")
        analyzer = DCTTransform(block_size=block_size)
        result = analyzer.analyze(frames)
        
        assert result is not None, f"Block {block_size}: Result should not be None"
        print(f"    ✓ Block size {block_size} processed")
    
    return True


def test_dct_with_patterns():
    """Test DCT with known patterns"""
    print("\nTesting with known patterns...")
    
    analyzer = DCTTransform(block_size=8)
    
    # Create frames with different frequency content
    frames = []
    
    # Frame 1: Low frequency (smooth gradient)
    frame1 = np.zeros((128, 128, 3), dtype=np.uint8)
    for i in range(128):
        frame1[i, :] = int(i * 255 / 128)
    frames.append(frame1)
    
    # Frame 2: High frequency (noise)
    frame2 = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
    frames.append(frame2)
    
    # Frame 3: Mixed (half smooth, half textured)
    frame3 = np.zeros((128, 128, 3), dtype=np.uint8)
    frame3[:, :64] = 128  # Smooth left half
    # Textured right half
    for i in range(0, 128, 4):
        frame3[i:i+2, 64:] = 255
    frames.append(frame3)
    
    result = analyzer.analyze(frames)
    
    assert result is not None, "Result should not be None"
    assert 'dct_coefficients' in result.data or 'statistics' in result.data, \
           "Should have DCT data"
    
    print(f"  ✓ Analyzed {len(frames)} frames with different patterns")
    print(f"  ✓ Low frequency, high frequency, and mixed patterns processed")
    
    return True


def test_dct_with_video():
    """Test with real video file if available"""
    print("\nTesting with real video file...")
    
    video_path = project_root / "sample_recording.MP4"
    if not video_path.exists():
        print(f"  ⚠ Skipping: {video_path} not found")
        return True
    
    analyzer = DCTTransform(block_size=8, downsample=0.5)
    
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
        result = analyzer.analyze(frames)
        
        print(f"  ✓ Processed {len(frames)} frames from video")
        if 'statistics' in result.data:
            stats = result.data['statistics']
            print(f"  ✓ Statistics computed")
    else:
        print("  ⚠ No frames read from video")
    
    return True


def test_dct_grayscale():
    """Test DCT with grayscale images"""
    print("\nTesting with grayscale images...")
    
    analyzer = DCTTransform(block_size=8)
    
    # Create grayscale frames
    frames = []
    for i in range(3):
        # Create grayscale frame with pattern
        gray = np.zeros((100, 100), dtype=np.uint8)
        # Add circular pattern
        center = (50, 50)
        for radius in range(10, 50, 10):
            cv2.circle(gray, center, radius, int(255 * (radius / 50)), 2)
        # Convert to 3-channel for consistency
        frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        frames.append(frame)
    
    result = analyzer.analyze(frames)
    
    assert result is not None, "Result should not be None"
    print(f"  ✓ Processed {len(frames)} grayscale frames")
    
    return True


def main():
    """Run all smoke tests"""
    print("=" * 50)
    print("DCT TRANSFORM SMOKE TESTS")
    print("=" * 50)
    
    tests = [
        ("Basic Functionality", test_dct_basic),
        ("Block Sizes", test_dct_block_sizes),
        ("Known Patterns", test_dct_with_patterns),
        ("Grayscale Images", test_dct_grayscale),
        ("Real Video", test_dct_with_video),
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