"""
Run: python -m videokurt.smoke_tests.raw_analysis.frequency_analysis.test_frequency_analysis

Smoke test for Frequency Analysis raw analysis
Tests basic functionality with real video file
"""

import sys
import cv2
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from videokurt.raw_analysis.frequency_fft import FrequencyFFT


def test_frequency_basic():
    """Test basic frequency analysis"""
    print("Testing Frequency FFT...")
    
    # Initialize analyzer (needs enough frames for FFT window)
    analyzer = FrequencyFFT(window_size=32, overlap=0.5, downsample=0.5)
    
    # Create test frames with temporal patterns
    frames = []
    for t in range(64):  # Need at least window_size frames
        frame = np.zeros((128, 128, 3), dtype=np.uint8)
        
        # Add oscillating pattern (creates temporal frequency)
        brightness = int(128 + 100 * np.sin(2 * np.pi * t / 10))  # 10-frame period
        frame[:64, :64] = brightness
        
        # Add static region
        frame[64:, :64] = 200
        
        # Add high-frequency flicker
        if t % 2 == 0:
            frame[:64, 64:] = 255
        else:
            frame[:64, 64:] = 0
            
        # Add random noise region
        frame[64:, 64:] = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        
        frames.append(frame)
    
    # Analyze frames
    result = analyzer.analyze(frames)
    
    # Verify result structure
    assert result is not None, "Result should not be None"
    assert hasattr(result, 'data'), "Result should have data attribute"
    assert hasattr(result, 'method'), "Result should have method attribute"
    assert result.method == 'frequency_fft', f"Method should be 'frequency_fft', got {result.method}"
    
    print(f"✓ Analyzed {len(frames)} frames")
    print(f"✓ Method: {result.method}")
    if 'frequency_spectrum' in result.data:
        print(f"✓ Frequency spectrum computed")
    
    return True


def test_different_window_sizes():
    """Test FFT with different window sizes"""
    print("\nTesting different window sizes...")
    
    window_sizes = [16, 32, 64]
    
    for window_size in window_sizes:
        print(f"  Testing window size {window_size}...")
        
        # Create enough frames for the window
        frames = []
        for t in range(window_size * 2):  # 2x window size for safety
            frame = np.zeros((64, 64, 3), dtype=np.uint8)
            # Simple oscillating pattern
            value = int(128 + 64 * np.sin(2 * np.pi * t / window_size))
            frame[:, :] = value
            frames.append(frame)
        
        analyzer = FrequencyFFT(window_size=window_size, overlap=0.0, downsample=1.0)
        result = analyzer.analyze(frames)
        
        assert result is not None, f"Window {window_size}: Result should not be None"
        print(f"    ✓ Window size {window_size} processed")
    
    return True


def test_temporal_patterns():
    """Test detection of different temporal patterns"""
    print("\nTesting temporal patterns...")
    
    analyzer = FrequencyFFT(window_size=32, overlap=0.5, downsample=0.5)
    
    # Create frames with known temporal frequencies
    frames = []
    for t in range(64):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Region 1: Slow oscillation (low frequency)
        slow = int(128 + 100 * np.sin(2 * np.pi * t / 32))  # 32-frame period
        frame[:50, :50] = slow
        
        # Region 2: Fast oscillation (high frequency)  
        fast = int(128 + 100 * np.sin(2 * np.pi * t / 4))   # 4-frame period
        frame[:50, 50:] = fast
        
        # Region 3: Step function (mixed frequencies)
        if (t // 16) % 2 == 0:
            frame[50:, :50] = 255
        else:
            frame[50:, :50] = 0
            
        # Region 4: Constant (no temporal change)
        frame[50:, 50:] = 128
        
        frames.append(frame)
    
    result = analyzer.analyze(frames)
    
    assert result is not None, "Result should not be None"
    print(f"  ✓ Analyzed frames with different temporal frequencies")
    print(f"  ✓ Low, high, step, and constant patterns processed")
    
    return True


def test_minimum_frames():
    """Test with minimum required frames"""
    print("\nTesting minimum frame requirements...")
    
    window_size = 16
    analyzer = FrequencyFFT(window_size=window_size, overlap=0.0)
    
    # Test with exactly window_size frames
    frames = []
    for t in range(window_size):
        frame = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        frames.append(frame)
    
    result = analyzer.analyze(frames)
    assert result is not None, "Should work with exactly window_size frames"
    print(f"  ✓ Processed with exactly {window_size} frames")
    
    # Test with fewer frames (should fail)
    try:
        insufficient_frames = frames[:window_size-1]
        result = analyzer.analyze(insufficient_frames)
        print(f"  ✗ Should have failed with {window_size-1} frames")
        return False
    except ValueError as e:
        print(f"  ✓ Correctly rejected {window_size-1} frames: {e}")
    
    return True


def test_with_grayscale():
    """Test with grayscale-like input"""
    print("\nTesting with grayscale patterns...")
    
    analyzer = FrequencyFFT(window_size=32, overlap=0.5)
    
    # Create grayscale frames
    frames = []
    for t in range(64):
        # Create single-channel pattern
        gray = np.zeros((80, 80), dtype=np.uint8)
        
        # Add temporal sine wave
        value = int(128 + 100 * np.sin(2 * np.pi * t / 16))
        gray[:, :] = value
        
        # Convert to 3-channel for consistency
        frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        frames.append(frame)
    
    result = analyzer.analyze(frames)
    
    assert result is not None, "Result should not be None"
    print(f"  ✓ Processed grayscale temporal patterns")
    
    return True


def main():
    """Run all smoke tests"""
    print("=" * 50)
    print("FREQUENCY FFT SMOKE TESTS")
    print("=" * 50)
    
    tests = [
        ("Basic Functionality", test_frequency_basic),
        ("Window Sizes", test_different_window_sizes),
        ("Temporal Patterns", test_temporal_patterns),
        ("Minimum Frames", test_minimum_frames),
        ("Grayscale Input", test_with_grayscale),
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