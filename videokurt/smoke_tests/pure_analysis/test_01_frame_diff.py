"""
Test 01: Frame Differencing Analysis
Tests the FrameDiff analysis class for basic frame differencing.

Run: python -m videokurt.smoke_tests.pure_analysis.test_01_frame_diff
"""

import numpy as np
import time
from videokurt.analysis_models import FrameDiff


def create_test_frames(motion=True):
    """Create simple test frames."""
    frames = []
    for i in range(5):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        if motion:
            # Moving square
            x = (i * 20) % 80
            frame[10:30, x:x+20] = 255
        else:
            # Static square
            frame[10:30, 40:60] = 255
        frames.append(frame)
    return frames


def test_basic_differencing():
    """Test basic frame differencing."""
    print("\nTest 1: Basic Frame Differencing")
    print("-" * 40)
    
    # Create frames with motion
    frames = create_test_frames(motion=True)
    
    # Run analysis
    analyzer = FrameDiff(downsample=1.0, threshold=0.1)
    result = analyzer.analyze(frames)
    
    # Check results
    assert result.method == 'frame_diff'
    assert 'pixel_diff' in result.data
    assert result.data['pixel_diff'].shape[0] == len(frames) - 1
    
    print(f"✓ Method: {result.method}")
    print(f"✓ Output shape: {result.data['pixel_diff'].shape}")
    print(f"✓ Processing time: {result.processing_time:.3f}s")
    

def test_no_motion():
    """Test with static frames."""
    print("\nTest 2: Static Frames (No Motion)")
    print("-" * 40)
    
    # Create static frames
    frames = create_test_frames(motion=False)
    
    # Run analysis
    analyzer = FrameDiff(downsample=1.0)
    result = analyzer.analyze(frames)
    
    # Check that differences are minimal
    pixel_diffs = result.data['pixel_diff']
    mean_diff = np.mean(pixel_diffs)
    
    assert mean_diff < 1.0, f"Static frames should have minimal difference, got {mean_diff}"
    
    print(f"✓ Mean difference for static frames: {mean_diff:.4f}")
    print(f"✓ Max difference: {np.max(pixel_diffs):.4f}")


def test_downsampling():
    """Test downsampling parameter."""
    print("\nTest 3: Downsampling")
    print("-" * 40)
    
    frames = create_test_frames(motion=True)
    
    # Full resolution
    analyzer_full = FrameDiff(downsample=1.0)
    result_full = analyzer_full.analyze(frames)
    
    # Half resolution
    analyzer_half = FrameDiff(downsample=0.5)
    result_half = analyzer_half.analyze(frames)
    
    # Check shapes
    shape_full = result_full.data['pixel_diff'].shape
    shape_half = result_half.data['pixel_diff'].shape
    
    assert shape_half[1] == shape_full[1] // 2
    assert shape_half[2] == shape_full[2] // 2
    
    print(f"✓ Full resolution shape: {shape_full}")
    print(f"✓ Half resolution shape: {shape_half}")
    print(f"✓ Speedup: {result_full.processing_time / result_half.processing_time:.2f}x")


def test_parameters_stored():
    """Test that parameters are stored correctly."""
    print("\nTest 4: Parameter Storage")
    print("-" * 40)
    
    frames = create_test_frames()
    
    # Custom parameters
    analyzer = FrameDiff(downsample=0.75, threshold=0.2)
    result = analyzer.analyze(frames)
    
    assert result.parameters['downsample'] == 0.75
    assert result.parameters['threshold'] == 0.2
    
    print(f"✓ Parameters stored: {result.parameters}")
    print(f"✓ Output shapes stored: {result.output_shapes}")
    print(f"✓ Data types stored: {result.dtype_info}")


if __name__ == "__main__":
    print("="*50)
    print("Frame Differencing Analysis Tests")
    print("="*50)
    
    try:
        test_basic_differencing()
        test_no_motion()
        test_downsampling()
        test_parameters_stored()
        
        print("\n" + "="*50)
        print("ALL TESTS PASSED ✓")
        print("="*50)
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        exit(1)
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        exit(1)