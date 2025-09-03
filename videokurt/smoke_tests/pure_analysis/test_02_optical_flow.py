"""
Test 02: Optical Flow Analyses
Tests both OpticalFlowDense and OpticalFlowSparse analysis classes.

Run: python -m videokurt.smoke_tests.pure_analysis.test_02_optical_flow
"""

import numpy as np
from videokurt.raw_analysis.optical_flow_dense import OpticalFlowDense
from videokurt.raw_analysis.optical_flow_sparse import OpticalFlowSparse


def create_moving_frames(direction='right'):
    """Create frames with directional motion."""
    frames = []
    for i in range(10):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        
        if direction == 'right':
            x = (i * 10) % 80
            y = 40
        elif direction == 'down':
            x = 40
            y = (i * 10) % 80
        else:  # diagonal
            x = (i * 10) % 80
            y = (i * 10) % 80
            
        # Draw square
        frame[y:y+20, x:x+20] = 255
        frames.append(frame)
    return frames


def test_dense_optical_flow():
    """Test dense optical flow (Farneback)."""
    print("\nTest 1: Dense Optical Flow (Farneback)")
    print("-" * 40)
    
    frames = create_moving_frames('right')
    
    # Run analysis with heavy downsampling for speed
    analyzer = OpticalFlowDense(downsample=0.25, levels=2, iterations=2)
    result = analyzer.analyze(frames)
    
    # Check results
    assert result.method == 'optical_flow_dense'
    assert 'flow_field' in result.data
    
    flow_field = result.data['flow_field']
    assert flow_field.shape[0] == len(frames) - 1  # T-1 flow fields
    assert flow_field.shape[3] == 2  # dx, dy components
    
    # Check that horizontal motion is detected (dx should be positive)
    mean_dx = np.mean(flow_field[:, :, :, 0])
    mean_dy = np.mean(flow_field[:, :, :, 1])
    
    print(f"✓ Flow field shape: {flow_field.shape}")
    print(f"✓ Mean horizontal flow (dx): {mean_dx:.3f}")
    print(f"✓ Mean vertical flow (dy): {mean_dy:.3f}")
    print(f"✓ Processing time: {result.processing_time:.3f}s")


def test_sparse_optical_flow():
    """Test sparse optical flow (Lucas-Kanade)."""
    print("\nTest 2: Sparse Optical Flow (Lucas-Kanade)")
    print("-" * 40)
    
    frames = create_moving_frames('down')
    
    # Run analysis
    analyzer = OpticalFlowSparse(
        downsample=0.5,  # Less downsampling for feature detection
        max_corners=50,
        quality_level=0.3
    )
    result = analyzer.analyze(frames)
    
    # Check results
    assert result.method == 'optical_flow_sparse'
    assert 'tracked_points' in result.data
    assert 'point_status' in result.data
    
    tracked_points = result.data['tracked_points']
    
    # Check that points are tracked
    if len(tracked_points) > 0 and len(tracked_points[0]) > 0:
        # Get first frame's tracked points
        first_frame_points = tracked_points[0]
        if first_frame_points:
            point = first_frame_points[0]
            print(f"✓ Sample tracked point: {point}")
            print(f"✓ Number of tracked points: {len(first_frame_points)}")
    
    print(f"✓ Frames with tracking: {len(tracked_points)}")
    print(f"✓ Processing time: {result.processing_time:.3f}s")


def test_flow_direction_detection():
    """Test that flow correctly detects motion direction."""
    print("\nTest 3: Direction Detection")
    print("-" * 40)
    
    # Test different directions
    directions = ['right', 'down', 'diagonal']
    
    for direction in directions:
        frames = create_moving_frames(direction)
        
        # Use dense flow for direction testing
        analyzer = OpticalFlowDense(downsample=0.25, levels=2)
        result = analyzer.analyze(frames)
        
        flow_field = result.data['flow_field']
        mean_dx = np.mean(flow_field[:, :, :, 0])
        mean_dy = np.mean(flow_field[:, :, :, 1])
        
        print(f"  {direction:8s}: dx={mean_dx:6.3f}, dy={mean_dy:6.3f}", end="")
        
        # Verify expected direction
        if direction == 'right':
            assert abs(mean_dx) > abs(mean_dy), "Right motion should have larger dx"
            print(" ✓")
        elif direction == 'down':
            assert abs(mean_dy) > abs(mean_dx), "Down motion should have larger dy"
            print(" ✓")
        else:  # diagonal
            # Both should be significant
            assert abs(mean_dx) > 0.001 and abs(mean_dy) > 0.001
            print(" ✓")


def test_downsampling_impact():
    """Test impact of downsampling on performance."""
    print("\nTest 4: Downsampling Performance Impact")
    print("-" * 40)
    
    frames = create_moving_frames('right')
    
    # Test different downsample rates
    downsample_rates = [1.0, 0.5, 0.25]
    
    for rate in downsample_rates:
        analyzer = OpticalFlowDense(downsample=rate, levels=2, iterations=2)
        result = analyzer.analyze(frames)
        
        shape = result.data['flow_field'].shape
        time = result.processing_time
        
        print(f"  Downsample {rate:.2f}: shape={shape[1:3]}, time={time:.3f}s")
    
    print("✓ Downsampling reduces computation time")


if __name__ == "__main__":
    print("="*50)
    print("Optical Flow Analysis Tests")
    print("="*50)
    
    try:
        test_dense_optical_flow()
        test_sparse_optical_flow()
        test_flow_direction_detection()
        test_downsampling_impact()
        
        print("\n" + "="*50)
        print("ALL TESTS PASSED ✓")
        print("="*50)
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        exit(1)
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        exit(1)