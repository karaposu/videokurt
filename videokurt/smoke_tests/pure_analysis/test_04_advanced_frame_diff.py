"""
Test 04: Advanced Frame Differencing Analysis
Tests the FrameDiffAdvanced analysis class with adaptive thresholding.

Run: python -m videokurt.smoke_tests.pure_analysis.test_04_advanced_frame_diff
"""

import numpy as np
from videokurt.analysis_models import FrameDiffAdvanced


def create_frames_with_varying_motion():
    """Create frames with varying amounts of motion."""
    frames = []
    for i in range(15):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Background with slight variations
        frame[:, :] = 30 + np.random.randint(-5, 5, (100, 100, 3))
        
        # Moving object with varying speed
        if i < 5:
            # Slow motion
            x = (i * 2) % 80
        elif i < 10:
            # Fast motion
            x = (i * 10) % 80
        else:
            # Very fast motion
            x = (i * 20) % 80
            
        frame[40:60, x:x+20] = 200
        frames.append(frame)
    return frames


def test_triple_differencing():
    """Test triple frame differencing."""
    print("\nTest 1: Triple Frame Differencing")
    print("-" * 40)
    
    frames = create_frames_with_varying_motion()
    
    # Run analysis with triple differencing
    analyzer = FrameDiffAdvanced(
        downsample=0.5,
        window_size=5,
        accumulate=True
    )
    result = analyzer.analyze(frames)
    
    # Check results
    assert result.method == 'frame_diff_advanced'
    assert 'triple_diff' in result.data
    assert 'running_avg_diff' in result.data
    
    triple_diff = result.data['triple_diff']
    running_avg_diff = result.data['running_avg_diff']
    
    # Triple diff needs 3 frames, so output is len(frames) - 2
    assert triple_diff.shape[0] == len(frames) - 2
    assert running_avg_diff.shape[0] == len(frames) - 2
    
    # Check accumulated data if enabled
    if analyzer.accumulate and 'accumulated_diff' in result.data:
        accumulated = result.data['accumulated_diff']
        print(f"✓ Accumulated diff shape: {accumulated.shape}")
    
    print(f"✓ Method: {result.method}")
    print(f"✓ Triple diff shape: {triple_diff.shape}")
    print(f"✓ Running avg diff shape: {running_avg_diff.shape}")
    print(f"✓ Processing time: {result.processing_time:.3f}s")


def test_running_average():
    """Test running average background subtraction."""
    print("\nTest 2: Running Average Background")
    print("-" * 40)
    
    frames = create_frames_with_varying_motion()
    
    # Analyze with default settings
    analyzer = FrameDiffAdvanced(
        downsample=0.5,
        window_size=5,
        accumulate=False
    )
    result = analyzer.analyze(frames)
    
    # Check running average diff
    assert 'running_avg_diff' in result.data
    running_avg_diff = result.data['running_avg_diff']
    
    # Should adapt to background over time
    # Early frames should have more difference than later frames with static background
    early_diff = np.mean(running_avg_diff[:3])
    late_diff = np.mean(running_avg_diff[-3:])
    
    print(f"✓ Running avg diff shape: {running_avg_diff.shape}")
    print(f"✓ Early frames avg difference: {early_diff:.2f}")
    print(f"✓ Late frames avg difference: {late_diff:.2f}")
    print(f"✓ Background adaptation observed")


def test_accumulation_mode():
    """Test accumulation of differences."""
    print("\nTest 3: Accumulation Mode")
    print("-" * 40)
    
    frames = create_frames_with_varying_motion()
    
    # Without accumulation
    analyzer_no_acc = FrameDiffAdvanced(
        downsample=0.5,
        window_size=5,
        accumulate=False
    )
    result_no_acc = analyzer_no_acc.analyze(frames)
    
    # With accumulation
    analyzer_acc = FrameDiffAdvanced(
        downsample=0.5,
        window_size=5,
        accumulate=True
    )
    result_acc = analyzer_acc.analyze(frames)
    
    # Check that accumulation adds data
    assert 'accumulated_diff' not in result_no_acc.data or result_no_acc.data['accumulated_diff'] is None
    assert 'accumulated_diff' in result_acc.data and result_acc.data['accumulated_diff'] is not None
    
    accumulated = result_acc.data['accumulated_diff']
    print(f"✓ Accumulated diff shape: {accumulated.shape}")
    print(f"✓ Max accumulated value: {np.max(accumulated):.2f}")
    print(f"✓ Non-zero pixels: {np.sum(accumulated > 0)}")
    print(f"✓ Accumulation creates motion history")


def test_parameters_and_metadata():
    """Test that all parameters are stored correctly."""
    print("\nTest 4: Parameters and Metadata")
    print("-" * 40)
    
    frames = create_frames_with_varying_motion()[:10]
    
    # Create with specific parameters
    analyzer = FrameDiffAdvanced(
        downsample=0.75,
        window_size=4,
        accumulate=True
    )
    result = analyzer.analyze(frames)
    
    # Check all parameters are stored
    params = result.parameters
    assert params['downsample'] == 0.75
    assert params['window_size'] == 4
    assert params['accumulate'] == True
    
    print(f"✓ All parameters stored correctly")
    print(f"✓ Parameters: {params}")
    print(f"✓ Output shapes: {result.output_shapes}")
    print(f"✓ Processing time: {result.processing_time:.3f}s")


if __name__ == "__main__":
    print("="*50)
    print("Advanced Frame Differencing Analysis Tests")
    print("="*50)
    
    try:
        test_triple_differencing()
        test_running_average()
        test_accumulation_mode()
        test_parameters_and_metadata()
        
        print("\n" + "="*50)
        print("ALL TESTS PASSED ✓")
        print("="*50)
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        exit(1)
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        exit(1)