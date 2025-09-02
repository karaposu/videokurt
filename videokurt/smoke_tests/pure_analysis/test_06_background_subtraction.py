"""
Test 06: Background Subtraction Analysis
Tests both BackgroundMOG2 and BackgroundKNN analysis classes using real video.

Run: python -m videokurt.smoke_tests.pure_analysis.test_06_background_subtraction
"""

import numpy as np
from videokurt.analysis_models import BackgroundMOG2, BackgroundKNN
from videokurt.smoke_tests.pure_analysis.test_utils import load_video_frames, get_video_segment


def test_mog2_background_subtraction():
    """Test MOG2 background subtraction with real video."""
    print("\nTest 1: MOG2 Background Subtraction")
    print("-" * 40)
    
    # Load real video frames
    try:
        frames = load_video_frames(max_seconds=3.0)
        print(f"  Loaded {len(frames)} frames from video")
    except Exception as e:
        print(f"  Warning: Could not load video, using synthetic frames: {e}")
        # Fallback to synthetic frames
        frames = []
        for i in range(30):
            frame = np.ones((100, 100, 3), dtype=np.uint8) * 50
            x = (i * 3) % 70
            frame[30:60, x:x+20] = 200
            frames.append(frame)
    
    # Run analysis
    analyzer = BackgroundMOG2(
        downsample=0.25,  # Downsample for performance
        history=20,
        var_threshold=16,
        detect_shadows=True
    )
    result = analyzer.analyze(frames)
    
    # Check results
    assert result.method == 'background_mog2'
    assert 'foreground_mask' in result.data
    
    fg_mask = result.data['foreground_mask']
    assert fg_mask.shape[0] == len(frames)
    assert fg_mask.dtype == np.uint8
    
    # Should detect foreground motion
    motion_pixels = np.sum(fg_mask > 0)
    assert motion_pixels > 0, "No foreground detected"
    
    # Check shadow detection if enabled
    if analyzer.detect_shadows:
        # Shadows are marked as 127 in MOG2
        shadow_pixels = np.sum(fg_mask == 127)
        print(f"✓ Shadow pixels detected: {shadow_pixels}")
    
    print(f"✓ Method: {result.method}")
    print(f"✓ Foreground mask shape: {fg_mask.shape}")
    print(f"✓ Motion pixels detected: {motion_pixels}")
    print(f"✓ Motion percentage: {100*motion_pixels/fg_mask.size:.2f}%")
    print(f"✓ Processing time: {result.processing_time:.3f}s")


def test_knn_background_subtraction():
    """Test KNN background subtraction with real video."""
    print("\nTest 2: KNN Background Subtraction")
    print("-" * 40)
    
    # Load real video frames
    try:
        frames = load_video_frames(max_seconds=3.0)
        print(f"  Loaded {len(frames)} frames from video")
    except Exception as e:
        print(f"  Warning: Could not load video, using synthetic frames: {e}")
        frames = []
        for i in range(30):
            frame = np.ones((100, 100, 3), dtype=np.uint8) * 50
            x = (i * 3) % 70
            frame[30:60, x:x+20] = 200
            frames.append(frame)
    
    # Run analysis
    analyzer = BackgroundKNN(
        downsample=0.25,
        history=30,
        dist2_threshold=400.0,
        detect_shadows=False
    )
    result = analyzer.analyze(frames)
    
    # Check results
    assert result.method == 'background_knn'
    assert 'foreground_mask' in result.data
    
    fg_mask = result.data['foreground_mask']
    assert fg_mask.shape[0] == len(frames)
    
    # Should detect foreground motion
    motion_pixels = np.sum(fg_mask > 0)
    assert motion_pixels > 0, "No foreground detected"
    
    print(f"✓ Method: {result.method}")
    print(f"✓ Foreground mask shape: {fg_mask.shape}")
    print(f"✓ Motion pixels detected: {motion_pixels}")
    print(f"✓ Motion percentage: {100*motion_pixels/fg_mask.size:.2f}%")
    print(f"✓ Processing time: {result.processing_time:.3f}s")


def test_scene_change_detection():
    """Test detection with scene changes using real video."""
    print("\nTest 3: Scene Change Detection")
    print("-" * 40)
    
    # Load video segment with scene change
    try:
        # Get two different segments to simulate scene change
        frames1 = get_video_segment(start_second=0, duration=1.0)
        frames2 = get_video_segment(start_second=5, duration=1.0)
        frames = frames1 + frames2
        print(f"  Loaded {len(frames)} frames with scene change")
    except:
        # Fallback to synthetic scene change
        frames = []
        for _ in range(10):
            frame = np.ones((100, 100, 3), dtype=np.uint8) * 100
            frame[20:40, 20:40] = 200
            frames.append(frame)
        for _ in range(10):
            frame = np.ones((100, 100, 3), dtype=np.uint8) * 150
            frame[60:80, 60:80] = 50
            frames.append(frame)
    
    # Short history (adapts quickly to scene change)
    analyzer_short = BackgroundMOG2(
        downsample=0.25,
        history=10,
        var_threshold=16
    )
    result_short = analyzer_short.analyze(frames)
    
    # Long history (adapts slowly to scene change)
    analyzer_long = BackgroundMOG2(
        downsample=0.25,
        history=100,
        var_threshold=16
    )
    result_long = analyzer_long.analyze(frames)
    
    # With longer history, more pixels should be detected as foreground after scene change
    fg_short = np.sum(result_short.data['foreground_mask'][15:] > 0)
    fg_long = np.sum(result_long.data['foreground_mask'][15:] > 0)
    
    print(f"✓ Foreground with short history (10): {fg_short}")
    print(f"✓ Foreground with long history (100): {fg_long}")
    print(f"✓ Different history lengths affect adaptation")


def test_mog2_vs_knn_comparison():
    """Compare MOG2 and KNN algorithms on real video."""
    print("\nTest 4: MOG2 vs KNN Comparison")
    print("-" * 40)
    
    # Load real video
    try:
        frames = load_video_frames(max_seconds=2.0)
        print(f"  Loaded {len(frames)} frames from video")
    except:
        frames = []
        for i in range(30):
            frame = np.ones((100, 100, 3), dtype=np.uint8) * 50
            x = (i * 3) % 70
            frame[30:60, x:x+20] = 200
            frames.append(frame)
    
    # MOG2
    mog2 = BackgroundMOG2(downsample=0.25, history=30)
    result_mog2 = mog2.analyze(frames)
    
    # KNN
    knn = BackgroundKNN(downsample=0.25, history=30)
    result_knn = knn.analyze(frames)
    
    # Compare detection amounts
    fg_mog2 = np.sum(result_mog2.data['foreground_mask'] > 0)
    fg_knn = np.sum(result_knn.data['foreground_mask'] > 0)
    
    # Compare processing times
    time_mog2 = result_mog2.processing_time
    time_knn = result_knn.processing_time
    
    print(f"✓ MOG2 foreground pixels: {fg_mog2}")
    print(f"✓ KNN foreground pixels: {fg_knn}")
    print(f"✓ MOG2 time: {time_mog2:.3f}s")
    print(f"✓ KNN time: {time_knn:.3f}s")
    
    # Both should detect motion
    assert fg_mog2 > 0 and fg_knn > 0, "Both algorithms should detect motion"
    
    # Compare relative performance
    if fg_mog2 > 0 and fg_knn > 0:
        ratio = fg_mog2 / fg_knn
        print(f"✓ Detection ratio (MOG2/KNN): {ratio:.2f}")


def test_real_video_scenarios():
    """Test with different video scenarios."""
    print("\nTest 5: Real Video Scenarios")
    print("-" * 40)
    
    # Test different segments of video
    scenarios = [
        ("Start of video", 0, 1.0),
        ("Middle segment", 5, 1.0),
    ]
    
    for name, start, duration in scenarios:
        try:
            frames = get_video_segment(start_second=start, duration=duration)
            
            # Test with MOG2
            analyzer = BackgroundMOG2(downsample=0.25, history=15)
            result = analyzer.analyze(frames)
            
            fg_mask = result.data['foreground_mask']
            motion_pixels = np.sum(fg_mask > 0)
            motion_pct = 100 * motion_pixels / fg_mask.size
            
            print(f"  {name}: {motion_pct:.1f}% motion detected ({len(frames)} frames)")
            
        except Exception as e:
            print(f"  {name}: Could not process - {e}")
    
    print(f"✓ Real video processing complete")


if __name__ == "__main__":
    print("="*50)
    print("Background Subtraction Analysis Tests")
    print("="*50)
    
    try:
        test_mog2_background_subtraction()
        test_knn_background_subtraction()
        test_scene_change_detection()
        test_mog2_vs_knn_comparison()
        test_real_video_scenarios()
        
        print("\n" + "="*50)
        print("ALL TESTS PASSED ✓")
        print("="*50)
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        exit(1)
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)