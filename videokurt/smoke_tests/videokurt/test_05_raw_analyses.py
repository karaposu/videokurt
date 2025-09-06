"""
Smoke Test 05: VideoKurt Raw Analyses Integration
Tests all raw analyses through the VideoKurt interface.

Run: python -m videokurt.smoke_tests.videokurt.test_05_raw_analyses
"""

import sys
import numpy as np
import cv2
from pathlib import Path
from videokurt.videokurt import VideoKurt
from videokurt.samplemaker import create_frames_with_pattern


def test_all_analyses_available():
    """Test that all raw analyses are available."""
    print("Testing analysis availability...")
    
    vk = VideoKurt()
    available = vk.list_available_analyses()
    
    expected = [
        'frame_diff', 'edge_canny', 'frame_diff_advanced', 'contour_detection',
        'background_mog2', 'background_knn', 'optical_flow_sparse', 'optical_flow_dense',
        'motion_heatmap', 'frequency_fft', 'flow_hsv_viz', 'color_histogram',
        'dct_transform', 'texture_descriptors'
    ]
    
    for analysis in expected:
        assert analysis in available, f"Missing analysis: {analysis}"
    
    print(f"✓ All {len(expected)} analyses available")
    print(f"  Available: {', '.join(available)}")
    
    return True


def test_add_analyses_by_name():
    """Test adding analyses by string name."""
    print("\nTesting adding analyses by name...")
    
    vk = VideoKurt()
    
    # Add various analyses
    analyses_to_add = ['frame_diff', 'optical_flow_dense', 'edge_canny', 'motion_heatmap']
    
    for analysis in analyses_to_add:
        vk.add_analysis(analysis)
    
    configured = vk.list_analyses()
    assert len(configured) == len(analyses_to_add), f"Expected {len(analyses_to_add)} analyses, got {len(configured)}"
    
    for analysis in analyses_to_add:
        assert analysis in configured, f"Analysis {analysis} not in configured list"
    
    print(f"✓ Added {len(analyses_to_add)} analyses by name")
    print(f"  Configured: {', '.join(configured)}")
    
    return True


def test_add_analyses_with_params():
    """Test adding analyses with custom parameters."""
    print("\nTesting adding analyses with parameters...")
    
    vk = VideoKurt()
    
    # Add analyses with custom parameters
    vk.add_analysis('frame_diff', threshold=0.2, downsample=0.5)
    vk.add_analysis('motion_heatmap', decay_factor=0.9, snapshot_interval=10)
    vk.add_analysis('background_knn', history=100, detect_shadows=True)
    
    configured = vk.list_analyses()
    assert len(configured) == 3, f"Expected 3 analyses, got {len(configured)}"
    
    # Verify the analyses were added with correct names
    assert 'frame_diff' in configured
    assert 'motion_heatmap' in configured
    assert 'background_knn' in configured
    
    print(f"✓ Added {len(configured)} analyses with custom parameters")
    
    return True


def test_run_single_analysis():
    """Test running a single analysis on synthetic video."""
    print("\nTesting single analysis execution...")
    
    # Create test video
    frames = create_frames_with_pattern(num_frames=20, width=200, height=150, pattern='moving_circle')
    test_video = Path('/Users/ns/Desktop/projects/videokurt/test_single_analysis.mp4')
    VideoKurt.save_video(frames, test_video, fps=15)
    
    # Run single analysis
    vk = VideoKurt()
    vk.add_analysis('frame_diff')
    vk.configure(frame_step=1, resolution_scale=0.5)
    
    results = vk.analyze(test_video)
    
    # Verify results
    assert results is not None, "Results should not be None"
    assert 'frame_diff' in results.analyses, "frame_diff should be in results"
    
    frame_diff_result = results.analyses['frame_diff']
    assert frame_diff_result.method == 'frame_diff'
    assert 'pixel_diff' in frame_diff_result.data
    
    print(f"✓ Executed frame_diff analysis")
    print(f"  Output shape: {frame_diff_result.data['pixel_diff'].shape}")
    print(f"  Processing time: {frame_diff_result.processing_time:.3f}s")
    
    # Clean up
    test_video.unlink(missing_ok=True)
    
    return True


def test_run_multiple_analyses():
    """Test running multiple analyses together."""
    print("\nTesting multiple analyses execution...")
    
    # Create test video
    frames = create_frames_with_pattern(num_frames=15, width=150, height=100, pattern='moving_line')
    test_video = Path('/Users/ns/Desktop/projects/videokurt/test_multiple_analyses.mp4')
    VideoKurt.save_video(frames, test_video, fps=10)
    
    # Configure multiple analyses
    vk = VideoKurt()
    analyses = ['frame_diff', 'edge_canny', 'optical_flow_dense', 'color_histogram']
    for analysis in analyses:
        vk.add_analysis(analysis)
    
    vk.configure(frame_step=2, resolution_scale=0.5)
    
    # Run analyses
    results = vk.analyze(test_video)
    
    # Verify all analyses ran
    assert len(results.analyses) == len(analyses), f"Expected {len(analyses)} results"
    
    print(f"✓ Executed {len(analyses)} analyses")
    for name, result in results.analyses.items():
        print(f"  - {name}: {list(result.data.keys())}, time={result.processing_time:.3f}s")
    
    # Clean up
    test_video.unlink(missing_ok=True)
    
    return True


def test_memory_heavy_analyses():
    """Test memory-intensive analyses with downsampling."""
    print("\nTesting memory-heavy analyses...")
    
    # Create test video with enough frames for FFT (needs 64+)
    frames = create_frames_with_pattern(num_frames=70, width=200, height=150, pattern='gradient')
    test_video = Path('/Users/ns/Desktop/projects/videokurt/test_memory_heavy.mp4')
    VideoKurt.save_video(frames, test_video, fps=10)
    
    vk = VideoKurt()
    
    # Add memory-heavy analyses
    vk.add_analysis('motion_heatmap')  # Default downsample=0.25
    vk.add_analysis('optical_flow_dense')
    vk.add_analysis('frequency_fft')
    
    # Use aggressive downsampling
    vk.configure(frame_step=1, resolution_scale=0.25)  # No frame_step to keep 70 frames for FFT
    
    results = vk.analyze(test_video)
    
    # Verify results
    assert 'motion_heatmap' in results.analyses
    assert 'optical_flow_dense' in results.analyses
    assert 'frequency_fft' in results.analyses
    
    print("✓ Memory-heavy analyses completed with downsampling")
    print(f"  Total processing time: {results.elapsed_time:.3f}s")
    
    # Clean up
    test_video.unlink(missing_ok=True)
    
    return True


def test_analysis_with_real_video():
    """Test analyses with real video if available."""
    print("\nTesting with real video...")
    
    video_path = Path('/Users/ns/Desktop/projects/videokurt/sample_recording.MP4')
    
    if not video_path.exists():
        print("⚠ Sample video not found, skipping real video test")
        return True
    
    vk = VideoKurt()
    
    # Add a variety of analyses
    vk.add_analysis('frame_diff')
    vk.add_analysis('edge_canny')
    vk.add_analysis('background_mog2')
    vk.add_analysis('optical_flow_sparse')
    vk.add_analysis('color_histogram')
    
    # Configure for reasonable performance
    vk.configure(
        frame_step=10,
        resolution_scale=0.25,
        blur=False
    )
    
    # Analyze first 5 seconds (assuming 30fps = 150 frames, with frame_step=10 = 15 frames)
    results = vk.analyze(video_path)
    
    print(f"✓ Analyzed real video")
    print(f"  Video dimensions: {results.dimensions}")
    print(f"  Frames processed: {results.frame_count}")
    print(f"  Total time: {results.elapsed_time:.3f}s")
    
    # Show results for each analysis
    for name, result in results.analyses.items():
        data_info = {k: str(v.shape) if hasattr(v, 'shape') else type(v).__name__ 
                     for k, v in result.data.items()}
        print(f"  - {name}: {data_info}")
    
    return True


def test_error_handling():
    """Test error handling for invalid analyses."""
    print("\nTesting error handling...")
    
    vk = VideoKurt()
    
    # Test invalid analysis name
    try:
        vk.add_analysis('non_existent_analysis')
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert 'Unknown analysis' in str(e)
        print("✓ Correctly rejected invalid analysis name")
    
    # Test invalid parameter type
    try:
        vk.add_analysis(123)  # Invalid type
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert 'must be string or BaseAnalysis' in str(e)
        print("✓ Correctly rejected invalid parameter type")
    
    # Test configuration validation
    try:
        vk.configure(frame_step=0)  # Invalid value
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert 'frame_step must be >= 1' in str(e)
        print("✓ Correctly validated frame_step")
    
    try:
        vk.configure(resolution_scale=2.0)  # Invalid value
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert 'resolution_scale must be between 0 and 1' in str(e)
        print("✓ Correctly validated resolution_scale")
    
    return True


def test_clear_and_reconfigure():
    """Test clearing configuration and reconfiguring."""
    print("\nTesting clear and reconfigure...")
    
    vk = VideoKurt()
    
    # Add some analyses
    vk.add_analysis('frame_diff')
    vk.add_analysis('edge_canny')
    vk.configure(frame_step=5, resolution_scale=0.5)
    
    assert len(vk.list_analyses()) == 2, "Should have 2 analyses"
    
    # Clear everything
    vk.clear()
    
    assert len(vk.list_analyses()) == 0, "Should have no analyses after clear"
    assert vk._config['frame_step'] == 1, "frame_step should be reset to 1"
    assert vk._config['resolution_scale'] == 1.0, "resolution_scale should be reset to 1.0"
    
    # Reconfigure
    vk.add_analysis('optical_flow_dense')
    vk.configure(frame_step=2)
    
    assert len(vk.list_analyses()) == 1, "Should have 1 analysis after reconfigure"
    assert vk._config['frame_step'] == 2, "frame_step should be 2"
    
    print("✓ Clear and reconfigure working correctly")
    
    return True


def test_analysis_combinations():
    """Test specific combinations of analyses that work well together."""
    print("\nTesting analysis combinations...")
    
    # Create test video
    frames = create_frames_with_pattern(num_frames=20, width=100, height=100, pattern='moving_circle')
    test_video = Path('/Users/ns/Desktop/projects/videokurt/test_combinations.mp4')
    VideoKurt.save_video(frames, test_video, fps=10)
    
    # Test 1: Motion detection combo
    vk1 = VideoKurt()
    vk1.add_analysis('frame_diff')
    vk1.add_analysis('optical_flow_dense')
    vk1.add_analysis('motion_heatmap')
    vk1.configure(resolution_scale=0.5)
    
    results1 = vk1.analyze(test_video)
    assert len(results1.analyses) == 3, "Motion combo should have 3 analyses"
    print("✓ Motion detection combo completed")
    
    # Test 2: Edge and contour combo
    vk2 = VideoKurt()
    vk2.add_analysis('edge_canny')
    vk2.add_analysis('contour_detection')
    vk2.configure(resolution_scale=0.5)
    
    results2 = vk2.analyze(test_video)
    assert len(results2.analyses) == 2, "Edge combo should have 2 analyses"
    print("✓ Edge and contour combo completed")
    
    # Test 3: Background subtraction combo
    vk3 = VideoKurt()
    vk3.add_analysis('background_knn')
    vk3.add_analysis('background_mog2')
    vk3.configure(resolution_scale=0.5)
    
    results3 = vk3.analyze(test_video)
    assert len(results3.analyses) == 2, "Background combo should have 2 analyses"
    print("✓ Background subtraction combo completed")
    
    # Clean up
    test_video.unlink(missing_ok=True)
    
    return True


def main():
    """Run all raw analyses tests."""
    print("="*50)
    print("VideoKurt Smoke Test 05: Raw Analyses")
    print("="*50)
    
    tests = [
        ("All analyses available", test_all_analyses_available),
        ("Add analyses by name", test_add_analyses_by_name),
        ("Add analyses with params", test_add_analyses_with_params),
        ("Run single analysis", test_run_single_analysis),
        ("Run multiple analyses", test_run_multiple_analyses),
        ("Memory-heavy analyses", test_memory_heavy_analyses),
        ("Analysis with real video", test_analysis_with_real_video),
        ("Error handling", test_error_handling),
        ("Clear and reconfigure", test_clear_and_reconfigure),
        ("Analysis combinations", test_analysis_combinations),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*30}")
            print(f"Running: {test_name}")
            print('='*30)
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ {test_name} error: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("Test Summary:")
    print("="*50)
    
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(success for _, success in results)
    
    if all_passed:
        print("\n" + "="*50)
        print("All raw analyses tests passed! ✓")
        print("="*50)
    else:
        print("\n" + "="*50)
        print("Some tests failed. Check the output above.")
        print("="*50)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)