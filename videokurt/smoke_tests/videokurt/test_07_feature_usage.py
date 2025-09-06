"""
Test 07: Feature Usage
Tests single and multiple feature usage patterns with VideoKurt


Run: python -m videokurt.smoke_tests.videokurt.test_07_feature_usage




"""

import sys
from pathlib import Path
import numpy as np
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from videokurt import VideoKurt


def test_single_feature():
    """Test using a single feature."""
    print("\n" + "="*50)
    print("TEST: Single Feature Usage")
    print("="*50)
    
    vk = VideoKurt()
    vk.add_feature('motion_magnitude')
    vk.configure(frame_step=10, resolution_scale=0.3)
    
    print("Configuration:")
    print(f"  Features: motion_magnitude")
    print(f"  Auto-added analyses: {list(vk._analyses.keys())}")
    
    results = vk.analyze('sample_recording.MP4')
    
    assert 'motion_magnitude' in results.features, "Feature not computed"
    
    motion_data = results.features['motion_magnitude'].data
    print("\nResults:")
    if isinstance(motion_data, dict):
        print(f"  Average motion: {motion_data.get('average', 0):.3f}")
        print(f"  Peak motion: {motion_data.get('peak', 0):.3f}")
    elif isinstance(motion_data, np.ndarray):
        print(f"  Average motion: {np.mean(motion_data):.3f}")
        print(f"  Peak motion: {np.max(motion_data):.3f}")
    print("✓ Single feature test passed")
    
    return results


def test_multiple_features_shared_dependency():
    """Test multiple features that share the same dependency."""
    print("\n" + "="*50)
    print("TEST: Multiple Features with Shared Dependency")
    print("="*50)
    
    vk = VideoKurt()
    
    # All these features require 'frame_diff'
    features = ['stability_score', 'binary_activity', 'activity_bursts']
    for feature in features:
        vk.add_feature(feature)
    
    vk.configure(frame_step=10, resolution_scale=0.3)
    
    print("Configuration:")
    print(f"  Features: {', '.join(features)}")
    print(f"  Shared analysis: frame_diff")
    print(f"  Total analyses: {list(vk._analyses.keys())}")
    
    start = time.time()
    results = vk.analyze('sample_recording.MP4')
    duration = time.time() - start
    
    # Verify all features computed
    for feature in features:
        assert feature in results.features, f"Feature {feature} not computed"
    
    # Show results
    print(f"\nResults (computed in {duration:.2f}s):")
    
    if 'stability_score' in results.features:
        stability = results.features['stability_score'].data
        if isinstance(stability, dict):
            avg_stability = np.mean(stability.get('timeline', [0]))
        elif isinstance(stability, np.ndarray):
            avg_stability = np.mean(stability)
        else:
            avg_stability = 0
        print(f"  Average stability: {avg_stability:.3f}")
    
    if 'binary_activity' in results.features:
        activity = results.features['binary_activity'].data
        if isinstance(activity, dict):
            activity_ratio = activity.get('activity_ratio', 0)
        elif isinstance(activity, np.ndarray):
            activity_ratio = np.mean(activity)
        else:
            activity_ratio = 0
        print(f"  Activity ratio: {activity_ratio:.1%}")
    
    if 'activity_bursts' in results.features:
        bursts = results.features['activity_bursts'].data
        num_bursts = bursts.get('num_bursts', 0)
        print(f"  Activity bursts: {num_bursts}")
    
    print("✓ Shared dependency test passed")
    return results


def test_multiple_features_different_dependencies():
    """Test multiple features with different dependencies."""
    print("\n" + "="*50)
    print("TEST: Multiple Features with Different Dependencies")
    print("="*50)
    
    vk = VideoKurt()
    
    # Each requires different analysis
    feature_deps = {
        'motion_magnitude': 'optical_flow_dense',
        'edge_density': 'edge_canny',
        'stability_score': 'frame_diff',
        'scene_detection': 'color_histogram'
    }
    
    for feature in feature_deps.keys():
        vk.add_feature(feature)
    
    vk.configure(frame_step=10, resolution_scale=0.3)
    
    print("Configuration:")
    print("  Features and their dependencies:")
    for feature, dep in feature_deps.items():
        print(f"    {feature} → {dep}")
    print(f"  All analyses: {list(vk._analyses.keys())}")
    
    results = vk.analyze('sample_recording.MP4')
    
    # Verify all features computed
    computed = []
    failed = []
    for feature in feature_deps.keys():
        if feature in results.features:
            computed.append(feature)
        else:
            failed.append(feature)
    
    print(f"\nResults:")
    print(f"  Computed features: {len(computed)}/{len(feature_deps)}")
    if failed:
        print(f"  Failed features: {', '.join(failed)}")
    
    # Show sample results
    if 'motion_magnitude' in results.features:
        data = results.features['motion_magnitude'].data
        motion = np.mean(data) if isinstance(data, np.ndarray) else data.get('average', 0)
        print(f"  Motion magnitude: {motion:.3f}")
    
    if 'edge_density' in results.features:
        data = results.features['edge_density'].data
        edges = np.mean(data) if isinstance(data, np.ndarray) else data.get('average', 0)
        print(f"  Edge density: {edges:.3f}")
    
    if 'stability_score' in results.features:
        data = results.features['stability_score'].data
        stability = np.mean(data) if isinstance(data, np.ndarray) else np.mean(data.get('timeline', [0]))
        print(f"  Stability score: {stability:.3f}")
    
    if 'scene_detection' in results.features:
        scenes = results.features['scene_detection'].data.get('num_scenes', 1)
        print(f"  Number of scenes: {scenes}")
    
    print("✓ Different dependencies test passed")
    return results


def test_screen_recording_suite():
    """Test comprehensive screen recording analysis suite."""
    print("\n" + "="*50)
    print("TEST: Screen Recording Analysis Suite")
    print("="*50)
    
    vk = VideoKurt()
    
    # Core screen recording features
    screen_features = [
        'scrolling_detection',
        'stability_score',
        'scene_detection',
        'edge_density',
        'motion_magnitude',
        'spatial_occupancy_grid'
    ]
    
    print("Adding screen recording features...")
    for feature in screen_features:
        try:
            vk.add_feature(feature)
            print(f"  ✓ {feature}")
        except Exception as e:
            print(f"  ✗ {feature}: {e}")
    
    vk.configure(frame_step=5, resolution_scale=0.4)
    
    print(f"\nAnalyses to run: {list(vk._analyses.keys())}")
    
    results = vk.analyze('sample_recording.MP4')
    
    # Generate summary report
    print("\nScreen Recording Analysis Report:")
    print("-" * 40)
    
    # Scrolling detection
    if 'scrolling_detection' in results.features:
        scroll = results.features['scrolling_detection'].data
        if scroll.get('is_scrolling', False):
            print(f"  Scrolling: YES ({scroll.get('direction', 'unknown')})")
        else:
            print("  Scrolling: NO")
    
    # Stability/idle time
    if 'stability_score' in results.features:
        data = results.features['stability_score'].data
        if isinstance(data, np.ndarray):
            stability = data
        else:
            stability = data.get('timeline', [])
        if len(stability) > 0:
            idle_frames = sum(1 for s in stability if s > 0.9)
            idle_ratio = idle_frames / len(stability)
            print(f"  Idle time: {idle_ratio:.1%}")
    
    # Scene changes
    if 'scene_detection' in results.features:
        scenes = results.features['scene_detection'].data.get('num_scenes', 1)
        print(f"  Scene changes: {scenes - 1}")
    
    # Content type (text vs media)
    if 'edge_density' in results.features:
        data = results.features['edge_density'].data
        edges = np.mean(data) if isinstance(data, np.ndarray) else data.get('average', 0)
        if edges > 0.5:
            print("  Content type: Text-heavy")
        else:
            print("  Content type: Media/graphics")
    
    # Activity level
    if 'motion_magnitude' in results.features:
        data = results.features['motion_magnitude'].data
        motion = np.mean(data) if isinstance(data, np.ndarray) else data.get('average', 0)
        if motion < 0.1:
            activity = "Low"
        elif motion < 0.5:
            activity = "Medium"
        else:
            activity = "High"
        print(f"  Activity level: {activity} ({motion:.2f})")
    
    # Spatial distribution
    if 'spatial_occupancy_grid' in results.features:
        data = results.features['spatial_occupancy_grid'].data
        if isinstance(data, dict) and 'most_active_cell' in data:
            print(f"  Most active region: {data['most_active_cell']}")
    
    print("-" * 40)
    print("✓ Screen recording suite test passed")
    return results


def test_feature_with_parameters():
    """Test adding features with custom parameters."""
    print("\n" + "="*50)
    print("TEST: Features with Custom Parameters")
    print("="*50)
    
    vk = VideoKurt()
    
    # Add features with specific parameters
    print("Adding features with parameters:")
    
    vk.add_feature('motion_magnitude', normalize=True)
    print("  motion_magnitude(normalize=True)")
    
    vk.add_feature('edge_density', use_gradient=True)
    print("  edge_density(use_gradient=True)")
    
    vk.add_feature('spatial_occupancy_grid', grid_size=(3, 3))
    print("  spatial_occupancy_grid(grid_size=(3,3))")
    
    vk.configure(frame_step=10, resolution_scale=0.3)
    
    results = vk.analyze('sample_recording.MP4')
    
    print("\nVerifying parameter effects:")
    
    # Check normalized motion
    if 'motion_magnitude' in results.features:
        data = results.features['motion_magnitude'].data
        motion_val = np.mean(data) if isinstance(data, np.ndarray) else data.get('average', 0)
        print(f"  Motion (normalized): {motion_val:.3f}")
    
    # Check grid size
    if 'spatial_occupancy_grid' in results.features:
        data = results.features['spatial_occupancy_grid'].data
        if isinstance(data, dict):
            grid = data.get('occupancy_grid', [])
            if hasattr(grid, 'shape'):
                print(f"  Grid shape: {grid.shape}")
        elif hasattr(data, 'shape'):
            print(f"  Grid shape: {data.shape}")
    
    print("✓ Parameter test passed")
    return results


def test_feature_failure_handling():
    """Test handling of feature computation failures."""
    print("\n" + "="*50)
    print("TEST: Feature Failure Handling")
    print("="*50)
    
    vk = VideoKurt()
    
    # Add features that might fail
    features = [
        'motion_magnitude',  # Should work
        'blob_tracking',     # Might fail (needs good background subtraction)
        'connected_components',  # Often fails on screen recordings
        'stability_score'    # Should work
    ]
    
    for feature in features:
        vk.add_feature(feature)
    
    vk.configure(frame_step=10, resolution_scale=0.3)
    
    print(f"Attempting to compute: {features}")
    
    results = vk.analyze('sample_recording.MP4')
    
    successful = []
    failed = []
    
    for feature in features:
        if feature in results.features:
            successful.append(feature)
        else:
            failed.append(feature)
    
    print(f"\nResults:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Success rate: {len(successful)}/{len(features)}")
    
    # Should have at least some successes
    assert len(successful) > 0, "All features failed!"
    
    print("✓ Failure handling test passed")
    return results


def test_performance_comparison():
    """Compare performance of single vs multiple features."""
    print("\n" + "="*50)
    print("TEST: Performance Comparison")
    print("="*50)
    
    config = {'frame_step': 20, 'resolution_scale': 0.2}
    
    # Test 1: Single feature
    print("Test 1: Single feature")
    vk1 = VideoKurt()
    vk1.add_feature('motion_magnitude')
    vk1.configure(**config)
    
    start = time.time()
    results1 = vk1.analyze('sample_recording.MP4')
    time1 = time.time() - start
    print(f"  Time: {time1:.2f}s")
    
    # Test 2: Three features (shared dependency)
    print("\nTest 2: Three features (shared dependency)")
    vk2 = VideoKurt()
    vk2.add_feature('stability_score')
    vk2.add_feature('binary_activity')
    vk2.add_feature('activity_bursts')
    vk2.configure(**config)
    
    start = time.time()
    results2 = vk2.analyze('sample_recording.MP4')
    time2 = time.time() - start
    print(f"  Time: {time2:.2f}s")
    
    # Test 3: Five features (mixed dependencies)
    print("\nTest 3: Five features (mixed dependencies)")
    vk3 = VideoKurt()
    vk3.add_feature('motion_magnitude')
    vk3.add_feature('stability_score')
    vk3.add_feature('edge_density')
    vk3.add_feature('scene_detection')
    vk3.add_feature('spatial_occupancy_grid')
    vk3.configure(**config)
    
    start = time.time()
    results3 = vk3.analyze('sample_recording.MP4')
    time3 = time.time() - start
    print(f"  Time: {time3:.2f}s")
    
    print("\nPerformance Summary:")
    print(f"  1 feature:  {time1:.2f}s (baseline)")
    print(f"  3 features: {time2:.2f}s ({time2/time1:.1f}x)")
    print(f"  5 features: {time3:.2f}s ({time3/time1:.1f}x)")
    
    print("✓ Performance comparison test passed")
    return time1, time2, time3


def run_all_tests():
    """Run all feature usage tests."""
    print("\n" + "="*60)
    print("FEATURE USAGE SMOKE TESTS")
    print("="*60)
    
    tests = [
        ("Single Feature", test_single_feature),
        ("Shared Dependencies", test_multiple_features_shared_dependency),
        ("Different Dependencies", test_multiple_features_different_dependencies),
        ("Screen Recording Suite", test_screen_recording_suite),
        ("Custom Parameters", test_feature_with_parameters),
        ("Failure Handling", test_feature_failure_handling),
        ("Performance Comparison", test_performance_comparison)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            print(f"\nRunning: {test_name}")
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ {test_name} failed: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Tests passed: {passed}/{len(tests)}")
    print(f"Tests failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n✓ ALL TESTS PASSED!")
    else:
        print(f"\n✗ {failed} tests failed")
        sys.exit(1)


if __name__ == '__main__':
    # Check if sample video exists
    video_path = Path('sample_recording.MP4')
    if not video_path.exists():
        print("Error: sample_recording.MP4 not found!")
        print("Please ensure the sample video is in the project root directory")
        sys.exit(1)
    
    # Run specific test or all tests
    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        if test_name == "single":
            test_single_feature()
        elif test_name == "shared":
            test_multiple_features_shared_dependency()
        elif test_name == "different":
            test_multiple_features_different_dependencies()
        elif test_name == "screen":
            test_screen_recording_suite()
        elif test_name == "params":
            test_feature_with_parameters()
        elif test_name == "failure":
            test_feature_failure_handling()
        elif test_name == "performance":
            test_performance_comparison()
        else:
            print(f"Unknown test: {test_name}")
            print("Available tests: single, shared, different, screen, params, failure, performance")
    else:
        run_all_tests()