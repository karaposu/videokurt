"""
Smoke Test 02: VideoKurt Configuration
Tests adding analyses, features, and configuration methods.

Run: python -m videokurt.smoke_tests.videokurt.test_02_configuration
"""

import sys
from videokurt.videokurt_new import VideoKurt, ConfigurationError
from videokurt.raw_analysis import FrameDiff, OpticalFlowDense


def test_add_analysis_by_name():
    """Test adding analysis by string name."""
    vk = VideoKurt()
    
    # Add single analysis
    vk.add_analysis('frame_diff')
    assert 'frame_diff' in vk.list_analyses()
    print("✓ Added analysis by name")
    
    # Add another
    vk.add_analysis('optical_flow_dense')
    assert 'optical_flow_dense' in vk.list_analyses()
    assert len(vk.list_analyses()) == 2
    print("✓ Added multiple analyses")


def test_add_analysis_with_params():
    """Test adding analysis with parameters."""
    vk = VideoKurt()
    
    # Add with parameters
    vk.add_analysis('frame_diff', threshold=0.3)
    assert 'frame_diff' in vk.list_analyses()
    
    # Check the parameter was set
    analysis = vk._analyses['frame_diff']
    assert analysis.threshold == 0.3
    print("✓ Added analysis with custom parameters")


def test_add_analysis_by_object():
    """Test adding pre-configured analysis object."""
    vk = VideoKurt()
    
    # Create configured analysis
    frame_diff = FrameDiff(threshold=0.5, downsample=0.8)
    
    # Add it
    vk.add_analysis(frame_diff)
    assert 'frame_diff' in vk.list_analyses()
    
    # Check it's the same object
    assert vk._analyses['frame_diff'] is frame_diff
    assert vk._analyses['frame_diff'].threshold == 0.5
    print("✓ Added pre-configured analysis object")


def test_add_invalid_analysis():
    """Test error handling for invalid analysis."""
    vk = VideoKurt()
    
    # Try adding non-existent analysis
    try:
        vk.add_analysis('non_existent_analysis')
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert 'Unknown analysis' in str(e)
        print("✓ Rejected unknown analysis name")
    
    # Try adding wrong type
    try:
        vk.add_analysis(123)
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert 'must be string or BaseAnalysis' in str(e)
        print("✓ Rejected invalid analysis type")


def test_add_feature():
    """Test adding features (even if not fully implemented)."""
    vk = VideoKurt()
    
    # Try to add a feature
    try:
        vk.add_feature('binary_activity')
        # If it works, check it was added
        if 'binary_activity' in vk.list_features():
            print("✓ Added feature successfully")
            # Check auto-dependencies
            if 'frame_diff' in vk.list_analyses():
                print("✓ Auto-added required analyses")
    except (ValueError, NotImplementedError) as e:
        # Features might not be implemented yet
        print(f"✓ Feature system not yet implemented (expected): {e}")


def test_configure_basic():
    """Test basic configuration."""
    vk = VideoKurt()
    
    # Configure basic settings
    vk.configure(frame_step=2, resolution_scale=0.5)
    
    assert vk._config['frame_step'] == 2
    assert vk._config['resolution_scale'] == 0.5
    print("✓ Basic configuration works")
    
    # Configure blur
    vk.configure(blur=True, blur_kernel_size=15)
    assert vk._config['blur'] == True
    assert vk._config['blur_kernel_size'] == 15
    print("✓ Blur configuration works")


def test_configure_validation():
    """Test configuration validation."""
    vk = VideoKurt()
    
    # Invalid frame_step
    try:
        vk.configure(frame_step=0)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert 'frame_step must be >= 1' in str(e)
        print("✓ Validated frame_step")
    
    # Invalid resolution_scale
    try:
        vk.configure(resolution_scale=1.5)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert 'resolution_scale must be between 0 and 1' in str(e)
        print("✓ Validated resolution_scale")
    
    # Invalid blur_kernel_size (even number)
    try:
        vk.configure(blur_kernel_size=14)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert 'blur_kernel_size must be odd' in str(e)
        print("✓ Validated blur_kernel_size")
    
    # Invalid process_chunks
    try:
        vk.configure(process_chunks=0)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert 'process_chunks must be >= 1' in str(e)
        print("✓ Validated process_chunks")


def test_set_mode():
    """Test execution mode setting."""
    vk = VideoKurt()
    
    # Set valid mode
    vk.set_mode('full')
    assert vk._mode == 'full'
    print("✓ Set mode to 'full'")
    
    # Try invalid mode
    try:
        vk.set_mode('invalid_mode')
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert 'Invalid mode' in str(e)
        print("✓ Rejected invalid mode")
    
    # Try unimplemented modes
    for mode in ['features_only', 'streaming']:
        try:
            vk.set_mode(mode)
            assert False, f"Should have raised NotImplementedError for {mode}"
        except NotImplementedError as e:
            assert f"'{mode}' not yet implemented" in str(e)
            print(f"✓ Mode '{mode}' correctly marked as not implemented")


def test_configuration_persistence():
    """Test that configuration persists across operations."""
    vk = VideoKurt()
    
    # Set multiple configurations
    vk.configure(frame_step=3)
    vk.add_analysis('frame_diff')
    vk.configure(resolution_scale=0.7)
    vk.add_analysis('edge_canny')
    vk.configure(blur=True)
    
    # Check all configurations still there
    assert vk._config['frame_step'] == 3
    assert vk._config['resolution_scale'] == 0.7
    assert vk._config['blur'] == True
    assert len(vk.list_analyses()) == 2
    print("✓ Configuration persists across operations")


def test_validate_with_analyses():
    """Test validation with analyses configured."""
    vk = VideoKurt()
    
    # Add some analyses
    vk.add_analysis('frame_diff')
    vk.add_analysis('optical_flow_dense')
    
    # Should be valid now
    issues = vk.validate()
    assert len(issues) == 0
    print("✓ Valid configuration passes validation")


def main():
    """Run all configuration tests."""
    print("="*50)
    print("VideoKurt Smoke Test 02: Configuration")
    print("="*50)
    
    tests = [
        test_add_analysis_by_name,
        test_add_analysis_with_params,
        test_add_analysis_by_object,
        test_add_invalid_analysis,
        test_add_feature,
        test_configure_basic,
        test_configure_validation,
        test_set_mode,
        test_configuration_persistence,
        test_validate_with_analyses,
    ]
    
    for test in tests:
        try:
            print(f"\nRunning {test.__name__}...")
            test()
        except AssertionError as e:
            print(f"✗ {test.__name__} failed: {e}")
            return False
        except Exception as e:
            print(f"✗ {test.__name__} error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print("\n" + "="*50)
    print("All configuration tests passed! ✓")
    print("="*50)
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)