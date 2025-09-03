"""
Smoke Test 01: VideoKurt Initialization and Basic Setup
Tests importing, initialization, and basic method availability.

Run: python -m videokurt.smoke_tests.videokurt.test_01_initialization
"""

import sys
from pathlib import Path

from videokurt.videokurt_new import VideoKurt, VideoLoadError, ConfigurationError


def test_import():
    """Test that VideoKurt can be imported."""
    print("✓ VideoKurt imported successfully")
    assert VideoKurt is not None
    assert VideoLoadError is not None
    assert ConfigurationError is not None


def test_initialization():
    """Test VideoKurt initialization."""
    vk = VideoKurt()
    print("✓ VideoKurt instance created")
    
    # Check initial state
    assert vk._analyses == {}
    assert vk._features == {}
    assert vk._mode == 'full'
    print("✓ Initial state correct")
    
    # Check default config
    assert vk._config['frame_step'] == 1
    assert vk._config['resolution_scale'] == 1.0
    assert vk._config['blur'] == False
    assert vk._config['blur_kernel_size'] == 13
    assert vk._config['process_chunks'] == 1
    assert vk._config['chunk_overlap'] == 30
    print("✓ Default configuration correct")


def test_list_methods():
    """Test listing methods work."""
    vk = VideoKurt()
    
    # Test list methods on empty instance
    assert vk.list_analyses() == []
    assert vk.list_features() == []
    print("✓ List methods work on empty instance")
    
    # Test available analyses list
    available = vk.list_available_analyses()
    assert isinstance(available, list)
    assert len(available) > 0
    assert 'frame_diff' in available
    print(f"✓ Found {len(available)} available analyses")
    
    # Test available features list
    features = vk.list_available_features()
    assert isinstance(features, list)
    # Features might be empty if not implemented yet
    print(f"✓ Found {len(features)} available features")


def test_clear_method():
    """Test clear method resets everything."""
    vk = VideoKurt()
    
    # Modify some settings
    vk._analyses['test'] = 'dummy'
    vk._features['test'] = 'dummy'
    vk._config['frame_step'] = 5
    vk._mode = 'features_only'
    
    # Clear
    vk.clear()
    
    # Check everything reset
    assert vk._analyses == {}
    assert vk._features == {}
    assert vk._config['frame_step'] == 1
    assert vk._mode == 'full'
    print("✓ Clear method resets all settings")


def test_repr():
    """Test string representation."""
    vk = VideoKurt()
    repr_str = repr(vk)
    
    assert 'VideoKurt' in repr_str
    assert 'analyses=' in repr_str
    assert 'features=' in repr_str
    assert 'config=' in repr_str
    assert 'mode=' in repr_str
    print("✓ String representation works")
    print(f"  {repr_str}")


def test_validate_empty():
    """Test validation on empty configuration."""
    vk = VideoKurt()
    issues = vk.validate()
    
    # Should have issues when nothing configured
    assert len(issues) > 0
    assert any('No analyses' in issue for issue in issues)
    print("✓ Validation detects empty configuration")


def test_convenience_function():
    """Test that convenience function is available."""
    from videokurt.videokurt_new import analyze_video
    
    assert analyze_video is not None
    print("✓ Convenience function available")


def main():
    """Run all initialization tests."""
    print("="*50)
    print("VideoKurt Smoke Test 01: Initialization")
    print("="*50)
    
    tests = [
        test_import,
        test_initialization,
        test_list_methods,
        test_clear_method,
        test_repr,
        test_validate_empty,
        test_convenience_function,
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
            return False
    
    print("\n" + "="*50)
    print("All initialization tests passed! ✓")
    print("="*50)
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)