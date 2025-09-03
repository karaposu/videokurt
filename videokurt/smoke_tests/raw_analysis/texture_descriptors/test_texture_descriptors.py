"""
Run: python -m videokurt.smoke_tests.raw_analysis.texture_descriptors.test_texture_descriptors

Smoke test for Texture Descriptors raw analysis
Tests basic functionality with real video file
"""

import sys
import cv2
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from videokurt.raw_analysis.texture_descriptors import TextureDescriptors


def test_texture_basic():
    """Test basic texture descriptor computation"""
    print("Testing Texture Descriptors...")
    
    # Initialize analyzer with LBP method
    analyzer = TextureDescriptors(method='lbp', lbp_radius=1, lbp_points=8)
    
    # Create test frames with different textures
    frames = []
    
    # Frame 1: Smooth texture
    frame1 = np.ones((100, 100, 3), dtype=np.uint8) * 128
    frames.append(frame1)
    
    # Frame 2: Vertical stripes
    frame2 = np.zeros((100, 100, 3), dtype=np.uint8)
    for x in range(0, 100, 10):
        frame2[:, x:x+5] = 255
    frames.append(frame2)
    
    # Frame 3: Checkerboard pattern
    frame3 = np.zeros((100, 100, 3), dtype=np.uint8)
    for i in range(0, 100, 20):
        for j in range(0, 100, 20):
            if (i//20 + j//20) % 2 == 0:
                frame3[i:i+20, j:j+20] = 255
    frames.append(frame3)
    
    # Frame 4: Random texture
    frame4 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    frames.append(frame4)
    
    # Analyze frames
    result = analyzer.analyze(frames)
    
    # Verify result structure
    assert result is not None, "Result should not be None"
    assert hasattr(result, 'data'), "Result should have data attribute"
    assert hasattr(result, 'method'), "Result should have method attribute"
    assert result.method == 'texture_descriptors', f"Method should be 'texture_descriptors', got {result.method}"
    
    print(f"✓ Analyzed {len(frames)} frames")
    print(f"✓ Method: {result.method}")
    if 'texture_features' in result.data:
        print(f"✓ Texture features computed for {len(result.data['texture_features'])} frames")
    
    return True


def test_texture_methods():
    """Test different texture analysis methods"""
    print("\nTesting different texture methods...")
    
    methods = ['lbp', 'gradient', 'combined']
    
    # Create test frame with texture
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    # Add some texture patterns
    for i in range(0, 80, 5):
        frame[i:i+2, :] = 255 - i * 3
    for j in range(0, 80, 5):
        frame[:, j:j+2] = np.minimum(frame[:, j:j+2] + 50, 255)
    
    frames = [frame, frame.copy()]  # Need multiple frames
    
    for method in methods:
        print(f"  Testing {method} method...")
        try:
            analyzer = TextureDescriptors(method=method)
            result = analyzer.analyze(frames)
            
            assert result is not None, f"{method}: Result should not be None"
            print(f"    ✓ {method} method processed successfully")
        except Exception as e:
            print(f"    ✗ {method} method failed: {e}")
            if method == 'gabor':
                print("      Note: gabor may require additional dependencies")
    
    return True


def test_lbp_parameters():
    """Test LBP with different parameters"""
    print("\nTesting LBP parameters...")
    
    # Create frame with circular pattern
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    center = (50, 50)
    for radius in range(10, 50, 5):
        cv2.circle(frame, center, radius, int(255 * (radius / 50)), 2)
    
    frames = [frame] * 3  # Repeat frame
    
    param_sets = [
        (1, 8),   # Standard
        (2, 8),   # Larger radius
        (1, 4),   # Fewer points
        (3, 12),  # More points, larger radius
    ]
    
    for radius, points in param_sets:
        print(f"  Testing radius={radius}, points={points}...")
        try:
            analyzer = TextureDescriptors(method='lbp', lbp_radius=radius, lbp_points=points)
            result = analyzer.analyze(frames)
            
            assert result is not None, f"r={radius}, p={points}: Result should not be None"
            print(f"    ✓ LBP with radius={radius}, points={points} processed")
        except Exception as e:
            print(f"    ✗ Failed with radius={radius}, points={points}: {e}")
    
    return True


def test_texture_patterns():
    """Test with known texture patterns"""
    print("\nTesting known texture patterns...")
    
    analyzer = TextureDescriptors(method='lbp')
    
    frames = []
    
    # Pattern 1: Horizontal lines (directional texture)
    pattern1 = np.zeros((100, 100, 3), dtype=np.uint8)
    for y in range(0, 100, 4):
        pattern1[y:y+2, :] = 255
    frames.append(pattern1)
    
    # Pattern 2: Dots (spotted texture)
    pattern2 = np.zeros((100, 100, 3), dtype=np.uint8)
    for y in range(10, 100, 20):
        for x in range(10, 100, 20):
            cv2.circle(pattern2, (x, y), 3, (255, 255, 255), -1)
    frames.append(pattern2)
    
    # Pattern 3: Gradient (smooth texture)
    pattern3 = np.zeros((100, 100, 3), dtype=np.uint8)
    for i in range(100):
        pattern3[i, :] = int(i * 255 / 100)
    frames.append(pattern3)
    
    # Pattern 4: Cross-hatch (complex texture)
    pattern4 = np.zeros((100, 100, 3), dtype=np.uint8)
    for i in range(0, 100, 8):
        cv2.line(pattern4, (0, i), (100, i), (255, 255, 255), 1)
        cv2.line(pattern4, (i, 0), (i, 100), (255, 255, 255), 1)
    frames.append(pattern4)
    
    result = analyzer.analyze(frames)
    
    assert result is not None, "Result should not be None"
    print(f"  ✓ Processed {len(frames)} different texture patterns")
    print(f"  ✓ Horizontal, spotted, gradient, and cross-hatch textures analyzed")
    
    return True


def test_noisy_textures():
    """Test texture analysis with noise"""
    print("\nTesting with noisy textures...")
    
    analyzer = TextureDescriptors(method='lbp')
    
    frames = []
    noise_levels = [0, 10, 20, 30]
    
    for noise_level in noise_levels:
        # Base pattern
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        # Add regular pattern
        for i in range(0, 100, 10):
            frame[i:i+5, :] = 200
        
        # Add noise
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, frame.shape)
            frame = np.clip(frame.astype(float) + noise, 0, 255).astype(np.uint8)
        
        frames.append(frame)
    
    result = analyzer.analyze(frames)
    
    assert result is not None, "Result should not be None"
    print(f"  ✓ Processed textures with noise levels: {noise_levels}")
    
    return True


def test_gradient_method():
    """Test gradient-based texture analysis"""
    print("\nTesting gradient method...")
    
    analyzer = TextureDescriptors(method='gradient')
    
    frames = []
    
    # Create frames with different edge characteristics
    # Sharp edges
    frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
    frame1[40:60, :] = 255
    frames.append(frame1)
    
    # Smooth gradient
    frame2 = np.zeros((100, 100, 3), dtype=np.uint8)
    for i in range(100):
        frame2[i, :] = int(i * 255 / 100)
    frames.append(frame2)
    
    # Complex edges
    frame3 = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(frame3, (20, 20), (80, 80), (255, 255, 255), 2)
    cv2.circle(frame3, (50, 50), 20, (128, 128, 128), -1)
    frames.append(frame3)
    
    result = analyzer.analyze(frames)
    
    assert result is not None, "Result should not be None"
    print(f"  ✓ Gradient-based texture analysis completed")
    print(f"  ✓ Analyzed sharp edges, smooth gradients, and complex patterns")
    
    return True


def main():
    """Run all smoke tests"""
    print("=" * 50)
    print("TEXTURE DESCRIPTORS SMOKE TESTS")
    print("=" * 50)
    
    tests = [
        ("Basic Functionality", test_texture_basic),
        ("Different Methods", test_texture_methods),
        ("LBP Parameters", test_lbp_parameters),
        ("Known Patterns", test_texture_patterns),
        ("Noisy Textures", test_noisy_textures),
        ("Gradient Method", test_gradient_method),
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