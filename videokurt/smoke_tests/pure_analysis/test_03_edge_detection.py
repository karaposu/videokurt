"""
Test 03: Edge Detection Analysis
Tests the EdgeCanny analysis class for edge detection.

Run: python -m videokurt.smoke_tests.pure_analysis.test_03_edge_detection

"""

import numpy as np
from videokurt.analysis_models import EdgeCanny


def create_frames_with_edges():
    """Create frames with clear edges."""
    frames = []
    for i in range(10):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Create sharp edges with rectangles
        frame[20:40, 20:40] = 255  # White square
        frame[60:80, 60:80] = 128  # Gray square
        
        # Moving vertical line
        x = (i * 5) % 90
        frame[:, x:x+2] = 200
        
        frames.append(frame)
    return frames


def test_edge_detection():
    """Test basic edge detection."""
    print("\nTest 1: Basic Edge Detection")
    print("-" * 40)
    
    frames = create_frames_with_edges()
    
    # Run analysis
    analyzer = EdgeCanny(downsample=1.0, low_threshold=50, high_threshold=150)
    result = analyzer.analyze(frames)
    
    # Check results
    assert result.method == 'edge_canny'
    assert 'edges' in result.data
    
    edges = result.data['edges']
    assert edges.shape[0] == len(frames)
    assert edges.dtype == np.uint8
    
    # Check that edges were detected (should have non-zero values)
    edge_pixels = np.sum(edges > 0)
    assert edge_pixels > 0, "No edges detected"
    
    print(f"✓ Method: {result.method}")
    print(f"✓ Edges shape: {edges.shape}")
    print(f"✓ Edge pixels detected: {edge_pixels}")
    print(f"✓ Processing time: {result.processing_time:.3f}s")


def test_threshold_sensitivity():
    """Test sensitivity to threshold parameters."""
    print("\nTest 2: Threshold Sensitivity")
    print("-" * 40)
    
    frames = create_frames_with_edges()
    
    # Low thresholds - more edges
    analyzer_low = EdgeCanny(downsample=0.5, low_threshold=30, high_threshold=80)
    result_low = analyzer_low.analyze(frames)
    edges_low = np.sum(result_low.data['edges'] > 0)
    
    # High thresholds - fewer edges
    analyzer_high = EdgeCanny(downsample=0.5, low_threshold=100, high_threshold=200)
    result_high = analyzer_high.analyze(frames)
    edges_high = np.sum(result_high.data['edges'] > 0)
    
    assert edges_low > edges_high, "Lower thresholds should detect more edges"
    
    print(f"✓ Low threshold edges: {edges_low}")
    print(f"✓ High threshold edges: {edges_high}")
    print(f"✓ Ratio: {edges_low/edges_high:.2f}x more edges with low threshold")


def test_blur_preprocessing():
    """Test Gaussian blur preprocessing effect."""
    print("\nTest 3: Blur Preprocessing")
    print("-" * 40)
    
    # Create noisy frames
    frames = []
    for i in range(5):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        # Add noise
        noise = np.random.randint(0, 50, (100, 100, 3), dtype=np.uint8)
        frame = frame + noise
        # Add strong edge
        frame[40:60, 40:60] = 255
        frames.append(frame)
    
    # Without blur
    analyzer_no_blur = EdgeCanny(downsample=0.5, blur_kernel=0)
    result_no_blur = analyzer_no_blur.analyze(frames)
    
    # With blur
    analyzer_blur = EdgeCanny(downsample=0.5, blur_kernel=5)
    result_blur = analyzer_blur.analyze(frames)
    
    # Blur should reduce noise edges
    noise_edges_no_blur = np.sum(result_no_blur.data['edges'] > 0)
    noise_edges_blur = np.sum(result_blur.data['edges'] > 0)
    
    print(f"✓ Edges without blur: {noise_edges_no_blur}")
    print(f"✓ Edges with blur: {noise_edges_blur}")
    print(f"✓ Blur reduces noisy edges: {noise_edges_no_blur > noise_edges_blur}")


if __name__ == "__main__":
    print("="*50)
    print("Edge Detection Analysis Tests")
    print("="*50)
    
    try:
        test_edge_detection()
        test_threshold_sensitivity()
        test_blur_preprocessing()
        
        print("\n" + "="*50)
        print("ALL TESTS PASSED ✓")
        print("="*50)
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        exit(1)
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        exit(1)