"""
Test 03: Edge Detection Analysis
Tests the EdgeCanny analysis class for edge detection.

Run: python -m videokurt.smoke_tests.pure_analysis.test_03_edge_detection

"""

import numpy as np
from videokurt.raw_analysis.edge_canny import EdgeCanny


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
    assert 'edge_map' in result.data
    
    edges = result.data['edge_map']
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
    edges_low = np.sum(result_low.data['edge_map'] > 0)
    
    # High thresholds - fewer edges
    analyzer_high = EdgeCanny(downsample=0.5, low_threshold=100, high_threshold=200)
    result_high = analyzer_high.analyze(frames)
    edges_high = np.sum(result_high.data['edge_map'] > 0)
    
    assert edges_low > edges_high, "Lower thresholds should detect more edges"
    
    print(f"✓ Low threshold edges: {edges_low}")
    print(f"✓ High threshold edges: {edges_high}")
    print(f"✓ Ratio: {edges_low/edges_high:.2f}x more edges with low threshold")


def test_downsample_performance():
    """Test downsampling for performance."""
    print("\nTest 3: Downsample Performance")
    print("-" * 40)
    
    # Create test frames
    frames = []
    for i in range(5):
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        # Add patterns
        frame[50:150, 50:150] = 255
        frame[75:125, 75:125] = 0
        frames.append(frame)
    
    # Full resolution
    analyzer_full = EdgeCanny(downsample=1.0, low_threshold=50, high_threshold=150)
    result_full = analyzer_full.analyze(frames)
    
    # Half resolution
    analyzer_half = EdgeCanny(downsample=0.5, low_threshold=50, high_threshold=150)
    result_half = analyzer_half.analyze(frames)
    
    # Quarter resolution
    analyzer_quarter = EdgeCanny(downsample=0.25, low_threshold=50, high_threshold=150)
    result_quarter = analyzer_quarter.analyze(frames)
    
    print(f"✓ Full resolution shape: {result_full.data['edge_map'].shape}")
    print(f"✓ Half resolution shape: {result_half.data['edge_map'].shape}")
    print(f"✓ Quarter resolution shape: {result_quarter.data['edge_map'].shape}")
    print(f"✓ Speedup (full->quarter): {result_full.processing_time / result_quarter.processing_time:.2f}x")


if __name__ == "__main__":
    print("="*50)
    print("Edge Detection Analysis Tests")
    print("="*50)
    
    try:
        test_edge_detection()
        test_threshold_sensitivity()
        test_downsample_performance()
        
        print("\n" + "="*50)
        print("ALL TESTS PASSED ✓")
        print("="*50)
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        exit(1)
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        exit(1)