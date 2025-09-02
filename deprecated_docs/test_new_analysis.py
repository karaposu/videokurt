"""Test the new analysis system."""

import numpy as np
from videokurt.models import RawAnalysisResults, RawAnalysis
from videokurt.analysis_models import (
    FrameDiff,
    OpticalFlowDense,
    MotionHeatmap,
    ANALYSIS_REGISTRY
)

def create_test_frames(num_frames=10, height=100, width=100):
    """Create simple test frames with moving square."""
    frames = []
    for i in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Draw a moving white square
        x = (i * 10) % (width - 20)
        y = (i * 5) % (height - 20)
        frame[y:y+20, x:x+20] = 255
        frames.append(frame)
    return frames

def test_individual_analysis():
    """Test individual analysis classes."""
    print("Testing individual analyses...")
    print("-" * 50)
    
    # Create test frames
    frames = create_test_frames(20)
    print(f"Created {len(frames)} test frames of size {frames[0].shape}")
    
    # Test 1: Frame Differencing
    print("\n1. Testing FrameDiff...")
    frame_diff = FrameDiff(downsample=0.5, threshold=0.1)
    result = frame_diff.analyze(frames)
    print(f"   Method: {result.method}")
    print(f"   Processing time: {result.processing_time:.3f}s")
    print(f"   Output shape: {result.output_shapes}")
    print(f"   Data keys: {list(result.data.keys())}")
    
    # Test 2: Optical Flow (with heavy downsampling for speed)
    print("\n2. Testing OpticalFlowDense...")
    optical_flow = OpticalFlowDense(downsample=0.25, levels=2)
    result = optical_flow.analyze(frames)
    print(f"   Method: {result.method}")
    print(f"   Processing time: {result.processing_time:.3f}s")
    print(f"   Flow field shape: {result.data['flow_field'].shape}")
    
    # Test 3: Motion Heatmap
    print("\n3. Testing MotionHeatmap...")
    heatmap = MotionHeatmap(downsample=0.5, decay_factor=0.95)
    result = heatmap.analyze(frames)
    print(f"   Method: {result.method}")
    print(f"   Processing time: {result.processing_time:.3f}s")
    print(f"   Cumulative heatmap shape: {result.data['cumulative'].shape}")
    print(f"   Weighted heatmap shape: {result.data['weighted'].shape}")

def test_results_integration():
    """Test the RawAnalysisResults integration."""
    print("\n" + "="*50)
    print("Testing RawAnalysisResults integration...")
    print("-" * 50)
    
    # Create test frames
    frames = create_test_frames(10)
    
    # Run multiple analyses
    analyses = {}
    
    # Run frame diff
    frame_diff = FrameDiff(downsample=1.0)
    analyses['frame_diff'] = frame_diff.analyze(frames)
    
    # Run optical flow  
    optical_flow = OpticalFlowDense(downsample=0.25)
    analyses['optical_flow_dense'] = optical_flow.analyze(frames)
    
    # Create results object
    results = RawAnalysisResults(
        dimensions=(100, 100),
        fps=30.0,
        duration=10/30.0,
        frame_count=10,
        analyses=analyses,
        elapsed_time=sum(a.processing_time for a in analyses.values()),
        filename="test_video.mp4"
    )
    
    # Test convenience methods
    print(f"Analyses run: {results.list_analyses()}")
    print(f"Has frame_diff: {results.has_analysis('frame_diff')}")
    print(f"Has motion_heatmap: {results.has_analysis('motion_heatmap')}")
    
    # Get specific analysis
    if results.has_analysis('optical_flow_dense'):
        flow_analysis = results.get_analysis('optical_flow_dense')
        print(f"\nOptical flow details:")
        print(f"  Processing time: {flow_analysis.processing_time:.3f}s")
        print(f"  Parameters: {flow_analysis.parameters}")
    
    # Print summary
    print("\nSummary:")
    results.print_summary()
    
    # Test JSON conversion
    print("\nJSON output (truncated):")
    json_str = results.to_json()
    print(json_str[:500] + "...")

def test_registry():
    """Test the analysis registry."""
    print("\n" + "="*50)
    print("Testing Analysis Registry...")
    print("-" * 50)
    
    print(f"Available analyses: {len(ANALYSIS_REGISTRY)}")
    for name, cls in ANALYSIS_REGISTRY.items():
        print(f"  - {name}: {cls.__name__}")

if __name__ == "__main__":
    test_individual_analysis()
    test_results_integration()
    test_registry()
    
    print("\n" + "="*50)
    print("All tests completed successfully!")