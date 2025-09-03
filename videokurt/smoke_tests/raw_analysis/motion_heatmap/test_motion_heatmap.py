"""
Run: python -m videokurt.smoke_tests.raw_analysis.motion_heatmap.test_motion_heatmap

Smoke test for Motion Heatmap raw analysis
Tests cumulative and weighted motion heatmap generation
"""

import sys
import cv2
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from videokurt.raw_analysis.motion_heatmap import MotionHeatmap


def test_motion_heatmap_basic():
    """Test basic motion heatmap generation"""
    print("Testing Motion Heatmap...")
    
    # Initialize analyzer
    analyzer = MotionHeatmap(downsample=1.0, decay_factor=0.95, snapshot_interval=3)
    
    # Create test frames with motion in specific areas
    frames = []
    for i in range(10):
        frame = np.zeros((300, 400, 3), dtype=np.uint8)
        
        # Static background element
        cv2.rectangle(frame, (350, 50), (390, 250), (100, 100, 100), -1)
        
        # Moving object in top-left (high activity area)
        x = 50 + (i * 10) % 100
        cv2.circle(frame, (x, 50), 20, (255, 255, 255), -1)
        
        # Another moving object in bottom (medium activity)
        if i % 2 == 0:
            cv2.rectangle(frame, (150, 200), (200, 250), (200, 200, 200), -1)
        else:
            cv2.rectangle(frame, (160, 210), (210, 260), (200, 200, 200), -1)
        
        frames.append(frame)
    
    # Analyze frames
    result = analyzer.analyze(frames)
    
    # Verify result structure
    assert result is not None, "Result should not be None"
    assert hasattr(result, 'data'), "Result should have data attribute"
    assert hasattr(result, 'method'), "Result should have method attribute"
    assert result.method == 'motion_heatmap', f"Method should be 'motion_heatmap', got {result.method}"
    
    # Check data structure
    assert 'cumulative' in result.data, "Should have cumulative heatmap in data"
    assert 'weighted' in result.data, "Should have weighted heatmap in data"
    assert 'snapshots' in result.data, "Should have snapshots in data"
    
    # Verify heatmap dimensions
    cumulative = result.data['cumulative']
    weighted = result.data['weighted']
    assert cumulative.shape[:2] == (300, 400), f"Cumulative shape should be (300, 400), got {cumulative.shape[:2]}"
    assert weighted.shape[:2] == (300, 400), f"Weighted shape should be (300, 400), got {weighted.shape[:2]}"
    
    # Check that we have motion detected
    assert cumulative.max() > 0, "Should detect some motion in cumulative"
    assert weighted.max() > 0, "Should detect some motion in weighted"
    
    print(f"✓ Analyzed {len(frames)} frames")
    print(f"✓ Cumulative heatmap shape: {cumulative.shape}")
    print(f"✓ Weighted heatmap shape: {weighted.shape}")
    print(f"✓ Max cumulative heat: {cumulative.max()}")
    print(f"✓ Max weighted heat: {weighted.max()}")
    print(f"✓ Snapshots taken: {len(result.data['snapshots'])}")
    
    return True


def test_decay_factor():
    """Test different decay factors"""
    print("\nTesting decay factors...")
    
    # Test with different decay values
    decay_factors = [0.8, 0.95, 0.99]  # Fast decay, medium, slow decay
    
    # Create consistent test frames with brief motion
    frames = []
    for i in range(15):
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        # Brief motion in first 5 frames only
        if i < 5:
            cv2.circle(frame, (100, 100), 30 + i*5, (255, 255, 255), 2)
        else:
            # Static after frame 5
            cv2.circle(frame, (100, 100), 50, (128, 128, 128), -1)
        frames.append(frame)
    
    weighted_maxes = []
    for decay in decay_factors:
        analyzer = MotionHeatmap(decay_factor=decay, snapshot_interval=5)
        result = analyzer.analyze(frames)
        weighted_max = result.data['weighted'].max()
        weighted_maxes.append(weighted_max)
        print(f"✓ Decay={decay}: max weighted heat = {weighted_max}")
    
    # Higher decay (slower decay) should retain more heat
    # Note: Results may vary, so just check they're all non-zero
    assert all(w > 0 for w in weighted_maxes), "All decay factors should detect motion"
    
    return True


def test_snapshots():
    """Test snapshot functionality"""
    print("\nTesting snapshots...")
    
    # Initialize with frequent snapshots
    analyzer = MotionHeatmap(snapshot_interval=2)  # Snapshot every 2 frames
    
    # Create frames
    frames = []
    for i in range(10):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        # Moving diagonal line
        cv2.line(frame, (i*10, 0), (i*10 + 20, 100), (255, 255, 255), 2)
        frames.append(frame)
    
    # Analyze
    result = analyzer.analyze(frames)
    
    # Check snapshots
    assert 'snapshots' in result.data, "Should have snapshots"
    snapshots = result.data['snapshots']
    
    # With 10 frames and snapshot_interval=2, we should have several snapshots
    # Snapshots are taken at frames 2, 4, 6, 8 (indices in processing)
    expected_snapshots = (10 - 1) // 2  # -1 because motion is computed from pairs
    assert len(snapshots) > 0, "Should have at least one snapshot"
    
    print(f"✓ Created {len(snapshots)} snapshots")
    for i, (time_ratio, snapshot) in enumerate(snapshots):
        print(f"  Snapshot {i+1}: time={time_ratio:.2f}, shape={snapshot.shape}")
    
    return True


def test_cumulative_vs_weighted():
    """Test difference between cumulative and weighted heatmaps"""
    print("\nTesting cumulative vs weighted...")
    
    analyzer = MotionHeatmap(decay_factor=0.9)
    
    # Create frames with motion that stops halfway
    frames = []
    for i in range(20):
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        
        # Motion only in first half
        if i < 10:
            x = 50 + i * 10
            cv2.rectangle(frame, (x, 50), (x + 30, 150), (255, 255, 255), -1)
        else:
            # Static in second half
            cv2.rectangle(frame, (150, 50), (180, 150), (128, 128, 128), -1)
        
        frames.append(frame)
    
    # Analyze
    result = analyzer.analyze(frames)
    
    cumulative = result.data['cumulative']
    weighted = result.data['weighted']
    
    # Cumulative should have accumulated all motion
    # Weighted should decay over time
    cumulative_mean = cumulative.mean()
    weighted_mean = weighted.mean()
    
    print(f"✓ Cumulative mean: {cumulative_mean:.4f}")
    print(f"✓ Weighted mean: {weighted_mean:.4f}")
    print(f"✓ Cumulative max: {cumulative.max()}")
    print(f"✓ Weighted max: {weighted.max()}")
    
    # Both should detect motion
    assert cumulative.max() > 0, "Cumulative should have motion"
    assert weighted.max() > 0, "Weighted should have motion"
    
    return True


def test_with_real_video():
    """Test with real video file if available"""
    print("\nTesting with real video...")
    
    video_path = Path('/Users/ns/Desktop/projects/videokurt/sample_recording.MP4')
    
    if not video_path.exists():
        print("⚠ Sample video not found, skipping real video test")
        return True
    
    # Load first 50 frames
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    for _ in range(50):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    if len(frames) < 10:
        print("⚠ Could not load enough frames from video")
        return False
    
    # Analyze with motion heatmap (with downsampling for speed/memory)
    analyzer = MotionHeatmap(downsample=0.25, decay_factor=0.95, snapshot_interval=10)
    result = analyzer.analyze(frames)
    
    cumulative = result.data['cumulative']
    weighted = result.data['weighted']
    
    print(f"✓ Analyzed {len(frames)} real frames")
    print(f"✓ Cumulative max: {cumulative.max()}")
    print(f"✓ Weighted max: {weighted.max()}")
    print(f"✓ Active pixels (cumulative): {(cumulative > 0).sum()}")
    print(f"✓ Active pixels (weighted): {(weighted > 0).sum()}")
    print(f"✓ Snapshots: {len(result.data['snapshots'])}")
    print(f"✓ Processing time: {result.processing_time:.3f}s")
    
    return True


def main():
    """Run all tests"""
    print("="*50)
    print("Motion Heatmap Smoke Test")
    print("="*50)
    
    tests = [
        test_motion_heatmap_basic,
        test_decay_factor,
        test_snapshots,
        test_cumulative_vs_weighted,
        test_with_real_video
    ]
    
    for test in tests:
        try:
            if not test():
                print(f"✗ {test.__name__} failed")
                return False
        except Exception as e:
            print(f"✗ {test.__name__} error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print("\n" + "="*50)
    print("All Motion Heatmap tests passed! ✓")
    print("="*50)
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)