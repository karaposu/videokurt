"""
Run: python -m videokurt.smoke_tests.raw_analysis.frame_diff.test_frame_diff

Smoke test for Frame Diff raw analysis
Tests basic functionality with synthetic and real video
"""

import sys
import cv2
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from videokurt.raw_analysis.frame_diff import FrameDiff


def test_frame_diff_basic():
    """Test basic frame differencing"""
    print("Testing Frame Diff...")
    
    # Initialize analyzer
    analyzer = FrameDiff(downsample=1.0, threshold=0.1)
    
    # Create test frames with motion
    frames = []
    for i in range(5):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Moving rectangle
        x = (i * 50) % 500
        cv2.rectangle(frame, (x, 200), (x + 100, 300), (255, 255, 255), -1)
        frames.append(frame)
    
    # Analyze frames
    result = analyzer.analyze(frames)
    
    # Verify result structure
    assert result is not None, "Result should not be None"
    assert hasattr(result, 'data'), "Result should have data attribute"
    assert hasattr(result, 'method'), "Result should have method attribute"
    assert result.method == 'frame_diff', f"Method should be 'frame_diff', got {result.method}"
    
    # Check data structure
    assert 'pixel_diff' in result.data, "Should have pixel_diff in data"
    assert len(result.data['pixel_diff']) == len(frames) - 1, "Should have n-1 diffs for n frames"
    
    # Calculate mean diff from pixel_diff array
    mean_diff = result.data['pixel_diff'].mean()
    
    print(f"✓ Analyzed {len(frames)} frames")
    print(f"✓ Pixel diff shape: {result.data['pixel_diff'].shape}")
    print(f"✓ Mean diff: {mean_diff:.4f}")
    print(f"✓ Method: {result.method}")
    
    return True


def test_frame_diff_static():
    """Test with static frames (no motion)"""
    print("\nTesting static frames...")
    
    analyzer = FrameDiff(threshold=0.05)
    
    # Create static frames
    frames = []
    static_frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
    for _ in range(5):
        frames.append(static_frame.copy())
    
    result = analyzer.analyze(frames)
    
    # With static frames, differences should be near zero
    mean_diff = result.data['pixel_diff'].mean()
    assert mean_diff < 0.01, f"Static frames should have low diff, got {mean_diff}"
    
    print(f"✓ Static mean diff: {mean_diff:.6f}")
    print(f"✓ Max pixel diff: {result.data['pixel_diff'].max():.6f}")
    
    return True


def test_with_real_video():
    """Test with real video file if available"""
    print("\nTesting with real video...")
    
    video_path = Path('/Users/ns/Desktop/projects/videokurt/sample_recording.MP4')
    
    if not video_path.exists():
        print("⚠ Sample video not found, skipping real video test")
        return True
    
    # Load first 30 frames
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    for _ in range(30):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    if len(frames) < 2:
        print("⚠ Could not load enough frames from video")
        return False
    
    # Analyze with downsampling for speed
    analyzer = FrameDiff(downsample=0.25, threshold=0.1)
    result = analyzer.analyze(frames)
    
    mean_diff = result.data['pixel_diff'].mean()
    
    print(f"✓ Analyzed {len(frames)} real frames")
    print(f"✓ Mean diff: {mean_diff:.4f}")
    print(f"✓ Processing time: {result.processing_time:.3f}s")
    
    return True


def main():
    """Run all tests"""
    print("="*50)
    print("Frame Diff Smoke Test")
    print("="*50)
    
    tests = [
        test_frame_diff_basic,
        test_frame_diff_static,
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
    print("All Frame Diff tests passed! ✓")
    print("="*50)
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)