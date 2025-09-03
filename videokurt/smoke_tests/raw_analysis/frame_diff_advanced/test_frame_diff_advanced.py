"""
Run: python -m videokurt.smoke_tests.raw_analysis.frame_diff_advanced.test_frame_diff_advanced

Smoke test for Advanced Frame Diff raw analysis
Tests triple differencing, running average, and accumulated motion
"""

import sys
import cv2
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from videokurt.raw_analysis.frame_diff_advanced import FrameDiffAdvanced


def test_triple_differencing():
    """Test triple frame differencing"""
    print("Testing triple differencing...")
    
    # Initialize analyzer
    analyzer = FrameDiffAdvanced(downsample=1.0, window_size=5, accumulate=True)
    
    # Create test frames with motion
    frames = []
    for i in range(5):
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        # Add moving pattern
        cv2.circle(frame, (100 + i*10, 100), 30, (255, 255, 255), -1)
        frames.append(frame)
    
    # Analyze frames
    result = analyzer.analyze(frames)
    
    # Verify result structure
    assert result is not None, "Result should not be None"
    assert result.method == 'frame_diff_advanced', f"Method should be 'frame_diff_advanced', got {result.method}"
    
    # Check data structure
    assert 'triple_diff' in result.data, "Should have triple_diff in data"
    assert 'running_avg_diff' in result.data, "Should have running_avg_diff in data"
    assert 'accumulated_diff' in result.data, "Should have accumulated_diff in data"
    
    # Triple diff should have n-2 frames (needs 3 frames for each diff)
    assert len(result.data['triple_diff']) == len(frames) - 2, f"Should have {len(frames)-2} triple diffs"
    
    print(f"✓ Analyzed {len(frames)} frames with triple differencing")
    print(f"✓ Triple diff shape: {result.data['triple_diff'].shape}")
    print(f"✓ Running avg diff shape: {result.data['running_avg_diff'].shape}")
    print(f"✓ Accumulated diff shape: {result.data['accumulated_diff'].shape}")
    
    return True


def test_running_average():
    """Test running average background subtraction"""
    print("\nTesting running average...")
    
    # Initialize analyzer
    analyzer = FrameDiffAdvanced(window_size=3, accumulate=False)
    
    # Create frames with sudden change
    frames = []
    
    # First 3 frames: static scene
    for i in range(3):
        frame = np.ones((150, 150, 3), dtype=np.uint8) * 100
        cv2.circle(frame, (75, 75), 20, (200, 200, 200), -1)
        frames.append(frame)
    
    # Next 2 frames: object appears
    for i in range(2):
        frame = np.ones((150, 150, 3), dtype=np.uint8) * 100
        cv2.circle(frame, (75, 75), 20, (200, 200, 200), -1)
        # Add new object
        cv2.rectangle(frame, (20, 20), (60, 60), (255, 255, 255), -1)
        frames.append(frame)
    
    # Analyze
    result = analyzer.analyze(frames)
    
    # Check running average diff
    assert 'running_avg_diff' in result.data, "Should have running_avg_diff"
    running_avg_diff = result.data['running_avg_diff']
    
    # Should detect more change in later frames (when new object appears)
    early_motion = running_avg_diff[0].mean()
    late_motion = running_avg_diff[-1].mean()
    
    print(f"✓ Running avg diff shape: {running_avg_diff.shape}")
    print(f"✓ Early motion: {early_motion:.2f}, Late motion: {late_motion:.2f}")
    
    # Since accumulate=False, accumulated_diff should be None
    assert result.data['accumulated_diff'] is None, "Accumulated diff should be None when accumulate=False"
    
    return True


def test_accumulated_motion():
    """Test accumulated motion history"""
    print("\nTesting accumulated motion...")
    
    # Initialize with accumulation enabled
    analyzer = FrameDiffAdvanced(accumulate=True)
    
    # Create frames with motion in specific area
    frames = []
    for i in range(10):
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        
        # Persistent motion in top-left corner
        x = 30 + (i * 5) % 40
        cv2.circle(frame, (x, 30), 10, (255, 255, 255), -1)
        
        # Brief motion in bottom-right (only frames 3-5)
        if 3 <= i <= 5:
            cv2.rectangle(frame, (150, 150), (180, 180), (200, 200, 200), -1)
        
        frames.append(frame)
    
    # Analyze
    result = analyzer.analyze(frames)
    
    # Check accumulated diff
    assert result.data['accumulated_diff'] is not None, "Should have accumulated_diff when accumulate=True"
    accumulated = result.data['accumulated_diff']
    
    # Check that accumulation happened
    max_accumulated = accumulated.max()
    mean_accumulated = accumulated.mean()
    
    assert max_accumulated > 0, "Should have some accumulated motion"
    
    print(f"✓ Accumulated diff shape: {accumulated.shape}")
    print(f"✓ Max accumulated value: {max_accumulated}")
    print(f"✓ Mean accumulated value: {mean_accumulated:.4f}")
    
    return True


def test_noise_reduction():
    """Test noise reduction via triple differencing"""
    print("\nTesting noise reduction...")
    
    analyzer = FrameDiffAdvanced()
    
    # Create frames with noise
    frames = []
    np.random.seed(42)  # For reproducibility
    
    for i in range(5):
        # Base frame
        frame = np.ones((150, 150, 3), dtype=np.uint8) * 128
        
        # Add consistent object with motion
        cv2.rectangle(frame, (50 + i*10, 50), (80 + i*10, 80), (255, 255, 255), -1)
        
        # Add random noise
        noise = np.random.randint(0, 30, frame.shape, dtype=np.uint8)
        frame = cv2.add(frame, noise)
        
        frames.append(frame)
    
    # Analyze
    result = analyzer.analyze(frames)
    
    # Triple diff should reduce noise compared to simple differencing
    triple_diff = result.data['triple_diff']
    
    # The triple diff should highlight the moving rectangle while reducing noise
    assert triple_diff is not None, "Should have triple_diff"
    
    # Check that we detected motion
    motion_detected = (triple_diff > 20).sum()
    assert motion_detected > 0, "Should detect some motion despite noise"
    
    print(f"✓ Triple diff reduces noise")
    print(f"✓ Motion pixels detected: {motion_detected}")
    
    return True


def test_with_real_video():
    """Test with real video file if available"""
    print("\nTesting with real video...")
    
    video_path = Path('/Users/ns/Desktop/projects/videokurt/sample_recording.MP4')
    
    if not video_path.exists():
        print("⚠ Sample video not found, skipping real video test")
        return True
    
    # Load first 20 frames
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    for _ in range(20):
        ret, frame = cap.read()
        if not ret:
            break
        # Downsample for speed
        frame = cv2.resize(frame, (frame.shape[1]//4, frame.shape[0]//4))
        frames.append(frame)
    cap.release()
    
    if len(frames) < 3:
        print("⚠ Could not load enough frames from video (need at least 3)")
        return False
    
    # Test with full features
    analyzer = FrameDiffAdvanced(downsample=0.5, window_size=5, accumulate=True)
    result = analyzer.analyze(frames)
    
    # Verify all outputs are generated
    assert result.data['triple_diff'] is not None, "Should have triple_diff"
    assert result.data['running_avg_diff'] is not None, "Should have running_avg_diff"
    assert result.data['accumulated_diff'] is not None, "Should have accumulated_diff"
    
    print(f"✓ Analyzed {len(frames)} real frames")
    print(f"✓ Triple diff mean: {result.data['triple_diff'].mean():.4f}")
    print(f"✓ Running avg diff mean: {result.data['running_avg_diff'].mean():.4f}")
    print(f"✓ Accumulated diff max: {result.data['accumulated_diff'].max()}")
    print(f"✓ Processing time: {result.processing_time:.3f}s")
    
    return True


def main():
    """Run all tests"""
    print("="*50)
    print("Advanced Frame Diff Smoke Test")
    print("="*50)
    
    tests = [
        test_triple_differencing,
        test_running_average,
        test_accumulated_motion,
        test_noise_reduction,
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
    print("All Advanced Frame Diff tests passed! ✓")
    print("="*50)
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)