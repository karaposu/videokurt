"""
Run: python -m videokurt.smoke_tests.raw_analysis.background_subtraction.test_background_subtraction

Smoke test for Background Subtraction raw analyses
Tests both KNN and MOG2 background subtraction methods
"""

import sys
import cv2
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from videokurt.raw_analysis.background_knn import BackgroundKNN
from videokurt.raw_analysis.background_mog2 import BackgroundMOG2


def test_background_knn():
    """Test KNN background subtraction"""
    print("Testing Background KNN...")
    
    # Initialize analyzer with correct parameters
    analyzer = BackgroundKNN(
        downsample=1.0,
        history=100,
        dist2_threshold=400.0,
        detect_shadows=True
    )
    
    # Create test frames with moving object
    frames = []
    for i in range(10):
        frame = np.ones((300, 400, 3), dtype=np.uint8) * 50  # Gray background
        
        # Add moving object every few frames
        if i > 2:  # Let background model stabilize
            x = 50 + i * 30
            cv2.rectangle(frame, (x, 100), (x + 50, 200), (255, 255, 255), -1)
        
        frames.append(frame)
    
    # Analyze frames
    result = analyzer.analyze(frames)
    
    # Verify result structure
    assert result is not None, "Result should not be None"
    assert hasattr(result, 'data'), "Result should have data attribute"
    assert hasattr(result, 'method'), "Result should have method attribute"
    assert result.method == 'background_knn', f"Method should be 'background_knn', got {result.method}"
    
    # Check data structure (returns 'foreground_mask' not 'foreground_masks')
    assert 'foreground_mask' in result.data, "Should have foreground_mask in data"
    foreground_mask = result.data['foreground_mask']
    assert len(foreground_mask) == len(frames), "Should have mask for each frame"
    
    # Check that motion is detected
    motion_pixels = (foreground_mask > 0).sum(axis=(1, 2))
    total_motion = motion_pixels.sum()
    assert total_motion > 0, "Should detect some motion"
    
    print(f"✓ Analyzed {len(frames)} frames")
    print(f"✓ Foreground mask shape: {foreground_mask.shape}")
    print(f"✓ Total motion pixels detected: {total_motion}")
    
    return True


def test_background_mog2():
    """Test MOG2 background subtraction"""
    print("\nTesting Background MOG2...")
    
    # Initialize analyzer with correct parameters
    analyzer = BackgroundMOG2(
        downsample=1.0,
        history=120,
        var_threshold=16.0,
        detect_shadows=False
    )
    
    # Create test frames with sudden scene change
    frames = []
    
    # First 5 frames: static scene
    for i in range(5):
        frame = np.ones((300, 400, 3), dtype=np.uint8) * 100
        cv2.circle(frame, (200, 150), 30, (200, 200, 200), -1)  # Static circle
        frames.append(frame)
    
    # Next 5 frames: moving object
    for i in range(5):
        frame = np.ones((300, 400, 3), dtype=np.uint8) * 100
        cv2.circle(frame, (200, 150), 30, (200, 200, 200), -1)  # Keep static circle
        # Add moving rectangle
        x = 50 + i * 40
        cv2.rectangle(frame, (x, 50), (x + 60, 250), (255, 0, 0), -1)
        frames.append(frame)
    
    # Analyze frames
    result = analyzer.analyze(frames)
    
    # Verify result structure
    assert result.method == 'background_mog2', f"Method should be 'background_mog2', got {result.method}"
    
    # Check data structure
    assert 'foreground_mask' in result.data, "Should have foreground_mask in data"
    foreground_mask = result.data['foreground_mask']
    assert len(foreground_mask) == len(frames), "Should have mask for each frame"
    
    # Check that motion is detected
    motion_pixels = (foreground_mask > 0).sum(axis=(1, 2))
    total_motion = motion_pixels.sum()
    assert total_motion > 0, "Should detect some motion"
    
    print(f"✓ Analyzed {len(frames)} frames")
    print(f"✓ Total motion pixels detected: {total_motion}")
    
    return True


def test_shadow_detection():
    """Test shadow detection capability"""
    print("\nTesting shadow detection...")
    
    # Test both with shadow detection
    analyzer_knn = BackgroundKNN(detect_shadows=True)
    analyzer_mog2 = BackgroundMOG2(detect_shadows=True)
    
    # Create frames with shadows
    frames = []
    for i in range(5):
        frame = np.ones((200, 200, 3), dtype=np.uint8) * 200  # Bright background
        
        if i > 0:
            # Add dark object (potential shadow)
            cv2.rectangle(frame, (50, 50), (150, 150), (100, 100, 100), -1)
        
        frames.append(frame)
    
    # Analyze with both methods
    result_knn = analyzer_knn.analyze(frames)
    result_mog2 = analyzer_mog2.analyze(frames)
    
    print(f"✓ KNN with shadows: processed {len(frames)} frames")
    print(f"✓ MOG2 with shadows: processed {len(frames)} frames")
    
    # Both should process frames successfully
    knn_mask = result_knn.data['foreground_mask']
    mog2_mask = result_mog2.data['foreground_mask']
    assert knn_mask.shape[0] == len(frames), "KNN processed all frames"
    assert mog2_mask.shape[0] == len(frames), "MOG2 processed all frames"
    
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
        # Downsample for speed
        frame = cv2.resize(frame, (frame.shape[1]//4, frame.shape[0]//4))
        frames.append(frame)
    cap.release()
    
    if len(frames) < 10:
        print("⚠ Could not load enough frames from video")
        return False
    
    # Test both methods with correct parameters
    analyzer_knn = BackgroundKNN(history=50, downsample=1.0)
    analyzer_mog2 = BackgroundMOG2(history=50, downsample=1.0)
    
    result_knn = analyzer_knn.analyze(frames)
    result_mog2 = analyzer_mog2.analyze(frames)
    
    # Calculate motion ratios
    knn_motion_ratio = (result_knn.data['foreground_mask'] > 0).sum() / result_knn.data['foreground_mask'].size
    mog2_motion_ratio = (result_mog2.data['foreground_mask'] > 0).sum() / result_mog2.data['foreground_mask'].size
    
    print(f"✓ KNN: motion ratio = {knn_motion_ratio:.4f}")
    print(f"✓ MOG2: motion ratio = {mog2_motion_ratio:.4f}")
    print(f"✓ Processing times: KNN={result_knn.processing_time:.3f}s, MOG2={result_mog2.processing_time:.3f}s")
    
    return True


def main():
    """Run all tests"""
    print("="*50)
    print("Background Subtraction Smoke Test")
    print("="*50)
    
    tests = [
        test_background_knn,
        test_background_mog2,
        test_shadow_detection,
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
    print("All Background Subtraction tests passed! ✓")
    print("="*50)
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)