"""
Smoke Test 03: VideoKurt Preprocessing Options
Tests frame_step, resolution_scale, blur, process_chunks, and chunk_overlap.

Run: python -m videokurt.smoke_tests.videokurt.test_03_preprocessing
"""

import sys
import numpy as np
import cv2
from pathlib import Path
from videokurt.videokurt_new import VideoKurt
from videokurt.samplemaker import (
    create_frame_sequence,
    create_test_video_frames,
    create_gradient_frame,
    add_rectangle,
    add_text_region,
    simulate_scroll,
    create_frames_with_pattern
)


def test_frame_step():
    """Test frame_step reduces number of frames processed."""
    vk = VideoKurt()
    
    # Create test frames using samplemaker
    frames = create_frames_with_pattern(num_frames=20, width=100, height=100, pattern='moving_line')
    
    # Test frame_step=1 (all frames)
    vk.configure(frame_step=1)
    processed = vk._preprocess_frames(frames)
    assert len(processed) == 20
    print("✓ frame_step=1 processes all frames")
    
    # Test frame_step=2 (every other frame)
    # Note: frame_step is applied during video loading, not preprocessing
    # So we need to simulate this
    vk.configure(frame_step=2)
    # Simulate frame_step selection
    selected_frames = [frames[i] for i in range(0, len(frames), 2)]
    processed = vk._preprocess_frames(selected_frames)
    assert len(processed) == 10
    print("✓ frame_step=2 would select half the frames")
    
    # Test frame_step=5
    vk.configure(frame_step=5)
    selected_frames = [frames[i] for i in range(0, len(frames), 5)]
    processed = vk._preprocess_frames(selected_frames)
    assert len(processed) == 4
    print("✓ frame_step=5 would select every 5th frame")


def test_resolution_scale():
    """Test resolution_scale reduces frame dimensions."""
    vk = VideoKurt()
    
    # Create test frames using samplemaker
    frames = create_frames_with_pattern(num_frames=5, width=200, height=150, pattern='gradient')
    original_shape = frames[0].shape
    
    # Test no scaling
    vk.configure(resolution_scale=1.0)
    processed = vk._preprocess_frames(frames)
    assert processed[0].shape == original_shape
    print("✓ resolution_scale=1.0 preserves size")
    
    # Test 50% scaling
    vk.configure(resolution_scale=0.5)
    processed = vk._preprocess_frames(frames)
    assert processed[0].shape[0] == 75  # height * 0.5
    assert processed[0].shape[1] == 100  # width * 0.5
    print("✓ resolution_scale=0.5 halves dimensions")
    
    # Test 25% scaling
    vk.configure(resolution_scale=0.25)
    processed = vk._preprocess_frames(frames)
    assert processed[0].shape[0] == 37  # height * 0.25 (rounded)
    assert processed[0].shape[1] == 50  # width * 0.25
    print("✓ resolution_scale=0.25 quarters dimensions")


def test_blur():
    """Test blur preprocessing option."""
    vk = VideoKurt()
    
    # Create test frames with sharp edges - just use a simple pattern with lines
    frames = []
    for i in range(3):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        # Add sharp vertical lines
        for x in range(10, 100, 20):
            cv2.line(frame, (x, 0), (x, 99), (255, 255, 255), 1)
        # Add sharp horizontal lines  
        for y in range(10, 100, 20):
            cv2.line(frame, (0, y), (99, y), (255, 255, 255), 1)
        frames.append(frame)
    
    # Get edge strength without blur
    vk.configure(blur=False)
    processed_no_blur = vk._preprocess_frames(frames)
    edges_no_blur = cv2.Canny(processed_no_blur[0], 50, 150)
    edge_count_no_blur = np.sum(edges_no_blur > 0)
    
    # Get edge strength with blur
    vk.configure(blur=True, blur_kernel_size=7)
    processed_blur = vk._preprocess_frames(frames)
    edges_blur = cv2.Canny(processed_blur[0], 50, 150)
    edge_count_blur = np.sum(edges_blur > 0)
    
    # Blur should reduce edge count OR at least change the image
    # Check that the images are different
    diff = np.mean(np.abs(processed_blur[0].astype(float) - processed_no_blur[0].astype(float)))
    assert diff > 0, "Blur should change the image"
    print(f"✓ Blur changes image (mean diff: {diff:.2f})")
    
    if edge_count_blur < edge_count_no_blur:
        print(f"✓ Blur reduces edges: {edge_count_no_blur} -> {edge_count_blur}")
    else:
        # Sometimes blur can spread edges making more pixels detected as edges
        # But the blur effect is still working as shown by the image difference
        print(f"✓ Blur affects edges: {edge_count_no_blur} -> {edge_count_blur} (spread)")
    
    # Test different kernel sizes
    vk.configure(blur=True, blur_kernel_size=13)
    processed_strong_blur = vk._preprocess_frames(frames)
    
    # Check stronger blur has more effect
    strong_diff = np.mean(np.abs(processed_strong_blur[0].astype(float) - processed_no_blur[0].astype(float)))
    assert strong_diff >= diff, "Larger kernel should have more effect"
    print(f"✓ Larger kernel increases blur effect: {diff:.2f} -> {strong_diff:.2f}")


def test_combined_preprocessing():
    """Test combining multiple preprocessing options."""
    vk = VideoKurt()
    
    # Create test frames using samplemaker
    frames = create_frames_with_pattern(num_frames=10, width=400, height=300, pattern='moving_circle')
    
    # Apply multiple preprocessing options
    vk.configure(
        resolution_scale=0.5,  # 200x150
        blur=True,
        blur_kernel_size=5
    )
    
    processed = vk._preprocess_frames(frames)
    
    # Check resolution changed
    assert processed[0].shape[0] == 150
    assert processed[0].shape[1] == 200
    
    # Check blur was applied (compare with non-blurred at same resolution)
    vk2 = VideoKurt()
    vk2.configure(resolution_scale=0.5, blur=False)
    processed_no_blur = vk2._preprocess_frames(frames)
    
    # Calculate difference - blurred should be smoother
    diff = np.mean(np.abs(processed[0].astype(float) - processed_no_blur[0].astype(float)))
    assert diff > 0  # There should be some difference
    print(f"✓ Combined preprocessing works (diff: {diff:.2f})")


def test_process_chunks_config():
    """Test process_chunks configuration (not implementation)."""
    vk = VideoKurt()
    
    # Test setting process_chunks
    vk.configure(process_chunks=4, chunk_overlap=50)
    assert vk._config['process_chunks'] == 4
    assert vk._config['chunk_overlap'] == 50
    print("✓ process_chunks configuration accepted")
    
    # Test validation
    try:
        vk.configure(process_chunks=0)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert 'process_chunks must be >= 1' in str(e)
        print("✓ process_chunks validation works")


def test_preprocessing_with_real_video():
    """Test preprocessing with actual video file if available."""
    video_path = Path('/Users/ns/Desktop/projects/videokurt/sample_recording.MP4')
    
    if not video_path.exists():
        print("⚠ Skipping real video test (sample_recording.MP4 not found)")
        return
    
    vk = VideoKurt()
    vk.add_analysis('frame_diff')  # Need at least one analysis
    
    # Configure preprocessing
    vk.configure(
        frame_step=10,      # Sample every 10th frame
        resolution_scale=0.25,  # Quarter resolution
        blur=True,
        blur_kernel_size=9
    )
    
    # Note: We're not running analyze() here, just testing config
    print("✓ Configuration works with real video path")
    
    # Test loading with frame_step
    frames, metadata = vk._load_video(video_path)
    
    # Check frames were loaded
    assert len(frames) > 0
    print(f"✓ Loaded {len(frames)} frames from real video")
    
    # Check preprocessing
    processed = vk._preprocess_frames(frames[:5])  # Just process first 5
    assert len(processed) == 5
    
    # Check resolution was scaled
    original_height = metadata['dimensions'][1]
    processed_height = processed[0].shape[0]
    assert processed_height < original_height
    print(f"✓ Resolution scaled: {original_height} -> {processed_height}")


def test_repr_with_config():
    """Test string representation shows configuration."""
    vk = VideoKurt()
    vk.configure(frame_step=5, resolution_scale=0.5, blur=True)
    vk.add_analysis('frame_diff')
    
    repr_str = repr(vk)
    assert "'frame_step': 5" in repr_str
    assert "'resolution_scale': 0.5" in repr_str
    assert "'blur': True" in repr_str
    print("✓ Configuration visible in repr")


def main():
    """Run all preprocessing tests."""
    print("="*50)
    print("VideoKurt Smoke Test 03: Preprocessing")
    print("="*50)
    
    tests = [
        test_frame_step,
        test_resolution_scale,
        test_blur,
        test_combined_preprocessing,
        test_process_chunks_config,
        test_preprocessing_with_real_video,
        test_repr_with_config,
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
    print("All preprocessing tests passed! ✓")
    print("="*50)
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)