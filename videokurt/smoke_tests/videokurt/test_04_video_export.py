"""
Smoke Test 04: VideoKurt Video Export with Preprocessing
Tests save_video functionality with real video and various preprocessing options.

Run: python -m videokurt.smoke_tests.videokurt.test_04_video_export
"""

import sys
import numpy as np
import cv2
from pathlib import Path
from videokurt.videokurt import VideoKurt


def test_resolution_scale_export():
    """Test resolution_scale preprocessing and save to video."""
    video_path = Path('/Users/ns/Desktop/projects/videokurt/sample_recording.MP4')
    output_path = Path('/Users/ns/Desktop/projects/videokurt/test_resolution_scale.mp4')
    
    if not video_path.exists():
        print("⚠ Skipping test (sample_recording.MP4 not found)")
        return False
    
    vk = VideoKurt()
    vk.add_analysis('frame_diff')  # Need at least one analysis to load video
    
    # Configure with resolution scaling
    vk.configure(resolution_scale=0.25)  # Quarter resolution
    
    # Load and preprocess video (using first 150 frames for speed)
    print("Loading video...")
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    frames = []
    for i in range(150):  # Load first 150 frames
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    print(f"Loaded {len(frames)} frames")
    print(f"Original resolution: {frames[0].shape[1]}x{frames[0].shape[0]}")
    
    # Apply preprocessing
    processed = vk._preprocess_frames(frames)
    print(f"Scaled resolution: {processed[0].shape[1]}x{processed[0].shape[0]}")
    
    # Save video
    success = VideoKurt.save_video(processed, output_path, fps=fps)
    
    if success:
        print(f"✓ Saved resolution-scaled video to {output_path}")
        # Check file exists and has size
        if output_path.exists() and output_path.stat().st_size > 0:
            print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")
            return True
    
    print("✗ Failed to save resolution-scaled video")
    return False


def test_frame_step_export():
    """Test frame_step preprocessing and save to video."""
    video_path = Path('/Users/ns/Desktop/projects/videokurt/sample_recording.MP4')
    output_path = Path('/Users/ns/Desktop/projects/videokurt/test_frame_step.mp4')
    
    if not video_path.exists():
        print("⚠ Skipping test (sample_recording.MP4 not found)")
        return False
    
    vk = VideoKurt()
    vk.add_analysis('frame_diff')
    
    # Configure with frame step (every 5th frame)
    vk.configure(frame_step=5)
    
    # Load video with frame_step applied
    print("Loading video with frame_step=5...")
    frames, metadata = vk._load_video(video_path)
    
    # Limit to first 30 frames after stepping (equivalent to 150 original frames)
    frames = frames[:30]
    
    print(f"Loaded {len(frames)} frames (every 5th frame)")
    print(f"Resolution: {frames[0].shape[1]}x{frames[0].shape[0]}")
    
    # Save video (no additional preprocessing needed, frame_step already applied)
    success = VideoKurt.save_video(frames, output_path, fps=metadata['fps']/5)  # Adjust fps
    
    if success:
        print(f"✓ Saved frame-stepped video to {output_path}")
        if output_path.exists() and output_path.stat().st_size > 0:
            print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")
            return True
    
    print("✗ Failed to save frame-stepped video")
    return False


def test_blur_export():
    """Test blur preprocessing and save to video."""
    video_path = Path('/Users/ns/Desktop/projects/videokurt/sample_recording.MP4')
    output_path = Path('/Users/ns/Desktop/projects/videokurt/test_blur.mp4')
    
    if not video_path.exists():
        print("⚠ Skipping test (sample_recording.MP4 not found)")
        return False
    
    vk = VideoKurt()
    vk.add_analysis('frame_diff')
    
    # Configure with blur
    vk.configure(blur=True, blur_kernel_size=21)  # Strong blur
    
    # Load video
    print("Loading video...")
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    frames = []
    for i in range(150):  # Load first 150 frames
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    print(f"Loaded {len(frames)} frames")
    print(f"Resolution: {frames[0].shape[1]}x{frames[0].shape[0]}")
    
    # Apply preprocessing
    print("Applying blur (kernel_size=21)...")
    processed = vk._preprocess_frames(frames)
    
    # Save video
    success = VideoKurt.save_video(processed, output_path, fps=fps)
    
    if success:
        print(f"✓ Saved blurred video to {output_path}")
        if output_path.exists() and output_path.stat().st_size > 0:
            print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")
            return True
    
    print("✗ Failed to save blurred video")
    return False


def test_combined_preprocessing_export():
    """Test all three preprocessing options combined and save to video."""
    video_path = Path('/Users/ns/Desktop/projects/videokurt/sample_recording.MP4')
    output_path = Path('/Users/ns/Desktop/projects/videokurt/test_combined.mp4')
    
    if not video_path.exists():
        print("⚠ Skipping test (sample_recording.MP4 not found)")
        return False
    
    vk = VideoKurt()
    vk.add_analysis('frame_diff')
    
    # Configure with all preprocessing options
    vk.configure(
        frame_step=3,           # Every 3rd frame
        resolution_scale=0.5,   # Half resolution
        blur=True,              # Apply blur
        blur_kernel_size=9      # Moderate blur
    )
    
    # Load video with frame_step
    print("Loading video with combined preprocessing...")
    frames, metadata = vk._load_video(video_path)
    
    # Limit to first 50 frames after stepping
    frames = frames[:50]
    
    print(f"Loaded {len(frames)} frames (every 3rd frame)")
    print(f"Original resolution: {frames[0].shape[1]}x{frames[0].shape[0]}")
    
    # Apply resolution scaling and blur
    processed = vk._preprocess_frames(frames)
    print(f"After preprocessing: {processed[0].shape[1]}x{processed[0].shape[0]}")
    print("  - frame_step=3 applied during loading")
    print("  - resolution_scale=0.5 applied")
    print("  - blur with kernel_size=9 applied")
    
    # Save video
    success = VideoKurt.save_video(processed, output_path, fps=metadata['fps']/3)  # Adjust fps
    
    if success:
        print(f"✓ Saved combined preprocessing video to {output_path}")
        if output_path.exists() and output_path.stat().st_size > 0:
            print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")
            return True
    
    print("✗ Failed to save combined preprocessing video")
    return False


def test_grayscale_export():
    """Test saving grayscale frames (bonus test)."""
    from videokurt.samplemaker import create_frames_with_pattern
    
    output_path = Path('/Users/ns/Desktop/projects/videokurt/test_grayscale.mp4')
    
    # Create grayscale test frames
    print("Creating grayscale test frames...")
    color_frames = create_frames_with_pattern(
        num_frames=30, 
        width=200, 
        height=150, 
        pattern='moving_circle'
    )
    
    # Convert to grayscale
    gray_frames = []
    for frame in color_frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frames.append(gray)
    
    print(f"Created {len(gray_frames)} grayscale frames")
    print(f"Frame shape: {gray_frames[0].shape} (grayscale)")
    
    # Save video (should handle grayscale automatically)
    success = VideoKurt.save_video(gray_frames, output_path, fps=15)
    
    if success:
        print(f"✓ Saved grayscale video to {output_path}")
        if output_path.exists() and output_path.stat().st_size > 0:
            print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")
            return True
    
    print("✗ Failed to save grayscale video")
    return False


def cleanup_test_videos():
    """Optional: Remove test videos after tests."""
    test_files = [
        'test_resolution_scale.mp4',
        'test_frame_step.mp4',
        'test_blur.mp4',
        'test_combined.mp4',
        'test_grayscale.mp4'
    ]
    
    root_dir = Path('/Users/ns/Desktop/projects/videokurt')
    
    print("\nCleanup option:")
    for filename in test_files:
        filepath = root_dir / filename
        if filepath.exists():
            print(f"  Found: {filename} ({filepath.stat().st_size / 1024:.1f} KB)")
    
    # Note: Not automatically deleting - just showing what was created


def main():
    """Run all video export tests."""
    print("="*50)
    print("VideoKurt Smoke Test 04: Video Export")
    print("="*50)
    
    video_path = Path('/Users/ns/Desktop/projects/videokurt/sample_recording.MP4')
    if not video_path.exists():
        print(f"\n✗ Error: sample_recording.MP4 not found at {video_path}")
        print("  This test requires the sample video file.")
        return False
    
    tests = [
        ("Resolution Scale Export", test_resolution_scale_export),
        ("Frame Step Export", test_frame_step_export),
        ("Blur Export", test_blur_export),
        ("Combined Preprocessing Export", test_combined_preprocessing_export),
        ("Grayscale Export", test_grayscale_export),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*30}")
            print(f"Running: {test_name}")
            print('='*30)
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ {test_name} error: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("Test Summary:")
    print("="*50)
    
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    # Show created files
    cleanup_test_videos()
    
    all_passed = all(success for _, success in results)
    
    if all_passed:
        print("\n" + "="*50)
        print("All video export tests passed! ✓")
        print("Check the root directory for exported test videos.")
        print("="*50)
    else:
        print("\n" + "="*50)
        print("Some tests failed. Check the output above.")
        print("="*50)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)