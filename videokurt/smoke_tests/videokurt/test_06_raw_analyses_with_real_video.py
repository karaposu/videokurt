"""
Smoke Test 06: VideoKurt Raw Analyses with Real Video
Tests all raw analyses with sample_recording.MP4 (first 12.5 seconds only).

Run: python -m videokurt.smoke_tests.videokurt.test_06_raw_analyses_with_real_video
"""

import sys
import time
import cv2
import numpy as np
from pathlib import Path
from videokurt.videokurt import VideoKurt


def load_video_segment(video_path, duration_seconds=12.5):
    """Load first N seconds of video."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    frames_to_load = int(fps * duration_seconds)
    frames = []
    
    print(f"Loading {duration_seconds} seconds ({frames_to_load} frames at {fps:.1f} fps)...")
    
    for _ in range(frames_to_load):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    
    print(f"Loaded {len(frames)} frames")
    return frames, fps


def save_segment(frames, output_path, fps):
    """Save frames as video segment."""
    if not frames:
        return False
    
    success = VideoKurt.save_video(frames, output_path, fps=fps)
    if success:
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"✓ Saved segment: {output_path.name} ({size_mb:.1f} MB)")
    return success


def test_motion_analyses():
    """Test motion-related analyses on real video."""
    print("\nTesting motion analyses on real video...")
    
    video_path = Path('/Users/ns/Desktop/projects/videokurt/sample_recording.MP4')
    segment_path = Path('/Users/ns/Desktop/projects/videokurt/test_segment_motion.mp4')
    
    # Load first 12.5 seconds
    frames, fps = load_video_segment(video_path, 12.5)
    save_segment(frames, segment_path, fps)
    
    vk = VideoKurt()
    
    # Add motion-related analyses
    vk.add_analysis('frame_diff')
    vk.add_analysis('frame_diff_advanced')
    vk.add_analysis('optical_flow_dense')
    vk.add_analysis('optical_flow_sparse')
    vk.add_analysis('motion_heatmap')
    
    # Configure for reasonable performance
    vk.configure(
        frame_step=5,  # Every 5th frame
        resolution_scale=0.25,  # Quarter resolution
        blur=False
    )
    
    print(f"\nRunning {len(vk.list_analyses())} motion analyses...")
    start = time.time()
    results = vk.analyze(segment_path)
    elapsed = time.time() - start
    
    print(f"\n✓ Motion analyses completed in {elapsed:.2f}s")
    print(f"  Frames processed: {results.frame_count}")
    print(f"  Processing resolution: {int(results.dimensions[0] * 0.25)} x {int(results.dimensions[1] * 0.25)}")
    
    # Check each analysis
    for name in ['frame_diff', 'frame_diff_advanced', 'optical_flow_dense', 
                 'optical_flow_sparse', 'motion_heatmap']:
        if name in results.analyses:
            analysis = results.analyses[name]
            print(f"\n  {name}:")
            print(f"    - Processing time: {analysis.processing_time:.3f}s")
            for key, data in analysis.data.items():
                if hasattr(data, 'shape'):
                    print(f"    - {key}: shape={data.shape}, dtype={data.dtype}")
                elif isinstance(data, list):
                    print(f"    - {key}: {len(data)} items")
                else:
                    print(f"    - {key}: {type(data).__name__}")
    
    # Clean up
    segment_path.unlink(missing_ok=True)
    
    return True


def test_edge_analyses():
    """Test edge and contour detection on real video."""
    print("\nTesting edge/contour analyses on real video...")
    
    video_path = Path('/Users/ns/Desktop/projects/videokurt/sample_recording.MP4')
    segment_path = Path('/Users/ns/Desktop/projects/videokurt/test_segment_edge.mp4')
    
    # Load first 12.5 seconds
    frames, fps = load_video_segment(video_path, 12.5)
    save_segment(frames, segment_path, fps)
    
    vk = VideoKurt()
    
    # Add edge-related analyses
    vk.add_analysis('edge_canny')
    vk.add_analysis('contour_detection')
    
    # Configure with some preprocessing
    vk.configure(
        frame_step=10,  # Every 10th frame for faster processing
        resolution_scale=0.5,  # Half resolution
        blur=True,
        blur_kernel_size=5  # Light blur
    )
    
    print(f"\nRunning edge/contour analyses...")
    start = time.time()
    results = vk.analyze(segment_path)
    elapsed = time.time() - start
    
    print(f"\n✓ Edge analyses completed in {elapsed:.2f}s")
    
    # Check edge detection
    if 'edge_canny' in results.analyses:
        edge = results.analyses['edge_canny']
        print(f"\n  edge_canny:")
        print(f"    - Edge map shape: {edge.data['edge_map'].shape}")
        print(f"    - Gradient magnitude shape: {edge.data['gradient_magnitude'].shape}")
        print(f"    - Non-zero edges: {(edge.data['edge_map'] > 0).sum()}")
    
    # Check contour detection
    if 'contour_detection' in results.analyses:
        contours = results.analyses['contour_detection']
        print(f"\n  contour_detection:")
        total_contours = sum(len(c) for c in contours.data['contours'])
        print(f"    - Total contours detected: {total_contours}")
        print(f"    - Frames with contours: {len(contours.data['contours'])}")
    
    # Clean up
    segment_path.unlink(missing_ok=True)
    
    return True


def test_background_analyses():
    """Test background subtraction on real video."""
    print("\nTesting background subtraction on real video...")
    
    video_path = Path('/Users/ns/Desktop/projects/videokurt/sample_recording.MP4')
    segment_path = Path('/Users/ns/Desktop/projects/videokurt/test_segment_bg.mp4')
    
    # Load first 12.5 seconds
    frames, fps = load_video_segment(video_path, 12.5)
    save_segment(frames, segment_path, fps)
    
    vk = VideoKurt()
    
    # Add background subtraction analyses
    vk.add_analysis('background_knn', history=100, detect_shadows=True)
    vk.add_analysis('background_mog2', history=100, detect_shadows=False)
    
    # Configure for background subtraction
    vk.configure(
        frame_step=3,  # Every 3rd frame
        resolution_scale=0.25,  # Quarter resolution for speed
        blur=False
    )
    
    print(f"\nRunning background subtraction analyses...")
    start = time.time()
    results = vk.analyze(segment_path)
    elapsed = time.time() - start
    
    print(f"\n✓ Background analyses completed in {elapsed:.2f}s")
    
    # Compare KNN vs MOG2
    for method in ['background_knn', 'background_mog2']:
        if method in results.analyses:
            bg = results.analyses[method]
            mask = bg.data['foreground_mask']
            motion_pixels = (mask > 0).sum(axis=(1, 2))
            
            print(f"\n  {method}:")
            print(f"    - Foreground mask shape: {mask.shape}")
            print(f"    - Average motion pixels: {motion_pixels.mean():.1f}")
            print(f"    - Max motion pixels: {motion_pixels.max()}")
            print(f"    - Frames with motion: {(motion_pixels > 100).sum()}")
    
    # Clean up
    segment_path.unlink(missing_ok=True)
    
    return True


def test_color_frequency_analyses():
    """Test color and frequency analyses on real video."""
    print("\nTesting color/frequency analyses on real video...")
    
    video_path = Path('/Users/ns/Desktop/projects/videokurt/sample_recording.MP4')
    segment_path = Path('/Users/ns/Desktop/projects/videokurt/test_segment_color.mp4')
    
    # Load first 12.5 seconds
    frames, fps = load_video_segment(video_path, 12.5)
    save_segment(frames, segment_path, fps)
    
    vk = VideoKurt()
    
    # Add color and frequency analyses
    vk.add_analysis('color_histogram')
    vk.add_analysis('frequency_fft')  # Needs 64+ frames
    vk.add_analysis('dct_transform')
    
    # Configure - no frame_step to keep enough frames for FFT
    vk.configure(
        frame_step=2,  # Every other frame (should still have 64+ frames)
        resolution_scale=0.25,  # Quarter resolution
        blur=False
    )
    
    print(f"\nRunning color/frequency analyses...")
    start = time.time()
    results = vk.analyze(segment_path)
    elapsed = time.time() - start
    
    print(f"\n✓ Color/frequency analyses completed in {elapsed:.2f}s")
    
    # Check color histogram
    if 'color_histogram' in results.analyses:
        hist = results.analyses['color_histogram']
        print(f"\n  color_histogram:")
        print(f"    - Histograms shape: {hist.data['histograms'].shape}")
        print(f"    - Mean histogram value: {hist.data['histograms'].mean():.2f}")
    
    # Check FFT
    if 'frequency_fft' in results.analyses:
        fft = results.analyses['frequency_fft']
        print(f"\n  frequency_fft:")
        for key, data in fft.data.items():
            if hasattr(data, 'shape'):
                print(f"    - {key}: shape={data.shape}")
    
    # Check DCT
    if 'dct_transform' in results.analyses:
        dct = results.analyses['dct_transform']
        print(f"\n  dct_transform:")
        print(f"    - DCT coefficients shape: {dct.data['dct_coefficients'].shape}")
    
    # Clean up
    segment_path.unlink(missing_ok=True)
    
    return True


def test_texture_flow_analyses():
    """Test texture and flow visualization on real video."""
    print("\nTesting texture/flow analyses on real video...")
    
    video_path = Path('/Users/ns/Desktop/projects/videokurt/sample_recording.MP4')
    segment_path = Path('/Users/ns/Desktop/projects/videokurt/test_segment_texture.mp4')
    
    # Load first 12.5 seconds
    frames, fps = load_video_segment(video_path, 12.5)
    save_segment(frames, segment_path, fps)
    
    vk = VideoKurt()
    
    # Add texture and flow visualization
    vk.add_analysis('texture_descriptors')
    vk.add_analysis('flow_hsv_viz')
    
    # Configure
    vk.configure(
        frame_step=5,
        resolution_scale=0.25,
        blur=False
    )
    
    print(f"\nRunning texture/flow visualization analyses...")
    start = time.time()
    results = vk.analyze(segment_path)
    elapsed = time.time() - start
    
    print(f"\n✓ Texture/flow analyses completed in {elapsed:.2f}s")
    
    # Check texture descriptors
    if 'texture_descriptors' in results.analyses:
        texture = results.analyses['texture_descriptors']
        print(f"\n  texture_descriptors:")
        for key, data in texture.data.items():
            if hasattr(data, 'shape'):
                print(f"    - {key}: shape={data.shape}")
    
    # Check flow visualization
    if 'flow_hsv_viz' in results.analyses:
        flow = results.analyses['flow_hsv_viz']
        print(f"\n  flow_hsv_viz:")
        print(f"    - HSV flow shape: {flow.data['hsv_flow'].shape}")
        print(f"    - Mean flow value: {flow.data['hsv_flow'].mean():.2f}")
    
    # Clean up
    segment_path.unlink(missing_ok=True)
    
    return True


def test_combined_analyses():
    """Test all analyses together on real video segment."""
    print("\nTesting all analyses combined on real video...")
    
    video_path = Path('/Users/ns/Desktop/projects/videokurt/sample_recording.MP4')
    segment_path = Path('/Users/ns/Desktop/projects/videokurt/test_segment_all.mp4')
    
    # Load first 12.5 seconds
    frames, fps = load_video_segment(video_path, 12.5)
    save_segment(frames, segment_path, fps)
    
    vk = VideoKurt()
    
    # Add all analyses (except frequency_fft which needs special handling)
    all_analyses = [
        'frame_diff', 'edge_canny', 'frame_diff_advanced', 'contour_detection',
        'background_mog2', 'background_knn', 'optical_flow_sparse', 'optical_flow_dense',
        'motion_heatmap', 'flow_hsv_viz', 'color_histogram', 'dct_transform', 
        'texture_descriptors'
    ]
    
    for analysis in all_analyses:
        vk.add_analysis(analysis)
    
    # Configure for balanced performance
    vk.configure(
        frame_step=10,  # Every 10th frame
        resolution_scale=0.2,  # 20% resolution
        blur=False
    )
    
    print(f"\nRunning {len(all_analyses)} analyses together...")
    start = time.time()
    results = vk.analyze(segment_path)
    elapsed = time.time() - start
    
    print(f"\n✓ All analyses completed in {elapsed:.2f}s")
    print(f"  Average time per analysis: {elapsed/len(all_analyses):.2f}s")
    print(f"  Frames processed: {results.frame_count}")
    print(f"  Video duration: {results.duration:.1f}s")
    
    # Summary of results
    print(f"\n  Successful analyses: {len(results.analyses)}/{len(all_analyses)}")
    
    # Show timing for each
    timings = [(name, a.processing_time) for name, a in results.analyses.items()]
    timings.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n  Top 5 slowest analyses:")
    for name, t in timings[:5]:
        print(f"    - {name}: {t:.3f}s")
    
    print(f"\n  Top 5 fastest analyses:")
    for name, t in timings[-5:]:
        print(f"    - {name}: {t:.3f}s")
    
    # Clean up
    segment_path.unlink(missing_ok=True)
    
    return len(results.analyses) == len(all_analyses)


def test_performance_comparison():
    """Compare performance with different preprocessing settings."""
    print("\nTesting performance with different settings...")
    
    video_path = Path('/Users/ns/Desktop/projects/videokurt/sample_recording.MP4')
    segment_path = Path('/Users/ns/Desktop/projects/videokurt/test_segment_perf.mp4')
    
    # Load first 12.5 seconds
    frames, fps = load_video_segment(video_path, 12.5)
    save_segment(frames, segment_path, fps)
    
    # Test configurations
    configs = [
        {"name": "High Quality", "frame_step": 1, "resolution_scale": 1.0, "blur": False},
        {"name": "Balanced", "frame_step": 5, "resolution_scale": 0.5, "blur": False},
        {"name": "Fast", "frame_step": 10, "resolution_scale": 0.25, "blur": True},
    ]
    
    # Analyses to test
    test_analyses = ['frame_diff', 'edge_canny', 'optical_flow_dense']
    
    print(f"\nComparing {len(configs)} configurations with {len(test_analyses)} analyses:")
    
    for config in configs:
        vk = VideoKurt()
        for analysis in test_analyses:
            vk.add_analysis(analysis)
        
        vk.configure(
            frame_step=config['frame_step'],
            resolution_scale=config['resolution_scale'],
            blur=config['blur']
        )
        
        start = time.time()
        results = vk.analyze(segment_path)
        elapsed = time.time() - start
        
        print(f"\n  {config['name']}:")
        print(f"    - Settings: step={config['frame_step']}, scale={config['resolution_scale']}, blur={config['blur']}")
        print(f"    - Frames processed: {results.frame_count}")
        print(f"    - Total time: {elapsed:.2f}s")
        print(f"    - Time per frame: {elapsed/results.frame_count*1000:.1f}ms")
    
    # Clean up
    segment_path.unlink(missing_ok=True)
    
    return True


def main():
    """Run all real video analyses tests."""
    print("="*60)
    print("VideoKurt Smoke Test 06: Raw Analyses with Real Video")
    print("Using first 12.5 seconds of sample_recording.MP4")
    print("="*60)
    
    video_path = Path('/Users/ns/Desktop/projects/videokurt/sample_recording.MP4')
    
    if not video_path.exists():
        print(f"\n✗ Error: sample_recording.MP4 not found at {video_path}")
        print("  This test requires the sample video file.")
        return False
    
    # Get video info
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps
    cap.release()
    
    print(f"\nVideo info:")
    print(f"  - Resolution: {width}x{height}")
    print(f"  - FPS: {fps:.1f}")
    print(f"  - Total duration: {duration:.1f}s")
    print(f"  - Total frames: {total_frames}")
    print(f"  - Using first 12.5s ({int(fps * 12.5)} frames)")
    
    tests = [
        ("Motion analyses", test_motion_analyses),
        ("Edge/contour analyses", test_edge_analyses),
        ("Background analyses", test_background_analyses),
        ("Color/frequency analyses", test_color_frequency_analyses),
        ("Texture/flow analyses", test_texture_flow_analyses),
        ("Combined analyses", test_combined_analyses),
        ("Performance comparison", test_performance_comparison),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*40}")
            print(f"Running: {test_name}")
            print('='*40)
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ {test_name} error: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary:")
    print("="*60)
    
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(success for _, success in results)
    
    if all_passed:
        print("\n" + "="*60)
        print("All real video analyses tests passed! ✓")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("Some tests failed. Check the output above.")
        print("="*60)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)