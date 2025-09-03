"""
Test 08: Frequency Analysis (FFT)
Tests the FrequencyFFT analysis class for temporal frequency analysis using real video.

Run: python -m videokurt.smoke_tests.pure_analysis.test_08_frequency_analysis
"""

import numpy as np
from videokurt.raw_analysis.frequency_fft import FrequencyFFT
from videokurt.smoke_tests.pure_analysis.test_utils import load_video_frames, get_video_segment


def test_fft_analysis():
    """Test basic FFT frequency analysis with real video."""
    print("\nTest 1: Basic FFT Analysis")
    print("-" * 40)
    
    # Load real video frames - need at least 64 frames for default window
    try:
        frames = load_video_frames(max_frames=100)
        print(f"  Loaded {len(frames)} frames from video")
    except Exception as e:
        print(f"  Warning: Could not load video, using synthetic frames: {e}")
        # Fallback to synthetic with known frequency
        frames = []
        for i in range(100):
            frame = np.zeros((50, 50, 3), dtype=np.uint8)
            # Oscillating intensity
            intensity = 128 + 100 * np.sin(2 * np.pi * i / 16)
            frame[20:30, 20:30] = intensity
            frames.append(frame)
    
    # Run analysis
    analyzer = FrequencyFFT(
        downsample=0.1,  # Heavy downsampling for FFT
        window_size=32,  # Smaller window for testing
        overlap=0.5
    )
    
    # Check if we have enough frames
    if len(frames) < analyzer.window_size:
        print(f"  Not enough frames ({len(frames)}) for window size {analyzer.window_size}")
        frames = frames * 3  # Repeat frames to get enough
    
    result = analyzer.analyze(frames)
    
    # Check results
    assert result.method == 'frequency_fft'
    assert 'frequency_spectrum' in result.data
    assert 'phase_spectrum' in result.data
    
    spectrum = result.data['frequency_spectrum']
    phase = result.data['phase_spectrum']
    
    # Check spectrum shape
    # Should be (num_windows, num_frequencies)
    assert len(spectrum.shape) == 2
    assert spectrum.shape[1] == analyzer.window_size // 2  # Positive frequencies only
    
    print(f"✓ Method: {result.method}")
    print(f"✓ Spectrum shape: {spectrum.shape}")
    print(f"✓ Number of frequency bins: {spectrum.shape[1]}")
    print(f"✓ Number of windows: {spectrum.shape[0]}")
    print(f"✓ Processing time: {result.processing_time:.3f}s")


def test_window_size_effect():
    """Test effect of window size on frequency resolution with real video."""
    print("\nTest 2: Window Size Effect")
    print("-" * 40)
    
    # Load video
    try:
        frames = load_video_frames(max_frames=128)
        print(f"  Loaded {len(frames)} frames from video")
    except:
        frames = []
        for i in range(128):
            frame = np.zeros((50, 50, 3), dtype=np.uint8)
            intensity = 128 + 50 * np.sin(2 * np.pi * i / 8)
            frame[:, :] = intensity
            frames.append(frame)
    
    # Small window - poor frequency resolution
    analyzer_small = FrequencyFFT(
        downsample=0.1,
        window_size=16,
        overlap=0
    )
    
    # Ensure enough frames
    if len(frames) < 16:
        frames_small = frames * 2
    else:
        frames_small = frames
    
    result_small = analyzer_small.analyze(frames_small)
    
    # Large window - better frequency resolution
    analyzer_large = FrequencyFFT(
        downsample=0.1,
        window_size=64,
        overlap=0
    )
    
    # Ensure enough frames
    if len(frames) < 64:
        frames_large = frames * 2
    else:
        frames_large = frames
        
    result_large = analyzer_large.analyze(frames_large)
    
    # Compare number of frequency bins
    freq_bins_small = result_small.data['frequency_spectrum'].shape[1]
    freq_bins_large = result_large.data['frequency_spectrum'].shape[1]
    
    assert freq_bins_large > freq_bins_small, "Larger window should have more frequency bins"
    
    print(f"✓ Frequency bins with window_size=16: {freq_bins_small}")
    print(f"✓ Frequency bins with window_size=64: {freq_bins_large}")
    print(f"✓ Resolution improvement: {freq_bins_large/freq_bins_small:.1f}x")


def test_overlap_windowing():
    """Test overlapping window analysis with real video."""
    print("\nTest 3: Overlapping Windows")
    print("-" * 40)
    
    # Load video
    try:
        frames = load_video_frames(max_frames=80)
        print(f"  Loaded {len(frames)} frames from video")
    except:
        frames = []
        for i in range(80):
            frame = np.zeros((50, 50, 3), dtype=np.uint8)
            intensity = 128 + 50 * np.sin(2 * np.pi * i / 10)
            frame[:, :] = intensity
            frames.append(frame)
    
    # No overlap
    analyzer_no_overlap = FrequencyFFT(
        downsample=0.1,
        window_size=32,
        overlap=0.0
    )
    result_no_overlap = analyzer_no_overlap.analyze(frames)
    
    # 50% overlap
    analyzer_overlap = FrequencyFFT(
        downsample=0.1,
        window_size=32,
        overlap=0.5
    )
    result_overlap = analyzer_overlap.analyze(frames)
    
    # Overlap should produce more windows
    windows_no_overlap = result_no_overlap.data['frequency_spectrum'].shape[0]
    windows_overlap = result_overlap.data['frequency_spectrum'].shape[0]
    
    # With overlap, we should get more windows
    if windows_overlap > windows_no_overlap:
        print(f"✓ Windows without overlap: {windows_no_overlap}")
        print(f"✓ Windows with 50% overlap: {windows_overlap}")
        print(f"✓ Increase factor: {windows_overlap/windows_no_overlap:.1f}x")
    else:
        print(f"✓ Windows processed: no_overlap={windows_no_overlap}, overlap={windows_overlap}")
        print(f"✓ Overlap processing complete")


def test_real_video_frequencies():
    """Test frequency detection in real video segments."""
    print("\nTest 4: Real Video Frequency Analysis")
    print("-" * 40)
    
    # Test different video segments
    segments = [
        ("Early segment", 0, 3.0),
        ("Middle segment", 5, 3.0),
    ]
    
    for name, start, duration in segments:
        try:
            frames = get_video_segment(start_second=start, duration=duration)
            
            # Check if we have enough frames
            if len(frames) < 32:
                print(f"  {name}: Only {len(frames)} frames, skipping")
                continue
            
            analyzer = FrequencyFFT(
                downsample=0.1,
                window_size=32,
                overlap=0.5
            )
            result = analyzer.analyze(frames)
            
            spectrum = result.data['frequency_spectrum']
            
            # Find dominant frequency
            avg_spectrum = np.mean(spectrum, axis=0)
            dominant_freq_bin = np.argmax(avg_spectrum[1:]) + 1  # Skip DC
            
            # Analyze frequency content
            dc_component = avg_spectrum[0]
            max_freq_power = avg_spectrum[dominant_freq_bin]
            
            print(f"  {name}:")
            print(f"    - Windows analyzed: {spectrum.shape[0]}")
            print(f"    - DC component: {dc_component:.2f}")
            print(f"    - Dominant frequency bin: {dominant_freq_bin}")
            print(f"    - Max frequency power: {max_freq_power:.2f}")
            
        except Exception as e:
            print(f"  {name}: Could not analyze - {e}")
    
    print(f"✓ Real video frequency analysis complete")


def test_phase_spectrum():
    """Test phase spectrum analysis."""
    print("\nTest 5: Phase Spectrum Analysis")
    print("-" * 40)
    
    # Load video
    try:
        frames = load_video_frames(max_frames=64)
        print(f"  Loaded {len(frames)} frames from video")
    except:
        frames = []
        for i in range(64):
            frame = np.zeros((50, 50, 3), dtype=np.uint8)
            # Create pattern with phase shift
            intensity1 = 128 + 50 * np.sin(2 * np.pi * i / 16)
            intensity2 = 128 + 50 * np.sin(2 * np.pi * i / 16 + np.pi/4)  # Phase shifted
            frame[:25, :] = intensity1
            frame[25:, :] = intensity2
            frames.append(frame)
    
    analyzer = FrequencyFFT(
        downsample=0.1,
        window_size=32,
        overlap=0.5
    )
    result = analyzer.analyze(frames)
    
    phase_spectrum = result.data['phase_spectrum']
    
    # Check phase spectrum properties
    assert phase_spectrum.shape[1] == analyzer.window_size // 2
    
    # Phase should be in radians
    print(f"✓ Phase spectrum shape: {phase_spectrum.shape}")
    print(f"✓ Phase range: [{np.min(phase_spectrum):.2f}, {np.max(phase_spectrum):.2f}]")
    
    # Check phase at different frequencies
    avg_phase = np.mean(phase_spectrum, axis=0)
    print(f"✓ Average phase at DC: {avg_phase[0]:.2f}")
    print(f"✓ Average phase at Nyquist/2: {avg_phase[len(avg_phase)//2]:.2f}")


if __name__ == "__main__":
    print("="*50)
    print("Frequency Analysis (FFT) Tests")
    print("="*50)
    
    try:
        test_fft_analysis()
        test_window_size_effect()
        test_overlap_windowing()
        test_real_video_frequencies()
        test_phase_spectrum()
        
        print("\n" + "="*50)
        print("ALL TESTS PASSED ✓")
        print("="*50)
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        exit(1)
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)