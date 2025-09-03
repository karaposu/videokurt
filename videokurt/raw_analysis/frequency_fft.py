"""FrequencyFFT analysis."""

import time
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import cv2

from .base import BaseAnalysis
from ..models import RawAnalysis


class FrequencyFFT(BaseAnalysis):
    """Frequency analysis using FFT."""
    
    METHOD_NAME = 'frequency_fft'
    
    def __init__(self, downsample: float = 0.1,  # Very small for FFT
                 window_size: int = 64, overlap: float = 0.5):
        """
        Args:
            downsample: Resolution scale (default 0.1 for FFT)
            window_size: Size of temporal window for FFT
            overlap: Overlap between windows (0.0 to 1.0)
        """
        super().__init__(downsample=downsample)
        self.window_size = window_size
        self.overlap = overlap
    
    def analyze(self, frames: List[np.ndarray]) -> RawAnalysis:
        """Analyze temporal frequency of pixel changes."""
        start_time = time.time()
        frames = self.preprocess_frames(frames)
        
        if len(frames) < self.window_size:
            raise ValueError(f"Need at least {self.window_size} frames for FFT analysis")
        
        # Convert to grayscale
        gray_frames = []
        for frame in frames:
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            gray_frames.append(gray.astype(np.float32) / 255.0)
        
        gray_array = np.array(gray_frames)  # [T, H, W]
        
        # Sample pixels for FFT (too expensive to do all pixels)
        h, w = gray_array.shape[1:]
        step = max(1, int(1 / np.sqrt(self.downsample)))  # Sample pixels based on downsample
        sampled_pixels = gray_array[:, ::step, ::step]  # [T, H', W']
        
        # Prepare for FFT analysis
        frequency_spectrums = []
        phase_spectrums = []
        
        # Sliding window FFT
        stride = int(self.window_size * (1 - self.overlap))
        
        for start in range(0, len(gray_frames) - self.window_size + 1, stride):
            window = sampled_pixels[start:start + self.window_size]
            
            # Apply FFT along time axis for each pixel
            fft_result = np.fft.fft(window, axis=0)
            
            # Get magnitude and phase
            magnitude = np.abs(fft_result)
            phase = np.angle(fft_result)
            
            # Average across spatial dimensions for summary
            avg_magnitude = np.mean(magnitude, axis=(1, 2))
            avg_phase = np.mean(phase, axis=(1, 2))
            
            frequency_spectrums.append(avg_magnitude[:self.window_size//2])  # Keep positive frequencies
            phase_spectrums.append(avg_phase[:self.window_size//2])
        
        return RawAnalysis(
            method=self.METHOD_NAME,
            data={
                'frequency_spectrum': np.array(frequency_spectrums),
                'phase_spectrum': np.array(phase_spectrums)
            },
            parameters={
                'downsample': self.downsample,
                'window_size': self.window_size,
                'overlap': self.overlap
            },
            processing_time=time.time() - start_time,
            output_shapes={
                'frequency_spectrum': np.array(frequency_spectrums).shape,
                'phase_spectrum': np.array(phase_spectrums).shape
            },
            dtype_info={
                'frequency_spectrum': 'float64',
                'phase_spectrum': 'float64'
            }
        )