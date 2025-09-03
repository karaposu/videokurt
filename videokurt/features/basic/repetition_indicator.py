"""Repetition detection from frequency analysis."""

import numpy as np
from typing import Dict, Any

from ..base import BasicFeature


class RepetitionIndicator(BasicFeature):
    """Detect repetitive patterns using FFT analysis."""
    
    FEATURE_NAME = 'repetition_indicator'
    REQUIRED_ANALYSES = ['frequency_fft']
    
    def __init__(self, peak_threshold: float = 0.3):
        """
        Args:
            peak_threshold: Threshold for detecting significant frequency peaks
        """
        super().__init__()
        self.peak_threshold = peak_threshold
    
    def _compute_basic(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect repetitive patterns from frequency spectrum.
        
        Returns:
            Dict with repetition score and dominant frequencies
        """
        fft_analysis = analysis_data['frequency_fft']
        freq_spectrum = fft_analysis.data['frequency_spectrum']
        
        # Analyze frequency peaks
        # Average spectrum across spatial dimensions if needed
        if len(freq_spectrum.shape) > 1:
            avg_spectrum = np.mean(freq_spectrum, axis=(1, 2)) if len(freq_spectrum.shape) > 2 else np.mean(freq_spectrum, axis=1)
        else:
            avg_spectrum = freq_spectrum
        
        # Find peaks in spectrum (excluding DC component)
        if len(avg_spectrum) > 1:
            normalized = avg_spectrum[1:] / np.max(avg_spectrum[1:]) if np.max(avg_spectrum[1:]) > 0 else avg_spectrum[1:]
            peaks = np.where(normalized > self.peak_threshold)[0]
            
            if len(peaks) > 0:
                # Found repetitive patterns
                repetition_score = float(np.max(normalized))
                dominant_freq_idx = peaks[np.argmax(normalized[peaks])] + 1
                
                return {
                    'has_repetition': True,
                    'repetition_score': repetition_score,
                    'dominant_frequency_idx': int(dominant_freq_idx),
                    'num_peaks': len(peaks)
                }
            else:
                return {
                    'has_repetition': False,
                    'repetition_score': 0.0,
                    'dominant_frequency_idx': 0,
                    'num_peaks': 0
                }
        else:
            return {
                'has_repetition': False,
                'repetition_score': 0.0,
                'dominant_frequency_idx': 0,
                'num_peaks': 0
            }