"""Periodicity strength detection from frequency analysis."""

import numpy as np
from typing import Dict, Any

from ..base import BaseFeature


class PeriodicityStrength(BaseFeature):
    """Measure strength and characteristics of periodic patterns."""
    
    FEATURE_NAME = 'periodicity_strength'
    REQUIRED_ANALYSES = ['frequency_fft']
    
    def __init__(self, min_frequency: float = 0.1, max_frequency: float = 10.0):
        """
        Args:
            min_frequency: Minimum frequency to consider (Hz)
            max_frequency: Maximum frequency to consider (Hz)
        """
        super().__init__()
        self.min_frequency = min_frequency
        self.max_frequency = max_frequency
    
    def compute(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze periodicity strength from FFT data.
        
        Returns:
            Dict with periodicity metrics and dominant frequencies
        """
        self.validate_inputs(analysis_data)
        
        fft_analysis = analysis_data['frequency_fft']
        freq_spectrum = fft_analysis.data['frequency_spectrum']
        
        # Handle different spectrum shapes
        if len(freq_spectrum.shape) > 1:
            # Average across spatial dimensions
            if len(freq_spectrum.shape) == 3:
                avg_spectrum = np.mean(freq_spectrum, axis=(1, 2))
            else:
                avg_spectrum = np.mean(freq_spectrum, axis=1)
        else:
            avg_spectrum = freq_spectrum
        
        if len(avg_spectrum) < 2:
            return {
                'has_periodicity': False,
                'periodicity_score': 0,
                'dominant_frequencies': [],
                'period_estimates': []
            }
        
        # Get frequency bins (assuming normalized frequencies)
        n_freqs = len(avg_spectrum)
        freq_bins = np.fft.fftfreq(n_freqs * 2)[:n_freqs]  # Positive frequencies only
        
        # Filter frequency range
        valid_mask = (freq_bins >= self.min_frequency) & (freq_bins <= self.max_frequency)
        
        if not np.any(valid_mask):
            return {
                'has_periodicity': False,
                'periodicity_score': 0,
                'dominant_frequencies': [],
                'period_estimates': []
            }
        
        filtered_spectrum = avg_spectrum.copy()
        filtered_spectrum[~valid_mask] = 0
        
        # Normalize spectrum (exclude DC component)
        if np.max(filtered_spectrum[1:]) > 0:
            normalized = filtered_spectrum[1:] / np.max(filtered_spectrum[1:])
        else:
            normalized = filtered_spectrum[1:]
        
        # Find peaks
        peaks = self._find_peaks(normalized)
        
        # Compute periodicity score
        if len(peaks) > 0:
            # Score based on peak prominence
            peak_values = normalized[peaks]
            periodicity_score = float(np.max(peak_values))
            
            # Get dominant frequencies
            dominant_freqs = []
            period_estimates = []
            
            for peak_idx in peaks[:5]:  # Top 5 peaks
                freq = freq_bins[peak_idx + 1]  # +1 because we excluded DC
                if freq > 0:
                    dominant_freqs.append(float(freq))
                    period_estimates.append(1.0 / freq)  # Period in frames
            
            has_periodicity = periodicity_score > 0.3
        else:
            has_periodicity = False
            periodicity_score = 0
            dominant_freqs = []
            period_estimates = []
        
        # Analyze harmonics
        harmonic_structure = self._analyze_harmonics(normalized, peaks)
        
        return {
            'has_periodicity': has_periodicity,
            'periodicity_score': periodicity_score,
            'dominant_frequencies': dominant_freqs,
            'period_estimates': period_estimates,
            'harmonic_structure': harmonic_structure,
            'spectrum_entropy': self._compute_entropy(normalized)
        }
    
    def _find_peaks(self, spectrum: np.ndarray, prominence: float = 0.2) -> np.ndarray:
        """Find peaks in frequency spectrum."""
        
        peaks = []
        
        for i in range(1, len(spectrum) - 1):
            # Check if local maximum
            if spectrum[i] > spectrum[i-1] and spectrum[i] > spectrum[i+1]:
                # Check prominence
                if spectrum[i] > prominence:
                    peaks.append(i)
        
        # Sort by magnitude
        peaks = sorted(peaks, key=lambda x: spectrum[x], reverse=True)
        
        return np.array(peaks)
    
    def _analyze_harmonics(self, spectrum: np.ndarray, peaks: np.ndarray) -> Dict:
        """Analyze harmonic relationships between peaks."""
        if len(peaks) < 2:
            return {'has_harmonics': False, 'fundamental_freq': 0}
        
        # Check if peaks are harmonically related
        fundamental = peaks[0]
        harmonics = []
        
        for peak in peaks[1:]:
            ratio = peak / fundamental
            if abs(ratio - round(ratio)) < 0.1:  # Close to integer ratio
                harmonics.append(int(round(ratio)))
        
        has_harmonics = len(harmonics) > 0
        
        return {
            'has_harmonics': has_harmonics,
            'fundamental_idx': int(fundamental) if len(peaks) > 0 else 0,
            'harmonic_ratios': harmonics
        }
    
    def _compute_entropy(self, spectrum: np.ndarray) -> float:
        """Compute spectral entropy as measure of randomness."""
        # Normalize to probability distribution
        if np.sum(spectrum) > 0:
            prob = spectrum / np.sum(spectrum)
            # Compute entropy
            entropy = -np.sum(prob * np.log(prob + 1e-10))
            # Normalize by maximum entropy
            max_entropy = np.log(len(spectrum))
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        else:
            normalized_entropy = 1.0  # Maximum entropy for uniform distribution
        
        return float(normalized_entropy)