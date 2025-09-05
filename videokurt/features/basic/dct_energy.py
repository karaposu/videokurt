"""DCT energy computation from DCT transform."""

import numpy as np
from typing import Dict, Any

from ..base import BaseFeature


class DCTEnergy(BaseFeature):
    """Compute energy from DCT coefficients."""
    
    FEATURE_NAME = 'dct_energy'
    REQUIRED_ANALYSES = ['dct_transform']
    
    def __init__(self, num_coeffs: int = 10):
        """
        Args:
            num_coeffs: Number of top coefficients to use for energy
        """
        super().__init__()
        self.num_coeffs = num_coeffs
    
    def compute(self, analysis_data: Dict[str, Any]) -> np.ndarray:
        """Compute energy from DCT coefficients.
        
        Returns:
            Array of DCT energy values per frame
        """
        self.validate_inputs(analysis_data)
        
        dct_analysis = analysis_data['dct_transform']
        dct_coeffs = dct_analysis.data['dct_coefficients']
        
        energies = []
        for coeffs in dct_coeffs:
            # Flatten coefficients if 2D
            if len(coeffs.shape) > 1:
                flat_coeffs = coeffs.flatten()
            else:
                flat_coeffs = coeffs
            
            # Sort by magnitude and take top N
            sorted_coeffs = np.sort(np.abs(flat_coeffs))[::-1]
            top_coeffs = sorted_coeffs[:self.num_coeffs]
            
            # Compute energy as sum of squares
            energy = np.sum(top_coeffs ** 2)
            energies.append(energy)
        
        return np.array(energies, dtype=np.float32)