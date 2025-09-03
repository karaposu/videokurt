"""Discrete Cosine Transform (DCT) analysis for perceptual hashing."""

import time
from typing import List, Optional, Dict, Any
import numpy as np
import cv2

from .base import BaseAnalysis
from ..models import RawAnalysis


class DCTTransform(BaseAnalysis):
    """Apply DCT transform to frames for frequency domain analysis."""
    
    METHOD_NAME = 'dct_transform'
    
    def __init__(self, downsample: float = 1.0,
                 block_size: int = 32,
                 keep_coeffs: int = 64):
        """
        Args:
            downsample: Resolution scale (0.5 = half resolution)
            block_size: Size to resize frames before DCT (e.g., 32x32)
            keep_coeffs: Number of DCT coefficients to keep
        """
        super().__init__(downsample=downsample)
        self.block_size = block_size
        self.keep_coeffs = keep_coeffs
    
    def analyze(self, frames: List[np.ndarray]) -> RawAnalysis:
        """Apply DCT transform to each frame."""
        start_time = time.time()
        frames = self.preprocess_frames(frames)
        
        dct_coefficients = []
        
        for frame in frames:
            # Convert to grayscale if needed
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            
            # Resize to block size for consistent DCT
            resized = cv2.resize(gray, (self.block_size, self.block_size))
            
            # Convert to float for DCT
            float_img = np.float32(resized)
            
            # Apply 2D DCT
            dct = cv2.dct(float_img)
            
            # Keep only top-left coefficients (low frequencies)
            # These contain most of the image energy
            coeffs_side = int(np.sqrt(self.keep_coeffs))
            if coeffs_side > self.block_size:
                coeffs_side = self.block_size
            
            # Extract top-left block of DCT coefficients
            dct_subset = dct[:coeffs_side, :coeffs_side].copy()
            
            # Flatten for easier processing
            dct_flat = dct_subset.flatten()
            
            # Pad if necessary to maintain consistent size
            if len(dct_flat) < self.keep_coeffs:
                dct_flat = np.pad(dct_flat, (0, self.keep_coeffs - len(dct_flat)))
            elif len(dct_flat) > self.keep_coeffs:
                dct_flat = dct_flat[:self.keep_coeffs]
            
            dct_coefficients.append(dct_flat)
        
        dct_array = np.array(dct_coefficients, dtype=np.float32)
        
        # Also compute perceptual hash from DCT
        perceptual_hashes = self._compute_perceptual_hash(dct_array)
        
        return RawAnalysis(
            method=self.METHOD_NAME,
            data={
                'dct_coefficients': dct_array,
                'perceptual_hashes': perceptual_hashes
            },
            parameters={
                'downsample': self.downsample,
                'block_size': self.block_size,
                'keep_coeffs': self.keep_coeffs
            },
            processing_time=time.time() - start_time,
            output_shapes={
                'dct_coefficients': dct_array.shape,
                'perceptual_hashes': perceptual_hashes.shape
            },
            dtype_info={
                'dct_coefficients': str(dct_array.dtype),
                'perceptual_hashes': str(perceptual_hashes.dtype)
            }
        )
    
    def _compute_perceptual_hash(self, dct_coeffs: np.ndarray) -> np.ndarray:
        """Compute perceptual hash from DCT coefficients.
        
        Args:
            dct_coeffs: Array of DCT coefficients per frame
            
        Returns:
            Binary hash array per frame
        """
        hashes = []
        
        for coeffs in dct_coeffs:
            # Compute median of coefficients (excluding DC component)
            median = np.median(coeffs[1:])  # Skip first coefficient (DC)
            
            # Create binary hash: 1 if coefficient > median, 0 otherwise
            hash_bits = (coeffs > median).astype(np.uint8)
            
            # Pack bits into bytes for compact storage
            # Each 8 bits becomes 1 byte
            hash_bytes = []
            for i in range(0, len(hash_bits), 8):
                byte_bits = hash_bits[i:i+8]
                if len(byte_bits) < 8:
                    byte_bits = np.pad(byte_bits, (0, 8 - len(byte_bits)))
                byte_val = np.packbits(byte_bits)[0]
                hash_bytes.append(byte_val)
            
            hashes.append(np.array(hash_bytes, dtype=np.uint8))
        
        return np.array(hashes)