"""Texture descriptor extraction using Local Binary Patterns and Gabor filters."""

import time
from typing import List, Optional, Dict, Any
import numpy as np
import cv2
from skimage.feature import local_binary_pattern

from .base import BaseAnalysis
from ..models import RawAnalysis


class TextureDescriptors(BaseAnalysis):
    """Extract texture features from frames using LBP and gradient analysis."""
    
    METHOD_NAME = 'texture_descriptors'
    
    def __init__(self, downsample: float = 1.0,
                 method: str = 'lbp',
                 lbp_radius: int = 1,
                 lbp_points: int = 8):
        """
        Args:
            downsample: Resolution scale (0.5 = half resolution)
            method: Texture method - 'lbp', 'gradient', 'gabor', 'combined'
            lbp_radius: Radius for LBP
            lbp_points: Number of points for LBP
        """
        super().__init__(downsample=downsample)
        self.method = method.lower()
        self.lbp_radius = lbp_radius
        self.lbp_points = lbp_points
    
    def analyze(self, frames: List[np.ndarray]) -> RawAnalysis:
        """Extract texture descriptors from each frame."""
        start_time = time.time()
        frames = self.preprocess_frames(frames)
        
        texture_features = []
        
        for frame in frames:
            # Convert to grayscale if needed
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            
            if self.method == 'lbp':
                features = self._compute_lbp(gray)
            elif self.method == 'gradient':
                features = self._compute_gradient_features(gray)
            elif self.method == 'gabor':
                features = self._compute_gabor_features(gray)
            elif self.method == 'combined':
                lbp = self._compute_lbp(gray)
                grad = self._compute_gradient_features(gray)
                # Stack features
                features = np.dstack([lbp, grad])
            else:
                # Default to gradient
                features = self._compute_gradient_features(gray)
            
            texture_features.append(features)
        
        texture_array = np.array(texture_features, dtype=np.float32)
        
        # Also compute texture statistics
        texture_stats = self._compute_texture_statistics(texture_array)
        
        return RawAnalysis(
            method=self.METHOD_NAME,
            data={
                'texture_features': texture_array,
                'texture_statistics': texture_stats
            },
            parameters={
                'downsample': self.downsample,
                'method': self.method,
                'lbp_radius': self.lbp_radius,
                'lbp_points': self.lbp_points
            },
            processing_time=time.time() - start_time,
            output_shapes={
                'texture_features': texture_array.shape,
                'texture_statistics': texture_stats.shape
            },
            dtype_info={
                'texture_features': str(texture_array.dtype),
                'texture_statistics': str(texture_stats.dtype)
            }
        )
    
    def _compute_lbp(self, gray: np.ndarray) -> np.ndarray:
        """Compute Local Binary Patterns."""
        # Compute LBP
        lbp = local_binary_pattern(gray, self.lbp_points, self.lbp_radius, method='uniform')
        
        # Normalize to 0-1 range
        lbp = lbp / lbp.max() if lbp.max() > 0 else lbp
        
        return lbp.astype(np.float32)
    
    def _compute_gradient_features(self, gray: np.ndarray) -> np.ndarray:
        """Compute gradient-based texture features."""
        # Compute gradients
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        
        # Compute gradient magnitude
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize
        magnitude = magnitude / magnitude.max() if magnitude.max() > 0 else magnitude
        
        return magnitude
    
    def _compute_gabor_features(self, gray: np.ndarray) -> np.ndarray:
        """Compute Gabor filter responses."""
        # Create Gabor kernels for different orientations
        features = []
        ksize = 31
        sigma = 4.0
        lambd = 10.0
        gamma = 0.5
        
        # 4 orientations
        for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
            kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)
            filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
            features.append(filtered)
        
        # Average responses
        gabor_response = np.mean(features, axis=0)
        
        # Normalize
        gabor_response = gabor_response / gabor_response.max() if gabor_response.max() > 0 else gabor_response
        
        return gabor_response
    
    def _compute_texture_statistics(self, texture_features: np.ndarray) -> np.ndarray:
        """Compute statistical summary of texture features."""
        stats = []
        
        for features in texture_features:
            # Flatten if multi-channel
            if len(features.shape) > 2:
                features = features.reshape(features.shape[0], -1)
            
            # Compute statistics
            feat_stats = {
                'mean': np.mean(features),
                'std': np.std(features),
                'energy': np.sum(features**2),
                'entropy': -np.sum(features * np.log(features + 1e-10))
            }
            
            stats.append([feat_stats['mean'], feat_stats['std'], 
                         feat_stats['energy'], feat_stats['entropy']])
        
        return np.array(stats, dtype=np.float32)