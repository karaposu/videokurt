"""Base class for all video analysis methods."""

import time
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import numpy as np
import cv2

from ..models import RawAnalysis


class BaseAnalysis(ABC):
    """Base class for all video analysis methods."""
    
    METHOD_NAME = None  # Must be overridden by subclasses
    
    def __init__(self, downsample: float = 1.0, **kwargs):
        """
        Args:
            downsample: Resolution scale (0.5 = half resolution)
            **kwargs: Analysis-specific parameters
        """
        if self.METHOD_NAME is None:
            raise NotImplementedError("METHOD_NAME must be defined")
        
        self.downsample = downsample
        self.config = kwargs
        
    def preprocess_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Apply generic preprocessing like downsampling."""
        if self.downsample < 1.0:
            processed = []
            for frame in frames:
                h, w = frame.shape[:2]
                new_h = int(h * self.downsample)
                new_w = int(w * self.downsample)
                resized = cv2.resize(frame, (new_w, new_h))
                processed.append(resized)
            return processed
        return frames
    
    @abstractmethod
    def analyze(self, frames: List[np.ndarray]) -> RawAnalysis:
        """Analyze video frames and return results.
        
        Args:
            frames: List of video frames as numpy arrays
            
        Returns:
            RawAnalysis object containing the analysis results
        """
        pass