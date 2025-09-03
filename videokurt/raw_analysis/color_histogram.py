"""Color histogram analysis for each frame."""

import time
from typing import List, Optional, Dict, Any
import numpy as np
import cv2

from .base import BaseAnalysis
from ..models import RawAnalysis


class ColorHistogram(BaseAnalysis):
    """Compute color/intensity histograms for each frame."""
    
    METHOD_NAME = 'color_histogram'
    
    def __init__(self, downsample: float = 1.0, 
                 bins: int = 256,
                 channels: str = 'gray',
                 normalize: bool = True):
        """
        Args:
            downsample: Resolution scale (0.5 = half resolution)
            bins: Number of histogram bins
            channels: Color space - 'gray', 'rgb', 'hsv', 'yuv'
            normalize: Whether to normalize histograms
        """
        super().__init__(downsample=downsample)
        self.bins = bins
        self.channels = channels.lower()
        self.normalize = normalize
    
    def analyze(self, frames: List[np.ndarray]) -> RawAnalysis:
        """Compute histograms for each frame."""
        start_time = time.time()
        frames = self.preprocess_frames(frames)
        
        histograms = []
        
        for frame in frames:
            # Convert to appropriate color space
            if self.channels == 'gray':
                if len(frame.shape) == 3:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    gray = frame
                
                # Compute histogram for grayscale
                hist = cv2.calcHist([gray], [0], None, [self.bins], [0, 256])
                hist = hist.flatten()
                
                if self.normalize:
                    hist = hist / np.sum(hist) if np.sum(hist) > 0 else hist
                
                histograms.append(hist)
                
            elif self.channels == 'rgb':
                if len(frame.shape) == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                
                # Compute histogram for each RGB channel
                rgb_hist = []
                for i in range(3):
                    hist = cv2.calcHist([frame], [i], None, [self.bins], [0, 256])
                    hist = hist.flatten()
                    if self.normalize:
                        hist = hist / np.sum(hist) if np.sum(hist) > 0 else hist
                    rgb_hist.append(hist)
                
                histograms.append(np.array(rgb_hist))
                
            elif self.channels == 'hsv':
                if len(frame.shape) == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                
                # H: 180 bins, S: 256 bins, V: 256 bins
                h_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180]).flatten()
                s_hist = cv2.calcHist([hsv], [1], None, [self.bins], [0, 256]).flatten()
                v_hist = cv2.calcHist([hsv], [2], None, [self.bins], [0, 256]).flatten()
                
                if self.normalize:
                    h_hist = h_hist / np.sum(h_hist) if np.sum(h_hist) > 0 else h_hist
                    s_hist = s_hist / np.sum(s_hist) if np.sum(s_hist) > 0 else s_hist
                    v_hist = v_hist / np.sum(v_hist) if np.sum(v_hist) > 0 else v_hist
                
                # Store as a list of arrays since they have different shapes
                histograms.append([h_hist, s_hist, v_hist])
                
            elif self.channels == 'yuv':
                if len(frame.shape) == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                
                yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
                
                # Compute histogram for each YUV channel
                yuv_hist = []
                for i in range(3):
                    hist = cv2.calcHist([yuv], [i], None, [self.bins], [0, 256])
                    hist = hist.flatten()
                    if self.normalize:
                        hist = hist / np.sum(hist) if np.sum(hist) > 0 else hist
                    yuv_hist.append(hist)
                
                histograms.append(np.array(yuv_hist))
        
        # Don't convert to numpy array if we have HSV (different shapes)
        histograms_array = histograms if self.channels == 'hsv' else np.array(histograms)
        
        return RawAnalysis(
            method=self.METHOD_NAME,
            data={'histograms': histograms_array},
            parameters={
                'downsample': self.downsample,
                'bins': self.bins,
                'channels': self.channels,
                'normalize': self.normalize
            },
            processing_time=time.time() - start_time,
            output_shapes={'histograms': f'{len(histograms_array)} frames' if self.channels == 'hsv' 
                          else histograms_array.shape},
            dtype_info={'histograms': 'list of arrays' if self.channels == 'hsv' 
                       else str(histograms_array.dtype)}
        )