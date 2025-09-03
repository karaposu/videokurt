"""FlowHSVViz analysis."""

import time
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import cv2

from .base import BaseAnalysis
from ..models import RawAnalysis


class FlowHSVViz(BaseAnalysis):
    """HSV visualization of optical flow."""
    
    METHOD_NAME = 'flow_hsv_viz'
    
    def __init__(self, downsample: float = 0.5,
                 max_magnitude: float = 20.0, saturation_boost: float = 1.5):
        """
        Args:
            downsample: Resolution scale
            max_magnitude: Maximum flow magnitude for normalization
            saturation_boost: Boost factor for saturation
        """
        super().__init__(downsample=downsample)
        self.max_magnitude = max_magnitude
        self.saturation_boost = saturation_boost
    
    def analyze(self, frames: List[np.ndarray]) -> RawAnalysis:
        """Convert optical flow to HSV color representation."""
        start_time = time.time()
        frames = self.preprocess_frames(frames)
        
        if len(frames) < 2:
            raise ValueError("Need at least 2 frames for flow visualization")
        
        # Convert to grayscale
        gray_frames = []
        for frame in frames:
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            gray_frames.append(gray)
        
        hsv_flows = []
        
        for i in range(1, len(gray_frames)):
            # Calculate optical flow
            flow = cv2.calcOpticalFlowFarneback(
                gray_frames[i-1], gray_frames[i], None,
                pyr_scale=0.5, levels=2, winsize=15,
                iterations=2, poly_n=5, poly_sigma=1.1, flags=0
            )
            
            # Convert flow to HSV
            h, w = flow.shape[:2]
            fx, fy = flow[:,:,0], flow[:,:,1]
            
            # Calculate magnitude and angle
            mag, ang = cv2.cartToPolar(fx, fy)
            
            # Create HSV image
            hsv = np.zeros((h, w, 3), dtype=np.uint8)
            
            # Hue = direction (shifted for better colors)
            hsv[:,:,0] = (ang * 180 / np.pi / 2 + 90) % 180
            
            # Saturation = magnitude (more motion = more saturated)
            sat_normalized = np.minimum(mag * 20 * self.saturation_boost, 255)
            hsv[:,:,1] = sat_normalized.astype(np.uint8)
            
            # Value = magnitude (normalized)
            normalized_mag = np.minimum(mag * self.max_magnitude, 255)
            hsv[:,:,2] = normalized_mag.astype(np.uint8)
            
            # Convert HSV to BGR for storage
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            hsv_flows.append(bgr)
        
        hsv_array = np.array(hsv_flows)
        
        return RawAnalysis(
            method=self.METHOD_NAME,
            data={'hsv_flow': hsv_array},
            parameters={
                'downsample': self.downsample,
                'max_magnitude': self.max_magnitude,
                'saturation_boost': self.saturation_boost
            },
            processing_time=time.time() - start_time,
            output_shapes={'hsv_flow': hsv_array.shape},
            dtype_info={'hsv_flow': str(hsv_array.dtype)}
        )


# =============================================================================
# Analysis Registry
# =============================================================================