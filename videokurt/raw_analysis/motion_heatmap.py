"""MotionHeatmap analysis."""

import time
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import cv2

from .base import BaseAnalysis
from ..models import RawAnalysis


class MotionHeatmap(BaseAnalysis):
    """Motion heatmap accumulation analysis."""
    
    METHOD_NAME = 'motion_heatmap'
    
    def __init__(self, downsample: float = 0.25,  # Heavy memory usage
                 decay_factor: float = 0.95, snapshot_interval: int = 30):
        """
        Args:
            downsample: Resolution scale (default 0.25 for memory)
            decay_factor: Decay factor for weighted heatmap
            snapshot_interval: Frames between snapshots
        """
        super().__init__(downsample=downsample)
        self.decay_factor = decay_factor
        self.snapshot_interval = snapshot_interval
    
    def analyze(self, frames: List[np.ndarray]) -> RawAnalysis:
        """Accumulate motion over time to find activity zones."""
        start_time = time.time()
        frames = self.preprocess_frames(frames)
        
        if len(frames) < 2:
            raise ValueError("Need at least 2 frames for motion heatmap")
        
        # Convert to grayscale
        gray_frames = []
        for frame in frames:
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            gray_frames.append(gray)
        
        h, w = gray_frames[0].shape
        
        # Initialize heatmaps
        cumulative_heatmap = np.zeros((h, w), dtype=np.float32)
        weighted_heatmap = np.zeros((h, w), dtype=np.float32)
        heatmap_snapshots = []
        
        # Process frame pairs
        for i in range(1, len(gray_frames)):
            # Calculate frame difference for motion
            diff = cv2.absdiff(gray_frames[i-1], gray_frames[i])
            motion_mask = diff.astype(np.float32) / 255.0
            
            # Update heatmaps
            cumulative_heatmap += motion_mask
            weighted_heatmap = weighted_heatmap * self.decay_factor + motion_mask
            
            # Take snapshots at intervals
            if i % self.snapshot_interval == 0:
                snapshot_time = i / len(gray_frames)
                heatmap_snapshots.append((snapshot_time, weighted_heatmap.copy()))
        
        # Normalize heatmaps
        if cumulative_heatmap.max() > 0:
            cumulative_heatmap = cumulative_heatmap / cumulative_heatmap.max()
        if weighted_heatmap.max() > 0:
            weighted_heatmap = weighted_heatmap / weighted_heatmap.max()
        
        # Convert to uint8 for storage
        cumulative_uint8 = (cumulative_heatmap * 255).astype(np.uint8)
        weighted_uint8 = (weighted_heatmap * 255).astype(np.uint8)
        
        return RawAnalysis(
            method=self.METHOD_NAME,
            data={
                'cumulative': cumulative_uint8,
                'weighted': weighted_uint8,
                'snapshots': heatmap_snapshots
            },
            parameters={
                'downsample': self.downsample,
                'decay_factor': self.decay_factor,
                'snapshot_interval': self.snapshot_interval
            },
            processing_time=time.time() - start_time,
            output_shapes={
                'cumulative': cumulative_uint8.shape,
                'weighted': weighted_uint8.shape,
                'snapshots': f"List[{len(heatmap_snapshots)}]"
            },
            dtype_info={
                'cumulative': 'uint8',
                'weighted': 'uint8',
                'snapshots': 'List[Tuple[float, np.ndarray]]'
            }
        )