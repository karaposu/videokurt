"""OpticalFlowSparse analysis."""

import time
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import cv2

from .base import BaseAnalysis
from ..models import RawAnalysis


class OpticalFlowSparse(BaseAnalysis):
    """Lucas-Kanade sparse optical flow analysis."""
    
    METHOD_NAME = 'optical_flow_sparse'
    
    def __init__(self, downsample: float = 1.0,  # Usually full res for accuracy
                 max_corners: int = 100, quality_level: float = 0.3,
                 min_distance: int = 7):
        """
        Args:
            downsample: Resolution scale (default 1.0 for accuracy)
            max_corners: Maximum number of corners to track
            quality_level: Quality level for corner detection
            min_distance: Minimum distance between corners
        """
        super().__init__(downsample=downsample)
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
    
    def analyze(self, frames: List[np.ndarray]) -> RawAnalysis:
        """Track specific feature points across frames."""
        start_time = time.time()
        frames = self.preprocess_frames(frames)
        
        if len(frames) < 2:
            raise ValueError("Need at least 2 frames for optical flow")
        
        # Convert to grayscale
        gray_frames = []
        for frame in frames:
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            gray_frames.append(gray)
        
        # Parameters for ShiTomasi corner detection
        feature_params = dict(
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            blockSize=7
        )
        
        # Parameters for Lucas-Kanade optical flow
        lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # Track points across all frames
        all_tracked_points = []
        all_point_status = []
        
        # Detect initial features in first frame
        p0 = cv2.goodFeaturesToTrack(gray_frames[0], mask=None, **feature_params)
        
        if p0 is not None:
            for i in range(1, len(gray_frames)):
                # Calculate optical flow
                p1, st, err = cv2.calcOpticalFlowPyrLK(
                    gray_frames[i-1], gray_frames[i], p0, None, **lk_params
                )
                
                # Select good points
                if p1 is not None:
                    good_new = p1[st == 1]
                    good_old = p0[st == 1]
                    
                    # Store tracked points
                    tracked = []
                    for j, (new, old) in enumerate(zip(good_new, good_old)):
                        tracked.append({
                            'id': j,
                            'old_pos': old.tolist(),
                            'new_pos': new.tolist(),
                            'dx': float(new[0] - old[0]),
                            'dy': float(new[1] - old[1])
                        })
                    
                    all_tracked_points.append(tracked)
                    all_point_status.append(st.flatten())
                    
                    # Update points for next iteration
                    p0 = good_new.reshape(-1, 1, 2)
                    
                    # Redetect features if too few points remain
                    if len(p0) < 20:
                        p0 = cv2.goodFeaturesToTrack(gray_frames[i], mask=None, **feature_params)
                else:
                    all_tracked_points.append([])
                    all_point_status.append(np.array([]))
        
        return RawAnalysis(
            method=self.METHOD_NAME,
            data={
                'tracked_points': all_tracked_points,
                'point_status': all_point_status
            },
            parameters={
                'downsample': self.downsample,
                'max_corners': self.max_corners,
                'quality_level': self.quality_level,
                'min_distance': self.min_distance
            },
            processing_time=time.time() - start_time,
            output_shapes={
                'tracked_points': f"List[{len(all_tracked_points)}]",
                'point_status': f"List[{len(all_point_status)}]"
            },
            dtype_info={
                'tracked_points': 'List[Dict]',
                'point_status': 'List[np.ndarray]'
            }
        )


# =============================================================================
# LEVEL 4: Complex Analyses (Computationally intensive)
# =============================================================================