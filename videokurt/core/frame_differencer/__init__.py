"""Frame Differencer module for VideoKurt.

This module provides various algorithms for detecting and quantifying
differences between video frames. It's the foundation for all motion
and change detection in VideoKurt.
"""

from .base import FrameDifferencer, DifferenceResult
from .simple import SimpleFrameDiff
from .histogram import HistogramFrameDiff
from .ssim import SSIMFrameDiff
from .hybrid import HybridFrameDiff
from .factory import create_differencer

__all__ = [
    'FrameDifferencer',
    'DifferenceResult', 
    'SimpleFrameDiff',
    'HistogramFrameDiff',
    'SSIMFrameDiff',
    'HybridFrameDiff',
    'create_differencer'
]