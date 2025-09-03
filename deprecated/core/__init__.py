"""Core VideoKurt components."""

# Import from the new frame_differencer module
from .frame_differencer import (
    FrameDifferencer,
    SimpleFrameDiff,
    HistogramFrameDiff,
    SSIMFrameDiff,
    HybridFrameDiff,
    DifferenceResult,
    create_differencer
)

__all__ = [
    'FrameDifferencer',
    'SimpleFrameDiff',
    'HistogramFrameDiff',
    'SSIMFrameDiff',
    'HybridFrameDiff',
    'DifferenceResult',
    'create_differencer'
]