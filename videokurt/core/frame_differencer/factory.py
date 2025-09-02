"""Factory functions for creating frame differencers."""

from typing import Any
from .base import FrameDifferencer
from .simple import SimpleFrameDiff
from .histogram import HistogramFrameDiff
from .ssim import SSIMFrameDiff
from .hybrid import HybridFrameDiff


def create_differencer(
    method: str = 'simple',
    **kwargs: Any
) -> FrameDifferencer:
    """Factory function to create appropriate differencer.
    
    Args:
        method: Type of differencer ('simple', 'histogram', 'ssim', 'hybrid')
        **kwargs: Configuration parameters for the differencer
        
    Returns:
        Configured FrameDifferencer instance
        
    Raises:
        ValueError: If unknown method is specified
    """
    differencers = {
        'simple': SimpleFrameDiff,
        'histogram': HistogramFrameDiff,
        'ssim': SSIMFrameDiff,
        'hybrid': HybridFrameDiff
    }
    
    if method not in differencers:
        raise ValueError(f"Unknown method: {method}. Choose from {list(differencers.keys())}")
    
    return differencers[method](**kwargs)