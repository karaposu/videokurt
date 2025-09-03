"""Base classes for frame differencing."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import numpy as np


@dataclass
class DifferenceResult:
    """Container for frame difference computation results."""
    
    score: float  # Normalized difference score (0.0-1.0)
    diff_mask: np.ndarray  # Pixel-level difference map
    metadata: Dict[str, Any]  # Algorithm-specific metadata
    
    @property
    def changed(self) -> bool:
        """Binary change indicator based on default threshold."""
        return self.score > 0.05
    
    def changed_with_threshold(self, threshold: float) -> bool:
        """Binary change indicator with custom threshold."""
        return self.score > threshold


class FrameDifferencer(ABC):
    """Abstract base class for frame differencing algorithms."""
    
    def __init__(self, **config):
        """Initialize with optional configuration."""
        self.config = config
        self._validate_config()
    
    @abstractmethod
    def compute_difference(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> DifferenceResult:
        """Compute difference between two frames.
        
        Args:
            frame1: First frame (BGR or grayscale)
            frame2: Second frame (BGR or grayscale)
            mask: Optional region of interest mask
            
        Returns:
            DifferenceResult containing score, mask, and metadata
        """
        pass
    
    def _validate_config(self):
        """Validate configuration parameters."""
        pass
    
    def _preprocess_frames(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess frames for comparison."""
        # Convert to grayscale if needed (before checking dimensions)
        try:
            import cv2
            if len(frame1.shape) == 3:
                frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            if len(frame2.shape) == 3:
                frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        except ImportError:
            # Fallback to simple averaging if cv2 not available
            if len(frame1.shape) == 3:
                frame1 = np.mean(frame1, axis=2).astype(np.uint8)
            if len(frame2.shape) == 3:
                frame2 = np.mean(frame2, axis=2).astype(np.uint8)
        
        # Now check dimensions after conversion
        if frame1.shape != frame2.shape:
            raise ValueError(f"Frame dimensions must match: {frame1.shape} != {frame2.shape}")
            
        return frame1, frame2