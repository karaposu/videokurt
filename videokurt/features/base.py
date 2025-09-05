"""Base class for all feature extraction."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Union
import numpy as np


class BaseFeature(ABC):
    """Base class for all feature extractors."""
    
    FEATURE_NAME = None  # Must be overridden by subclasses
    REQUIRED_ANALYSES = []  # List of required raw analyses
    
    def __init__(self, **kwargs):
        """
        Args:
            **kwargs: Feature-specific parameters
        """
        if self.FEATURE_NAME is None:
            raise NotImplementedError("FEATURE_NAME must be defined")
        
        self.config = kwargs
    
    @abstractmethod
    def compute(self, analysis_data: Dict[str, Any]) -> Union[np.ndarray, float, Dict]:
        """Compute the feature from raw analysis data.
        
        Args:
            analysis_data: Dictionary of raw analysis results
            
        Returns:
            Feature value(s) - can be scalar, array, or dict
        """
        pass
    
    def validate_inputs(self, analysis_data: Dict[str, Any]) -> bool:
        """Check if required analyses are present."""
        for required in self.REQUIRED_ANALYSES:
            if required not in analysis_data:
                raise ValueError(f"Required analysis '{required}' not found in input data")
        return True

