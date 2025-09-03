"""Base classes for feature extraction at different levels."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Union, List
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


class BasicFeature(BaseFeature):
    """Base class for basic features - simple computations on raw data."""
    
    def compute(self, analysis_data: Dict[str, Any]) -> Union[np.ndarray, float]:
        """Basic features should return simple numerical values."""
        self.validate_inputs(analysis_data)
        return self._compute_basic(analysis_data)
    
    @abstractmethod
    def _compute_basic(self, analysis_data: Dict[str, Any]) -> Union[np.ndarray, float]:
        """Implement the basic feature computation."""
        pass


class MiddleFeature(BaseFeature):
    """Base class for middle features - pattern extraction with structure."""
    
    def compute(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Middle features return structured data."""
        self.validate_inputs(analysis_data)
        return self._compute_middle(analysis_data)
    
    @abstractmethod
    def _compute_middle(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Implement the middle feature computation."""
        pass


class AdvancedFeature(BaseFeature):
    """Base class for advanced features - complex pattern detection."""
    
    def __init__(self, basic_features: Dict[str, Any] = None, 
                 middle_features: Dict[str, Any] = None, **kwargs):
        """
        Args:
            basic_features: Pre-computed basic features
            middle_features: Pre-computed middle features
            **kwargs: Feature-specific parameters
        """
        super().__init__(**kwargs)
        self.basic_features = basic_features or {}
        self.middle_features = middle_features or {}
    
    def compute(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Advanced features return pattern classifications."""
        self.validate_inputs(analysis_data)
        return self._compute_advanced(analysis_data)
    
    @abstractmethod
    def _compute_advanced(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Implement the advanced feature computation."""
        pass