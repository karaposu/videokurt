"""Edge density computation from Canny edge detection."""

import numpy as np
from typing import Dict, Any

from ..base import BasicFeature


class EdgeDensity(BasicFeature):
    """Compute percentage of edge pixels per frame."""
    
    FEATURE_NAME = 'edge_density'
    REQUIRED_ANALYSES = ['edge_canny']
    
    def __init__(self, use_gradient: bool = False):
        """
        Args:
            use_gradient: Use gradient magnitude instead of binary edges
        """
        super().__init__()
        self.use_gradient = use_gradient
    
    def _compute_basic(self, analysis_data: Dict[str, Any]) -> np.ndarray:
        """Compute edge density from edge detection.
        
        Returns:
            Array of edge density values (0-1) per frame
        """
        # Get edge data
        edge_analysis = analysis_data['edge_canny']
        
        if self.use_gradient:
            edges = edge_analysis.data['gradient_magnitude']
            # Use threshold for gradient magnitude
            threshold = 50
            densities = []
            for edge_map in edges:
                edge_pixels = np.sum(edge_map > threshold)
                total_pixels = edge_map.size
                density = edge_pixels / total_pixels
                densities.append(density)
        else:
            edges = edge_analysis.data['edge_map']
            densities = []
            for edge_map in edges:
                edge_pixels = np.sum(edge_map > 0)
                total_pixels = edge_map.size
                density = edge_pixels / total_pixels
                densities.append(density)
        
        return np.array(densities, dtype=np.float32)