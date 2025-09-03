"""Raw analysis module for VideoKurt.

This module contains all the raw analysis implementations.
Each analysis extracts pixel-level information from video frames.
"""

from .base import BaseAnalysis
from .frame_diff import FrameDiff
from .edge_canny import EdgeCanny
from .frame_diff_advanced import FrameDiffAdvanced
from .contour_detection import ContourDetection
from .background_mog2 import BackgroundMOG2
from .background_knn import BackgroundKNN
from .optical_flow_sparse import OpticalFlowSparse
from .optical_flow_dense import OpticalFlowDense
from .motion_heatmap import MotionHeatmap
from .frequency_fft import FrequencyFFT
from .flow_hsv_viz import FlowHSVViz
from .color_histogram import ColorHistogram
from .dct_transform import DCTTransform
from .texture_descriptors import TextureDescriptors

# Analysis registry for dynamic instantiation
ANALYSIS_REGISTRY = {
    'frame_diff': FrameDiff,
    'edge_canny': EdgeCanny,
    'frame_diff_advanced': FrameDiffAdvanced,
    'contour_detection': ContourDetection,
    'background_mog2': BackgroundMOG2,
    'background_knn': BackgroundKNN,
    'optical_flow_sparse': OpticalFlowSparse,
    'optical_flow_dense': OpticalFlowDense,
    'motion_heatmap': MotionHeatmap,
    'frequency_fft': FrequencyFFT,
    'flow_hsv_viz': FlowHSVViz,
    'color_histogram': ColorHistogram,
    'dct_transform': DCTTransform,
    'texture_descriptors': TextureDescriptors,
}

__all__ = [
    'BaseAnalysis',
    'ANALYSIS_REGISTRY',
    'FrameDiff',
    'EdgeCanny',
    'FrameDiffAdvanced',
    'ContourDetection',
    'BackgroundMOG2',
    'BackgroundKNN',
    'OpticalFlowSparse',
    'OpticalFlowDense',
    'MotionHeatmap',
    'FrequencyFFT',
    'FlowHSVViz',
    'ColorHistogram',
    'DCTTransform',
    'TextureDescriptors',
]