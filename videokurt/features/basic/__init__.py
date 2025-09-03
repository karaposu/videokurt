"""Basic features - simple computations on raw analysis data."""

from .binary_activity import BinaryActivity
from .motion_magnitude import MotionMagnitude
from .motion_direction_histogram import MotionDirectionHistogram
from .edge_density import EdgeDensity
from .change_regions import ChangeRegions
from .stability_score import StabilityScore
from .repetition_indicator import RepetitionIndicator
from .foreground_ratio import ForegroundRatio
from .frame_difference_percentile import FrameDifferencePercentile
from .dominant_flow_vector import DominantFlowVector
from .histogram_statistics import HistogramStatistics
from .dct_energy import DCTEnergy
from .texture_uniformity import TextureUniformity

BASIC_FEATURES = {
    'binary_activity': BinaryActivity,
    'motion_magnitude': MotionMagnitude,
    'motion_direction_histogram': MotionDirectionHistogram,
    'edge_density': EdgeDensity,
    'change_regions': ChangeRegions,
    'stability_score': StabilityScore,
    'repetition_indicator': RepetitionIndicator,
    'foreground_ratio': ForegroundRatio,
    'frame_difference_percentile': FrameDifferencePercentile,
    'dominant_flow_vector': DominantFlowVector,
    'histogram_statistics': HistogramStatistics,
    'dct_energy': DCTEnergy,
    'texture_uniformity': TextureUniformity,
}

__all__ = [
    'BinaryActivity',
    'MotionMagnitude',
    'MotionDirectionHistogram',
    'EdgeDensity',
    'ChangeRegions',
    'StabilityScore',
    'RepetitionIndicator',
    'ForegroundRatio',
    'FrameDifferencePercentile',
    'DominantFlowVector',
    'HistogramStatistics',
    'DCTEnergy',
    'TextureUniformity',
    'BASIC_FEATURES',
]