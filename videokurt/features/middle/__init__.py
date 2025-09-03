"""Middle features - pattern extraction and structured data."""

from .blob_tracking import BlobTracking
from .blob_stability import BlobStability
from .dwell_time_maps import DwellTimeMaps
from .zone_based_activity import ZoneBasedActivity
from .motion_trajectories import MotionTrajectories
from .interaction_zones import InteractionZones
from .activity_bursts import ActivityBursts
from .periodicity_strength import PeriodicityStrength
from .boundary_crossings import BoundaryCrossings
from .spatial_occupancy_grid import SpatialOccupancyGrid
from .temporal_activity_patterns import TemporalActivityPatterns
from .structural_similarity import StructuralSimilarity
from .perceptual_hashes import PerceptualHashes
from .connected_components import ConnectedComponents

MIDDLE_FEATURES = {
    'blob_tracking': BlobTracking,  # Includes blob count & cross-frame tracking
    'blob_stability': BlobStability,
    'dwell_time_maps': DwellTimeMaps,
    'zone_based_activity': ZoneBasedActivity,
    'motion_trajectories': MotionTrajectories,
    'interaction_zones': InteractionZones,
    'activity_bursts': ActivityBursts,
    'periodicity_strength': PeriodicityStrength,
    'boundary_crossings': BoundaryCrossings,
    'spatial_occupancy_grid': SpatialOccupancyGrid,
    'temporal_activity_patterns': TemporalActivityPatterns,
    'structural_similarity': StructuralSimilarity,
    'perceptual_hashes': PerceptualHashes,
    'connected_components': ConnectedComponents,
}

__all__ = [
    'BlobTracking',
    'BlobStability', 
    'DwellTimeMaps',
    'ZoneBasedActivity',
    'MotionTrajectories',
    'InteractionZones',
    'ActivityBursts',
    'PeriodicityStrength',
    'BoundaryCrossings',
    'SpatialOccupancyGrid',
    'TemporalActivityPatterns',
    'StructuralSimilarity',
    'PerceptualHashes',
    'ConnectedComponents',
    'MIDDLE_FEATURES',
]