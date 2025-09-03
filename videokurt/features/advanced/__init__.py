"""Advanced features - complex visual pattern detection."""

from .scene_detection import SceneDetection
from .camera_movement import CameraMovement
from .scrolling_detection import ScrollingDetection
from .ui_change_detection import UIChangeDetection
from .app_window_switching import AppWindowSwitching
from .motion_pattern_classification import MotionPatternClassification
from .shot_type_detection import ShotTypeDetection
from .transition_type_detection import TransitionTypeDetection
from .visual_anomaly_detection import VisualAnomalyDetection
from .repetitive_pattern_classification import RepetitivePatternClassification
from .motion_coherence_patterns import MotionCoherencePatterns
from .structural_change_patterns import StructuralChangePatterns

ADVANCED_FEATURES = {
    'scene_detection': SceneDetection,
    'camera_movement': CameraMovement,
    'scrolling_detection': ScrollingDetection,
    'ui_change_detection': UIChangeDetection,
    'app_window_switching': AppWindowSwitching,
    'motion_pattern_classification': MotionPatternClassification,
    'shot_type_detection': ShotTypeDetection,
    'transition_type_detection': TransitionTypeDetection,
    'visual_anomaly_detection': VisualAnomalyDetection,
    'repetitive_pattern_classification': RepetitivePatternClassification,
    'motion_coherence_patterns': MotionCoherencePatterns,
    'structural_change_patterns': StructuralChangePatterns,
}

__all__ = [
    'SceneDetection',
    'CameraMovement',
    'ScrollingDetection',
    'UIChangeDetection',
    'AppWindowSwitching',
    'MotionPatternClassification',
    'ShotTypeDetection',
    'TransitionTypeDetection',
    'VisualAnomalyDetection',
    'RepetitivePatternClassification',
    'MotionCoherencePatterns',
    'StructuralChangePatterns',
    'ADVANCED_FEATURES',
]