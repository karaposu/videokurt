"""Feature extraction module for VideoKurt.

Provides three levels of features:
- Basic: Simple computations on raw data
- Middle: Pattern extraction with structure
- Advanced: Complex visual pattern detection
"""

from .base import BaseFeature
from .basic import BASIC_FEATURES
from .middle import MIDDLE_FEATURES
from .advanced import ADVANCED_FEATURES

# Combined registry of all features
ALL_FEATURES = {
    **BASIC_FEATURES,
    **MIDDLE_FEATURES,
    **ADVANCED_FEATURES
}

# Feature categories for easy access
FEATURES_BY_LEVEL = {
    'basic': BASIC_FEATURES,
    'middle': MIDDLE_FEATURES,
    'advanced': ADVANCED_FEATURES
}

__all__ = [
    'BaseFeature',
    'BASIC_FEATURES',
    'MIDDLE_FEATURES',
    'ADVANCED_FEATURES',
    'ALL_FEATURES',
    'FEATURES_BY_LEVEL',
]