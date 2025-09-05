#!/usr/bin/env python3
"""Script to refactor all basic features from BasicFeature to BaseFeature."""

import os
import re
from pathlib import Path

def refactor_feature_file(filepath):
    """Refactor a single feature file."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Skip if already refactored (has BaseFeature)
    if 'from ..base import BaseFeature' in content:
        print(f"  Already refactored: {filepath.name}")
        return False
    
    # Replace import
    content = content.replace(
        'from ..base import BasicFeature',
        'from ..base import BaseFeature'
    )
    
    # Replace class inheritance
    content = re.sub(
        r'class (\w+)\(BasicFeature\):',
        r'class \1(BaseFeature):',
        content
    )
    
    # Replace _compute_basic with compute and add validation
    content = re.sub(
        r'def _compute_basic\(self, analysis_data: Dict\[str, Any\]\)(.*?):\n(.*?""".*?""")',
        r'def compute(self, analysis_data: Dict[str, Any])\1:\n\2\n        self.validate_inputs(analysis_data)',
        content,
        flags=re.DOTALL
    )
    
    # Write back
    with open(filepath, 'w') as f:
        f.write(content)
    
    print(f"  Refactored: {filepath.name}")
    return True

def main():
    """Refactor all basic feature files."""
    features_dir = Path('/Users/ns/Desktop/projects/videokurt/videokurt/features/basic')
    
    feature_files = [
        'change_regions.py',
        'dct_energy.py',
        'dominant_flow_vector.py',
        'edge_density.py',
        'foreground_ratio.py',
        'frame_difference_percentile.py',
        'histogram_statistics.py',
        'motion_direction_histogram.py',
        'motion_magnitude.py',
        'repetition_indicator.py',
        'stability_score.py',
        'texture_uniformity.py'
    ]
    
    print("Refactoring basic features...")
    refactored_count = 0
    
    for filename in feature_files:
        filepath = features_dir / filename
        if filepath.exists():
            if refactor_feature_file(filepath):
                refactored_count += 1
        else:
            print(f"  File not found: {filename}")
    
    print(f"\nRefactored {refactored_count} files")

if __name__ == "__main__":
    main()