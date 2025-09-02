# VideoKurt Analysis Classes Structure

## Directory Structure

```
videokurt/
├── models.py                 # RawAnalysisResults, RawAnalysis dataclasses
├── analyses/                  # All analysis implementations
│   ├── __init__.py           # Exports all analyses and registry
│   ├── base.py              # BaseAnalysis class
│   │
│   ├── level1/              # Basic analyses (real-time capable)
│   │   ├── __init__.py
│   │   ├── frame_diff.py    # FrameDiff class
│   │   └── edge_canny.py    # EdgeCanny class
│   │
│   ├── level2/              # Intermediate analyses
│   │   ├── __init__.py
│   │   ├── frame_diff_advanced.py  # FrameDiffAdvanced class
│   │   └── contour_detection.py    # ContourDetection class
│   │
│   ├── level3/              # Advanced analyses
│   │   ├── __init__.py
│   │   ├── background_mog2.py      # BackgroundMOG2 class
│   │   ├── background_knn.py       # BackgroundKNN class
│   │   └── optical_flow_sparse.py  # OpticalFlowSparse class
│   │
│   └── level4/              # Complex analyses (computationally intensive)
│       ├── __init__.py
│       ├── optical_flow_dense.py   # OpticalFlowDense class
│       ├── motion_heatmap.py       # MotionHeatmap class
│       ├── frequency_fft.py        # FrequencyFFT class
│       └── flow_hsv_viz.py         # FlowHSVViz class
```

## File Contents

### `videokurt/models.py`
```python
"""Data models for VideoKurt analysis results."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

@dataclass
class RawAnalysis:
    """Result from a single analysis method."""
    method: str  # 'frame_diff', 'optical_flow_dense', etc.
    data: Dict[str, Any]  # Named outputs (can be np.ndarray, lists, etc.)
    parameters: Dict[str, Any]  # Parameters used for this analysis
    processing_time: float  # Time taken for this analysis
    memory_usage: Optional[int] = None  # Bytes used
    output_shapes: Dict[str, tuple]  # Shape of each output
    dtype_info: Dict[str, str]  # Data type of each output

@dataclass
class RawAnalysisResults:
    """Collection of all analysis results for a video."""
    # Video metadata
    dimensions: tuple[int, int]
    fps: float
    duration: float
    frame_count: int
    filename: Optional[str] = None
    
    # Analysis results
    analyses: Dict[str, RawAnalysis]  # Key: method name, Value: RawAnalysis
    
    # Legacy support
    binary_activity: np.ndarray
    binary_activity_confidence: np.ndarray
    timeline: Optional['Timeline'] = None
    segments: Optional[List['Segment']] = None
    
    # Performance
    total_elapsed_time: float
    
    # Convenience methods
    def get_analysis(self, method: str) -> Optional[RawAnalysis]:
        return self.analyses.get(method)
    
    def has_analysis(self, method: str) -> bool:
        return method in self.analyses
    
    def list_analyses(self) -> List[str]:
        return list(self.analyses.keys())
```

### `videokurt/analyses/base.py`
```python
"""Base class for all video analysis methods."""

from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np
import cv2
from ..models import RawAnalysis

class BaseAnalysis(ABC):
    """Base class for all video analysis methods."""
    
    METHOD_NAME = None  # Must be overridden by subclasses
    
    def __init__(self, downsample: float = 1.0, **kwargs):
        """
        Args:
            downsample: Resolution scale (0.5 = half resolution)
            **kwargs: Analysis-specific parameters
        """
        if self.METHOD_NAME is None:
            raise NotImplementedError("METHOD_NAME must be defined")
        
        self.downsample = downsample
        self.config = kwargs
        
    def preprocess_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Apply generic preprocessing like downsampling."""
        if self.downsample < 1.0:
            processed = []
            for frame in frames:
                h, w = frame.shape[:2]
                new_h = int(h * self.downsample)
                new_w = int(w * self.downsample)
                resized = cv2.resize(frame, (new_w, new_h))
                processed.append(resized)
            return processed
        return frames
    
    @abstractmethod
    def analyze(self, frames: List[np.ndarray]) -> RawAnalysis:
        """Analyze video frames and return results.
        
        Args:
            frames: List of video frames as numpy arrays
            
        Returns:
            RawAnalysis object containing the analysis results
        """
        pass
```

### `videokurt/analyses/level1/frame_diff.py`
```python
"""Simple frame differencing analysis."""

import time
from typing import List
import numpy as np
import cv2
from ..base import BaseAnalysis
from ...models import RawAnalysis

class FrameDiff(BaseAnalysis):
    """Simple frame differencing analysis."""
    
    METHOD_NAME = 'frame_diff'
    
    def __init__(self, downsample: float = 1.0, threshold: float = 0.1):
        """
        Args:
            downsample: Resolution scale (0.5 = half resolution)
            threshold: Threshold for detecting changes
        """
        super().__init__(downsample=downsample)
        self.threshold = threshold
    
    def analyze(self, frames: List[np.ndarray]) -> RawAnalysis:
        """Compute pixel-wise differences between consecutive frames."""
        start_time = time.time()
        frames = self.preprocess_frames(frames)
        
        pixel_diffs = []
        for i in range(len(frames) - 1):
            # Convert to grayscale if needed
            if len(frames[i].shape) == 3:
                gray1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(frames[i+1], cv2.COLOR_BGR2GRAY)
            else:
                gray1 = frames[i]
                gray2 = frames[i+1]
            
            diff = cv2.absdiff(gray1, gray2)
            pixel_diffs.append(diff)
        
        pixel_diff_array = np.array(pixel_diffs)
        
        return RawAnalysis(
            method=self.METHOD_NAME,
            data={'pixel_diff': pixel_diff_array},
            parameters={
                'threshold': self.threshold,
                'downsample': self.downsample
            },
            processing_time=time.time() - start_time,
            output_shapes={'pixel_diff': pixel_diff_array.shape},
            dtype_info={'pixel_diff': str(pixel_diff_array.dtype)}
        )
```

### `videokurt/analyses/__init__.py`
```python
"""VideoKurt analysis methods."""

# Import all analysis classes
from .base import BaseAnalysis

# Level 1 - Basic
from .level1.frame_diff import FrameDiff
from .level1.edge_canny import EdgeCanny

# Level 2 - Intermediate
from .level2.frame_diff_advanced import FrameDiffAdvanced
from .level2.contour_detection import ContourDetection

# Level 3 - Advanced
from .level3.background_mog2 import BackgroundMOG2
from .level3.background_knn import BackgroundKNN
from .level3.optical_flow_sparse import OpticalFlowSparse

# Level 4 - Complex
from .level4.optical_flow_dense import OpticalFlowDense
from .level4.motion_heatmap import MotionHeatmap
from .level4.frequency_fft import FrequencyFFT
from .level4.flow_hsv_viz import FlowHSVViz

# Analysis Registry
ANALYSIS_REGISTRY = {
    # Level 1: Basic
    'frame_diff': FrameDiff,
    'edge_canny': EdgeCanny,
    
    # Level 2: Intermediate
    'frame_diff_advanced': FrameDiffAdvanced,
    'contour_detection': ContourDetection,
    
    # Level 3: Advanced
    'background_mog2': BackgroundMOG2,
    'background_knn': BackgroundKNN,
    'optical_flow_sparse': OpticalFlowSparse,
    
    # Level 4: Complex
    'optical_flow_dense': OpticalFlowDense,
    'motion_heatmap': MotionHeatmap,
    'frequency_fft': FrequencyFFT,
    'flow_hsv_viz': FlowHSVViz
}

# Export all
__all__ = [
    'BaseAnalysis',
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
    'ANALYSIS_REGISTRY'
]
```

## Usage in VideoKurt

### `videokurt/videokurt.py`
```python
from .models import RawAnalysisResults, RawAnalysis
from .analyses import ANALYSIS_REGISTRY, BaseAnalysis

class VideoKurt:
    def analyze_video(self, video_path: str, 
                      analyses: Union[List[str], List[BaseAnalysis]] = None,
                      analysis_configs: Dict[str, dict] = None) -> RawAnalysisResults:
        # ... implementation using ANALYSIS_REGISTRY
```

## Benefits of This Structure

1. **Clear Organization**: Each complexity level in its own folder
2. **Easy Import**: Single import from `videokurt.analyses`
3. **Modular**: Each analysis in its own file
4. **Scalable**: Easy to add new analyses
5. **Discoverable**: Clear where to find each analysis
6. **Testable**: Each analysis can be tested independently

## Implementation Order

1. Create `models.py` with dataclasses ✓
2. Create `analyses/base.py` with BaseAnalysis
3. Create `analyses/level1/frame_diff.py` as first implementation
4. Port explorations to their respective level folders
5. Create `analyses/__init__.py` to tie everything together