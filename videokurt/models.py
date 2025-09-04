"""VideoKurt raw analysis results models."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import json
import numpy as np


@dataclass
class FeatureResult:
    """Container for computed feature data.
    
    Provides a standardized way to store feature outputs along with
    their metadata, computation details, and dependencies.
    """
    # Feature identification
    name: str  # Feature name (e.g., 'binary_activity')
    
    # Feature output
    data: Any  # The actual feature data (np.array, dict, list, etc.)
    
    # Metadata
    metadata: Dict[str, Any]  # Parameters used for computation
    dtype: str  # Description of data type (e.g., 'binary_array', 'event_list')
    
    # Optional shape info for arrays
    shape: Optional[tuple] = None
    
    # Computation details
    compute_time: float = 0.0
    required_analyses: List[str] = field(default_factory=list)
    
    # Optional fields
    memory_usage: Optional[int] = None  # Bytes used
    
    def __repr__(self):
        return f"FeatureResult({self.name}, shape={self.shape})"
    
    @property
    def is_valid(self) -> bool:
        """Check if feature result contains valid data."""
        return self.data is not None


@dataclass
class RawAnalysis:
    """Result from a single analysis method.
    
    Each analysis returns this standardized object containing
    its raw outputs and metadata.
    """
    # Analysis identification
    method: str  # 'frame_diff', 'optical_flow_dense', etc.
    
    # Raw output data (analysis-specific)
    data: Dict[str, Any]  # Named outputs (can be np.ndarray, lists, etc.)
    
    # Metadata
    parameters: Dict[str, Any]  # Parameters used for this analysis
    processing_time: float  # Time taken for this analysis
    
    # Data shape info (for validation/debugging)
    output_shapes: Dict[str, Any]  # Shape of each output
    dtype_info: Dict[str, str]  # Data type of each output
    
    # Optional fields
    memory_usage: Optional[int] = None  # Bytes used (optional)


@dataclass
class RawAnalysisResults:
    """Raw results from VideoKurt analysis pipeline.
    
    Contains video metadata and raw analysis outputs only.
    Derived features and segments will be calculated later.
    """
    
    # Video/Frame properties
    dimensions: tuple[int, int]  # (width, height)
    fps: float  # Frames per second from video
    duration: float  # Total duration in seconds
    frame_count: int  # Total number of frames
    
    # Raw analysis results
    analyses: Dict[str, RawAnalysis]  # Key: method name, Value: RawAnalysis
    
    # Performance tracking
    elapsed_time: float  # Analysis time in seconds
    
    # Optional source info
    filename: Optional[str] = None  # Only for video files
    
    def get_analysis(self, method: str) -> Optional[RawAnalysis]:
        """Get analysis result by method name."""
        return self.analyses.get(method)
    
    def has_analysis(self, method: str) -> bool:
        """Check if analysis was run."""
        return method in self.analyses
    
    def list_analyses(self) -> List[str]:
        """List all analyses that were run."""
        return list(self.analyses.keys())
    
    @property
    def width(self) -> int:
        """Get video width."""
        return self.dimensions[0]
    
    @property
    def height(self) -> int:
        """Get video height."""
        return self.dimensions[1]
    
    @property
    def resolution(self) -> str:
        """Get resolution as string."""
        return f"{self.dimensions[0]}x{self.dimensions[1]}"
    
    
    def print_summary(self):
        """Print a human-readable summary."""
        print(f"\n{'='*50}")
        print("Raw Analysis Summary")
        print('='*50)
        if self.filename:
            print(f"File: {self.filename}")
        else:
            print(f"Source: Frame sequence")
        print(f"Resolution: {self.resolution} @ {self.fps:.1f} FPS")
        print(f"Duration: {self.duration:.1f}s ({self.frame_count} frames)")
        print(f"Analysis time: {self.elapsed_time:.2f}s")
        print(f"\nAnalyses run: {len(self.analyses)}")
        for method in self.list_analyses():
            analysis = self.get_analysis(method)
            print(f"  - {method}: {analysis.processing_time:.2f}s")
        print('='*50)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'video': {
                'filename': self.filename,
                'dimensions': list(self.dimensions),
                'fps': self.fps,
                'duration': self.duration,
                'frame_count': self.frame_count
            },
            'analysis': {
                'elapsed_time': round(self.elapsed_time, 3),
                'analyses_run': self.list_analyses(),
                'total_processing_time': sum(a.processing_time for a in self.analyses.values())
            },
            'analyses': {
                method: {
                    'processing_time': analysis.processing_time,
                    'parameters': analysis.parameters,
                    'output_shapes': analysis.output_shapes
                }
                for method, analysis in self.analyses.items()
            }
        }
    
    def to_json(self, indent=2) -> str:
        """Convert to clean JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    
    def _convert_numpy_types(self, obj):
        """Helper to convert numpy types for JSON serialization."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: self._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        return obj