"""
VideoKurt - Raw Visual Analysis System

Main interface for running computer vision analyses on video files.
Provides 11 different analyses that can be configured independently.
"""

import time
import cv2
import numpy as np
from typing import Dict, List, Optional, Union, Any
from pathlib import Path

from videokurt.models import RawAnalysisResults, RawAnalysis
from videokurt.analysis_models import (
    ANALYSIS_REGISTRY,
    BaseAnalysis,
    FrameDiff,
    EdgeCanny,
    FrameDiffAdvanced,
    ContourDetection,
    BackgroundMOG2,
    BackgroundKNN,
    OpticalFlowSparse,
    OpticalFlowDense,
    MotionHeatmap,
    FrequencyFFT,
    FlowHSVViz
)


class VideoLoadError(Exception):
    """Raised when video cannot be loaded."""
    pass


class AnalysisError(Exception):
    """Raised when analysis fails."""
    def __init__(self, message, partial_results=None):
        super().__init__(message)
        self.partial_results = partial_results


class VideoKurt:
    """
    Main interface for video analysis.
    
    Supports:
    - Running all or selective analyses
    - Custom configuration per analysis
    - Global configuration options
    - Performance optimization
    """
    
    def __init__(self,
                 max_frames: Optional[int] = None,
                 max_seconds: Optional[float] = None,
                 frame_step: int = 1,
                 downsample: float = 1.0,
                 parallel: bool = False,
                 max_workers: int = 4,
                 chunk_size: Optional[int] = None,
                 clear_frames: bool = False):
        """
        Initialize VideoKurt with global configuration.
        
        Args:
            max_frames: Maximum number of frames to process
            max_seconds: Maximum seconds to process (overrides max_frames)
            frame_step: Process every Nth frame (1 = all frames)
            downsample: Global downsampling factor (can be overridden per analysis)
            parallel: Run analyses in parallel (not implemented yet)
            max_workers: Number of parallel workers
            chunk_size: Process video in chunks (for memory management)
            clear_frames: Clear frame buffer after analysis
        """
        self.max_frames = max_frames
        self.max_seconds = max_seconds
        self.frame_step = frame_step
        self.global_downsample = downsample
        self.parallel = parallel
        self.max_workers = max_workers
        self.chunk_size = chunk_size
        self.clear_frames = clear_frames
    
    def analyze(self,
                video_path: Union[str, Path],
                analyses: Optional[Union[List[str], Dict[str, Optional[BaseAnalysis]]]] = None,
                **kwargs) -> RawAnalysisResults:
        """
        Analyze video with specified analyses.
        
        Args:
            video_path: Path to video file
            analyses: Can be:
                - None: Run all available analyses with defaults
                - List of strings: Run specified analyses by name with defaults
                - Dict mapping names to configured instances or None for defaults
            **kwargs: Additional parameters passed to analyses
        
        Returns:
            RawAnalysisResults containing all analysis outputs
        
        Examples:
            # Run all analyses
            results = vk.analyze("video.mp4")
            
            # Run specific analyses with defaults
            results = vk.analyze("video.mp4", analyses=['frame_diff', 'optical_flow_dense'])
            
            # Run with custom configurations
            results = vk.analyze("video.mp4", analyses={
                'frame_diff': FrameDiff(threshold=0.2),
                'optical_flow_dense': None  # Use defaults
            })
        """
        start_time = time.time()
        
        # Load video frames
        frames, metadata = self._load_video(video_path)
        
        # Determine which analyses to run
        analyses_to_run = self._prepare_analyses(analyses)
        
        # Run analyses
        analysis_results = {}
        partial_results = None
        
        try:
            for name, analyzer in analyses_to_run.items():
                try:
                    print(f"Running {name}...")
                    analysis_result = analyzer.analyze(frames)
                    analysis_results[name] = analysis_result
                except Exception as e:
                    print(f"Warning: {name} failed: {e}")
                    # Continue with other analyses
                    
        except Exception as e:
            # Save partial results if some analyses completed
            if analysis_results:
                partial_results = RawAnalysisResults(
                    dimensions=metadata['dimensions'],
                    fps=metadata['fps'],
                    duration=metadata['duration'],
                    frame_count=metadata['frame_count'],
                    analyses=analysis_results,
                    elapsed_time=time.time() - start_time,
                    filename=str(video_path)
                )
            raise AnalysisError(f"Analysis failed: {e}", partial_results)
        
        # Clear frames if requested
        if self.clear_frames:
            frames = None
        
        # Create results object
        results = RawAnalysisResults(
            dimensions=metadata['dimensions'],
            fps=metadata['fps'],
            duration=metadata['duration'],
            frame_count=metadata['frame_count'],
            analyses=analysis_results,
            elapsed_time=time.time() - start_time,
            filename=str(video_path)
        )
        
        return results
    
    def _load_video(self, video_path: Union[str, Path]) -> tuple[List[np.ndarray], Dict]:
        """
        Load video frames and metadata.
        
        Returns:
            Tuple of (frames list, metadata dict)
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise VideoLoadError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise VideoLoadError(f"Cannot open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        
        # Calculate frame limits
        max_frames = total_frames
        if self.max_seconds is not None:
            max_frames = min(max_frames, int(fps * self.max_seconds))
        if self.max_frames is not None:
            max_frames = min(max_frames, self.max_frames)
        
        # Load frames
        frames = []
        frame_count = 0
        frames_read = 0
        
        while frames_read < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Apply frame step
            if frame_count % self.frame_step == 0:
                frames.append(frame)
                frames_read += 1
            
            frame_count += 1
        
        cap.release()
        
        if len(frames) == 0:
            raise VideoLoadError(f"No frames could be loaded from {video_path}")
        
        metadata = {
            'dimensions': (width, height),
            'fps': fps,
            'duration': duration,
            'frame_count': len(frames),
            'original_frame_count': total_frames
        }
        
        return frames, metadata
    
    def _prepare_analyses(self, 
                         analyses: Optional[Union[List[str], Dict[str, Optional[BaseAnalysis]]]]) -> Dict[str, BaseAnalysis]:
        """
        Prepare analysis instances based on input.
        
        Returns:
            Dictionary mapping analysis names to configured instances
        """
        # If None, run all available analyses
        if analyses is None:
            return self._create_all_analyses()
        
        # If list of strings, create default instances
        if isinstance(analyses, list):
            result = {}
            for name in analyses:
                if name not in ANALYSIS_REGISTRY:
                    raise ValueError(f"Unknown analysis: {name}")
                result[name] = self._create_analysis(name)
            return result
        
        # If dict, use provided instances or create defaults
        if isinstance(analyses, dict):
            result = {}
            for name, instance in analyses.items():
                if name not in ANALYSIS_REGISTRY:
                    raise ValueError(f"Unknown analysis: {name}")
                
                if instance is None:
                    # Create default instance
                    result[name] = self._create_analysis(name)
                elif isinstance(instance, BaseAnalysis):
                    # Use provided instance
                    result[name] = instance
                else:
                    raise ValueError(f"Invalid analysis instance for {name}")
            return result
        
        raise ValueError(f"Invalid analyses argument type: {type(analyses)}")
    
    def _create_analysis(self, name: str) -> BaseAnalysis:
        """
        Create a default analysis instance with global downsample applied.
        """
        analysis_class = ANALYSIS_REGISTRY[name]
        
        # Apply global downsample if not 1.0
        if self.global_downsample != 1.0:
            return analysis_class(downsample=self.global_downsample)
        else:
            return analysis_class()
    
    def _create_all_analyses(self) -> Dict[str, BaseAnalysis]:
        """
        Create instances of all available analyses.
        """
        result = {}
        for name in ANALYSIS_REGISTRY:
            result[name] = self._create_analysis(name)
        return result
    
    def list_analyses(self) -> List[str]:
        """
        Get list of available analysis names.
        """
        return list(ANALYSIS_REGISTRY.keys())
    
    def get_analysis_info(self, name: str) -> Dict[str, Any]:
        """
        Get information about a specific analysis.
        """
        if name not in ANALYSIS_REGISTRY:
            raise ValueError(f"Unknown analysis: {name}")
        
        analysis_class = ANALYSIS_REGISTRY[name]
        instance = analysis_class()
        
        return {
            'name': name,
            'class': analysis_class.__name__,
            'method': analysis_class.METHOD_NAME,
            'default_downsample': instance.downsample,
            'parameters': instance.config if hasattr(instance, 'config') else {}
        }


def analyze_video(video_path: Union[str, Path],
                 analyses: Optional[List[str]] = None,
                 **kwargs) -> RawAnalysisResults:
    """
    Convenience function to analyze a video.
    
    Args:
        video_path: Path to video file
        analyses: List of analysis names to run (None = all)
        **kwargs: Additional parameters for VideoKurt
    
    Returns:
        RawAnalysisResults
    
    Example:
        results = analyze_video("video.mp4", analyses=['frame_diff', 'optical_flow_dense'])
    """
    vk = VideoKurt(**kwargs)
    return vk.analyze(video_path, analyses=analyses)


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m videokurt.videokurt <video_path> [analysis1,analysis2,...]")
        print("\nAvailable analyses:")
        vk = VideoKurt()
        for name in vk.list_analyses():
            info = vk.get_analysis_info(name)
            print(f"  - {name}: {info['class']}")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    # Parse analyses if provided
    analyses = None
    if len(sys.argv) > 2:
        analyses = sys.argv[2].split(',')
    
    # Run analysis
    print(f"Analyzing {video_path}...")
    vk = VideoKurt(downsample=0.5, max_seconds=10)
    
    try:
        results = vk.analyze(video_path, analyses=analyses)
        
        print(f"\nAnalysis complete!")
        print(f"Video: {results.dimensions[0]}x{results.dimensions[1]} @ {results.fps:.1f}fps")
        print(f"Duration: {results.duration:.1f}s ({results.frame_count} frames processed)")
        print(f"Processing time: {results.elapsed_time:.1f}s")
        
        print(f"\nAnalyses completed:")
        for name, analysis in results.analyses.items():
            print(f"  - {name}:")
            print(f"    Data keys: {list(analysis.data.keys())}")
            print(f"    Processing time: {analysis.processing_time:.2f}s")
            
    except VideoLoadError as e:
        print(f"Error loading video: {e}")
        sys.exit(1)
    except AnalysisError as e:
        print(f"Analysis error: {e}")
        if e.partial_results:
            print(f"Partial results available: {list(e.partial_results.analyses.keys())}")
        sys.exit(1)