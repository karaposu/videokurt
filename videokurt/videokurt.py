"""
VideoKurt - New Interface Implementation
Builder pattern for configuring and running video analyses.
"""

import time
import cv2
import numpy as np
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
from dataclasses import dataclass

from videokurt.models import RawAnalysisResults, RawAnalysis
from videokurt.raw_analysis import (
    ANALYSIS_REGISTRY,
    BaseAnalysis,
)
from videokurt.features import (
    BASIC_FEATURES,
    MIDDLE_FEATURES,
    ADVANCED_FEATURES,
)


class VideoLoadError(Exception):
    """Raised when video cannot be loaded."""
    pass


class ConfigurationError(Exception):
    """Raised when configuration is invalid."""
    pass


class AnalysisError(Exception):
    """Raised when analysis fails."""
    def __init__(self, message, partial_results=None):
        super().__init__(message)
        self.partial_results = partial_results


class VideoKurt:
    """
    Main interface for video analysis using builder pattern.
    
    Usage:
        vk = VideoKurt()
        vk.add_analysis('frame_diff')
        vk.add_feature('binary_activity')
        vk.configure(frame_step=2, resolution_scale=0.5)
        results = vk.analyze('video.mp4')
    """
    
    def __init__(self):
        """Initialize empty VideoKurt instance."""
        self._analyses = {}  # name -> BaseAnalysis instance
        self._features = {}  # name -> BaseFeature class
        self._config = {
            'frame_step': 1,
            'resolution_scale': 1.0,
            'blur': False,
            'blur_kernel_size': 13,
            'process_chunks': 1,  # 1 = no chunking
            'chunk_overlap': 30   # frames overlap between chunks
        }
        self._mode = 'full'  # 'full', 'features_only', 
        
    def add_analysis(self, 
                     analysis: Union[str, BaseAnalysis],
                     **kwargs):
        """
        Add analysis by name with kwargs or pre-configured object.
        
        Args:
            analysis: Either analysis name string or configured BaseAnalysis instance
            **kwargs: Parameters for analysis if using string name
            
        Examples:
            vk.add_analysis('frame_diff')
            vk.add_analysis('frame_diff', threshold=0.3)
            vk.add_analysis(FrameDiff(threshold=0.3))
        """
        if isinstance(analysis, str):
            if analysis not in ANALYSIS_REGISTRY:
                raise ValueError(f"Unknown analysis: {analysis}")
            analysis_class = ANALYSIS_REGISTRY[analysis]
            self._analyses[analysis] = analysis_class(**kwargs)
        elif isinstance(analysis, BaseAnalysis):
            self._analyses[analysis.METHOD_NAME] = analysis
        else:
            raise TypeError(f"analysis must be string or BaseAnalysis, got {type(analysis)}")
    
    def add_feature(self, feature: str, **kwargs):
        """
        Add feature and auto-include required analyses.
        
        Args:
            feature: Name of feature to compute
            **kwargs: Feature-specific parameters
            
        Note:
            This will automatically add any required analyses
            with default configurations if not already present.
        """
        # Check all feature registries
        if feature in BASIC_FEATURES:
            feature_class = BASIC_FEATURES[feature]
        elif feature in MIDDLE_FEATURES:
            feature_class = MIDDLE_FEATURES[feature]
        elif feature in ADVANCED_FEATURES:
            feature_class = ADVANCED_FEATURES[feature]
        else:
            raise ValueError(f"Unknown feature: {feature}")
        
        # Create feature instance with provided parameters
        self._features[feature] = feature_class(**kwargs)
        
        # Auto-add required analyses with defaults
        for req in feature_class.REQUIRED_ANALYSES:
            if req not in self._analyses:
                self.add_analysis(req)
    
    def configure(self, **kwargs):
        """
        Configure global preprocessing options.
        
        Args:
            frame_step: Process every Nth frame (1 = all frames)
            resolution_scale: Scale factor for frame resolution (0.5 = half size)
            blur: Whether to apply Gaussian blur
            blur_kernel_size: Kernel size for blur (must be odd)
            process_chunks: Number of chunks to divide video into
            chunk_overlap: Number of frames to overlap between chunks
        """
        # Validate configuration
        if 'frame_step' in kwargs:
            if kwargs['frame_step'] < 1:
                raise ValueError("frame_step must be >= 1")
                
        if 'resolution_scale' in kwargs:
            if not 0 < kwargs['resolution_scale'] <= 1:
                raise ValueError("resolution_scale must be between 0 and 1")
                
        if 'blur_kernel_size' in kwargs:
            if kwargs['blur_kernel_size'] % 2 == 0:
                raise ValueError("blur_kernel_size must be odd")
                
        if 'process_chunks' in kwargs:
            if kwargs['process_chunks'] < 1:
                raise ValueError("process_chunks must be >= 1")
        
        self._config.update(kwargs)
    
    def set_mode(self, mode: str):
        """
        Set execution mode.
        
        Args:
            mode: One of 'full', 'features_only', 'streaming'
                  (only 'full' is currently implemented)
        """
        if mode not in ['full', 'features_only', 'streaming']:
            raise ValueError(f"Invalid mode: {mode}")
        if mode in ['features_only', 'streaming']:
            raise NotImplementedError(f"Mode '{mode}' not yet implemented")
        self._mode = mode
    
    def analyze(self, 
                video_path: Union[str, Path],
                ) -> RawAnalysisResults:
        """
        Run configured analyses on video.
        
        Args:
            video_path: Path to video file
            
        Returns:
            RawAnalysisResults containing all analysis outputs
            
        Raises:
            ConfigurationError: If no analyses configured
            VideoLoadError: If video cannot be loaded
            AnalysisError: If analysis fails
        """
        # Validate we have something to do
        if not self._analyses and not self._features:
            raise ConfigurationError("No analyses or features configured")
        
        start_time = time.time()
        video_path = Path(video_path)
        
        # Load and preprocess video
        if self._config['process_chunks'] > 1:
            # Process in chunks
            results = self._analyze_chunked(video_path)
        else:
            # Process entire video at once
            frames, metadata = self._load_video(video_path)
            frames = self._preprocess_frames(frames)
            results = self._run_analyses(frames, metadata, video_path, start_time)
        
        # Compute features if configured
        if self._features:
            self._compute_features(results)
        
        return results
    
    def _load_video(self, video_path: Path) -> tuple[List[np.ndarray], Dict]:
        """
        Load video frames and metadata.
        
        Returns:
            Tuple of (frames list, metadata dict)
        """
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
        
        # Apply frame_step to determine which frames to load
        frames = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % self._config['frame_step'] == 0:
                frames.append(frame)
            
            frame_idx += 1
        
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
    
    def _preprocess_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Apply global preprocessing to frames.
        
        Args:
            frames: List of video frames
            
        Returns:
            List of preprocessed frames
        """
        processed = []
        
        for frame in frames:
            # Apply resolution scaling
            if self._config['resolution_scale'] < 1.0:
                h, w = frame.shape[:2]
                new_h = int(h * self._config['resolution_scale'])
                new_w = int(w * self._config['resolution_scale'])
                frame = cv2.resize(frame, (new_w, new_h))
            
            # Apply blur if configured
            if self._config['blur']:
                kernel_size = self._config['blur_kernel_size']
                frame = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
            
            processed.append(frame)
        
        return processed
    
    def _run_analyses(self, 
                      frames: List[np.ndarray],
                      metadata: Dict,
                      video_path: Path,
                      start_time: float) -> RawAnalysisResults:
        """
        Run all configured analyses on frames.
        
        Returns:
            RawAnalysisResults with all analysis outputs
        """
        analysis_results = {}
        
        for name, analyzer in self._analyses.items():
            try:
                print(f"Running {name}...")
                analysis_start = time.time()
                
                # Run analysis
                result = analyzer.analyze(frames)
                
                # Store result
                analysis_results[name] = result
                
                print(f"  Completed in {time.time() - analysis_start:.2f}s")
                
            except Exception as e:
                print(f"  Warning: {name} failed: {e}")
                # Continue with other analyses
        
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
    
    def _analyze_chunked(self, video_path: Path) -> RawAnalysisResults:
        """
        Process video in chunks to reduce memory usage.
        
        This is a simple implementation that processes chunks
        sequentially and merges results at the end.
        """
        # TODO: Implement chunked processing
        # For now, fall back to regular processing
        print(f"Note: Chunked processing not yet implemented, using regular processing")
        frames, metadata = self._load_video(video_path)
        frames = self._preprocess_frames(frames)
        return self._run_analyses(frames, metadata, video_path, time.time())
    
    def _compute_features(self, results: RawAnalysisResults):
        """
        Compute configured features from analysis results.
        
        Args:
            results: RawAnalysisResults to add features to
        """
        if not self._features:
            return
        
        # Build dict of analysis results for feature computation
        analysis_data = {
            name: result for name, result in results.analyses.items()
        }
        
        # Store computed features
        computed_features = {}
        
        for feature_name, feature_instance in self._features.items():
            try:
                print(f"Computing feature: {feature_name}...")
                start_time = time.time()
                
                # Check if required analyses are available
                missing_analyses = []
                for required in feature_instance.REQUIRED_ANALYSES:
                    if required not in analysis_data:
                        missing_analyses.append(required)
                
                if missing_analyses:
                    print(f"  Warning: Skipping {feature_name} - missing analyses: {missing_analyses}")
                    continue
                
                # Compute the feature
                feature_data = feature_instance.compute(analysis_data)
                
                # Create FeatureResult
                from .models import FeatureResult
                feature_result = FeatureResult(
                    name=feature_name,
                    data=feature_data,
                    metadata={'config': feature_instance.config if hasattr(feature_instance, 'config') else {}},
                    dtype=str(feature_data.dtype) if hasattr(feature_data, 'dtype') else type(feature_data).__name__,
                    shape=feature_data.shape if hasattr(feature_data, 'shape') else None,
                    compute_time=time.time() - start_time,
                    required_analyses=feature_instance.REQUIRED_ANALYSES
                )
                
                computed_features[feature_name] = feature_result
                print(f"  Completed in {feature_result.compute_time:.2f}s")
                
            except Exception as e:
                print(f"  Error computing {feature_name}: {e}")
                continue
        
        # Add features to results
        results.features = computed_features
    
    def list_analyses(self) -> List[str]:
        """List configured analyses."""
        return list(self._analyses.keys())
    
    def list_features(self) -> List[str]:
        """List configured features."""
        return list(self._features.keys())
    
    def list_available_analyses(self) -> List[str]:
        """List all available analysis types."""
        return list(ANALYSIS_REGISTRY.keys())
    
    def list_available_features(self) -> List[str]:
        """List all available feature types."""
        # TODO: Combine all feature registries
        return list(BASIC_FEATURES.keys())
    
    def validate(self) -> List[str]:
        """
        Validate current configuration.
        
        Returns:
            List of validation issues (empty if valid)
        """
        issues = []
        
        # Check if any analyses configured
        if not self._analyses and not self._features:
            issues.append("No analyses or features configured")
        
        # Check for conflicting configurations
        # TODO: Add more validation logic
        
        return issues
    
    def clear(self):
        """Clear all configurations."""
        self._analyses.clear()
        self._features.clear()
        self._config = {
            'frame_step': 1,
            'resolution_scale': 1.0,
            'blur': False,
            'blur_kernel_size': 13,
            'process_chunks': 1,
            'chunk_overlap': 30
        }
        self._mode = 'full'
    
    def __repr__(self):
        """String representation of VideoKurt configuration."""
        return (
            f"VideoKurt(\n"
            f"  analyses={list(self._analyses.keys())},\n"
            f"  features={list(self._features.keys())},\n"
            f"  config={self._config},\n"
            f"  mode='{self._mode}'\n"
            f")"
        )
    
    @staticmethod
    def save_video(frames: List[np.ndarray], 
                   output_path: Union[str, Path],
                   fps: float = 30.0,
                   codec: str = 'mp4v') -> bool:
        """
        Save frames as video file for debugging.
        
        Args:
            frames: List of numpy arrays (frames to save)
            output_path: Path where to save the video
            fps: Frames per second for output video
            codec: Four-character code for video codec ('mp4v', 'XVID', 'MJPG', etc.)
        
        Returns:
            True if successful, False otherwise
            
        Example:
            VideoKurt.save_video(processed_frames, 'debug_output.mp4')
            
        Note:
            - Handles both grayscale and RGB frames
            - Automatically converts grayscale to BGR for video encoding
        """
        if not frames:
            print("Error: No frames to save")
            return False
        
        output_path = Path(output_path)
        
        # Get frame properties from first frame
        first_frame = frames[0]
        height, width = first_frame.shape[:2]
        is_color = len(first_frame.shape) == 3
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height), isColor=True)
        
        if not writer.isOpened():
            print(f"Error: Could not open video writer with codec '{codec}'")
            # Try alternative codec
            if codec == 'mp4v':
                print("Trying XVID codec instead...")
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height), isColor=True)
            
            if not writer.isOpened():
                print("Error: Could not open video writer with any codec")
                return False
        
        # Write frames
        try:
            for i, frame in enumerate(frames):
                # Ensure frame is in BGR format (required for video writing)
                if len(frame.shape) == 2:  # Grayscale
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                elif frame.shape[2] == 4:  # BGRA
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                
                # Ensure frame dimensions match
                if frame.shape[:2] != (height, width):
                    print(f"Warning: Frame {i} has different dimensions, resizing...")
                    frame = cv2.resize(frame, (width, height))
                
                writer.write(frame)
            
            writer.release()
            print(f"✓ Saved {len(frames)} frames to {output_path}")
            return True
            
        except Exception as e:
            print(f"Error writing video: {e}")
            writer.release()
            return False


# Convenience function for quick analysis
def analyze_video(video_path: Union[str, Path],
                  analyses: List[str] = None,
                  features: List[str] = None,
                  **config) -> RawAnalysisResults:
    """
    Convenience function for quick video analysis.
    
    Args:
        video_path: Path to video file
        analyses: List of analysis names to run
        features: List of feature names to compute
        **config: Configuration parameters
        
    Returns:
        RawAnalysisResults
        
    Example:
        results = analyze_video('video.mp4', 
                               analyses=['frame_diff', 'optical_flow_dense'],
                               frame_step=2, resolution_scale=0.5)
    """
    vk = VideoKurt()
    
    if analyses:
        for analysis in analyses:
            vk.add_analysis(analysis)
    
    if features:
        for feature in features:
            vk.add_feature(feature)
    
    if config:
        vk.configure(**config)
    
    return vk.analyze(video_path)


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python videokurt_new.py <video_path> [analysis1,analysis2,...]")
        print("\nAvailable analyses:")
        vk = VideoKurt()
        for name in vk.list_available_analyses():
            print(f"  - {name}")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    # Parse analyses if provided
    analyses = None
    if len(sys.argv) > 2:
        analyses = sys.argv[2].split(',')
    
    # Create VideoKurt instance
    vk = VideoKurt()
    
    # Add analyses
    if analyses:
        for analysis in analyses:
            vk.add_analysis(analysis)
    else:
        # Default to frame_diff
        vk.add_analysis('frame_diff')
    
    # Configure for faster processing
    vk.configure(
        frame_step=5,
        resolution_scale=0.5
    )
    
    # Run analysis
    print(f"\nAnalyzing {video_path}...")
    print(vk)
    
    try:
        results = vk.analyze(video_path)
        
        print(f"\n{'='*50}")
        print("Analysis complete!")
        print(f"{'='*50}")
        results.print_summary()
        
    except (VideoLoadError, ConfigurationError, AnalysisError) as e:
        print(f"Error: {e}")
        sys.exit(1)