"""Video metadata and information."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class VideoInfo:
    """Video file metadata."""
    
    filepath: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float
    codec: Optional[str] = None
    
    @property
    def resolution(self) -> str:
        """Get resolution as string."""
        return f"{self.width}x{self.height}"
    
    @property
    def aspect_ratio(self) -> float:
        """Calculate aspect ratio."""
        return self.width / self.height if self.height > 0 else 0
    
    def __str__(self) -> str:
        """Human-readable video info."""
        return (
            f"Video: {self.filepath}\n"
            f"  Resolution: {self.resolution} ({self.aspect_ratio:.2f}:1)\n"
            f"  FPS: {self.fps:.2f}\n"
            f"  Duration: {self.duration_seconds:.2f}s\n"
            f"  Total frames: {self.frame_count}"
        )