"""Motion simulation utilities for frame generation."""

from typing import Tuple, Optional, List
import numpy as np
from .shapes import add_rectangle
from .frames import create_checkerboard, create_solid_frame


def simulate_scroll(
    frame1: np.ndarray,
    pixels: int = 5,
    direction: str = 'down'
) -> np.ndarray:
    """Simulate scrolling motion between frames.
    
    Args:
        frame1: Original frame
        pixels: Number of pixels to scroll
        direction: Scroll direction ('up', 'down', 'left', 'right')
        
    Returns:
        Scrolled frame
    """
    frame2 = np.zeros_like(frame1)
    
    if direction == 'down':
        frame2[pixels:, :] = frame1[:-pixels, :]
        frame2[:pixels, :] = frame1[-pixels:, :]  # Wrap around
    elif direction == 'up':
        frame2[:-pixels, :] = frame1[pixels:, :]
        frame2[-pixels:, :] = frame1[:pixels, :]
    elif direction == 'right':
        frame2[:, pixels:] = frame1[:, :-pixels]
        frame2[:, :pixels] = frame1[:, -pixels:]
    elif direction == 'left':
        frame2[:, :-pixels] = frame1[:, pixels:]
        frame2[:, -pixels:] = frame1[:, :pixels]
    
    return frame2


def simulate_scene_change(
    frame1: np.ndarray,
    change_type: str = 'cut'
) -> np.ndarray:
    """Simulate a scene change.
    
    Args:
        frame1: Original frame
        change_type: Type of change ('cut', 'fade', 'slide')
        
    Returns:
        Changed frame
    """
    if change_type == 'cut':
        # Complete change - generate different pattern
        if len(frame1.shape) == 2:
            frame2 = create_checkerboard(frame1.shape)
        else:
            frame2 = create_solid_frame(frame1.shape[:2], color=(100, 150, 200))
    elif change_type == 'fade':
        # Fade to different color
        frame2 = (frame1 * 0.3).astype(np.uint8)
        # Add some variation
        if len(frame2.shape) == 2:
            frame2 = frame2 + np.random.randint(0, 50, frame2.shape, dtype=np.uint8)
        else:
            for c in range(frame2.shape[2]):
                frame2[:, :, c] = frame2[:, :, c] + np.random.randint(0, 30, frame2.shape[:2], dtype=np.uint8)
        frame2 = np.clip(frame2, 0, 255).astype(np.uint8)
    elif change_type == 'slide':
        # Slide transition
        frame2 = frame1.copy()
        split = frame1.shape[1] // 2
        frame2[:, :split] = 128  # Different content on left half
    
    return frame2


def simulate_popup(
    frame: np.ndarray,
    popup_size: Tuple[int, int] = (10, 8),
    position: Optional[Tuple[int, int]] = None,
    bg_color: int = 200,
    border_color: int = 100
) -> np.ndarray:
    """Add a popup/modal to frame.
    
    Args:
        frame: Base frame
        popup_size: (width, height) of popup
        position: (x, y) position, or None for center
        bg_color: Background color of popup
        border_color: Border color
        
    Returns:
        Frame with popup
    """
    frame = frame.copy()
    h, w = frame.shape[:2]
    pw, ph = popup_size
    
    if position is None:
        # Center the popup
        x = (w - pw) // 2
        y = (h - ph) // 2
    else:
        x, y = position
    
    # Add semi-transparent overlay (darken background)
    frame = (frame * 0.7).astype(np.uint8)
    
    # Add popup background
    frame = add_rectangle(frame, (x, y), (pw, ph), (bg_color, bg_color, bg_color), filled=True)
    
    # Add border
    frame = add_rectangle(frame, (x, y), (pw, ph), (border_color, border_color, border_color), filled=False)
    
    return frame


def simulate_video_playback(
    base_frame: np.ndarray,
    num_frames: int = 30,
    video_region: Optional[Tuple[int, int, int, int]] = None,
    fps_variation: float = 0.8
) -> List[np.ndarray]:
    """Simulate video playback in a region of the screen.
    
    Creates frames that mimic video playing in a bounded region while
    the rest of the screen remains static (like Instagram feed with autoplay video).
    
    Args:
        base_frame: Background frame
        num_frames: Number of frames to generate
        video_region: (x, y, width, height) or None for auto
        fps_variation: Simulate video at different frame rate (0.8 = 24fps in 30fps recording)
        
    Returns:
        List of frames with simulated video playback
    """
    frames = []
    h, w = base_frame.shape[:2]
    
    # Default video region (like Instagram post in feed)
    if video_region is None:
        # Center region, typical video aspect ratio
        vw = min(w - 20, 320)  # Video width
        vh = min(h - 20, 180)  # 16:9 aspect ratio
        vx = (w - vw) // 2
        vy = (h - vh) // 2
        video_region = (vx, vy, vw, vh)
    
    vx, vy, vw, vh = video_region
    
    for i in range(num_frames):
        frame = base_frame.copy()
        
        # Simulate video content changing at video framerate
        # This creates the characteristic pattern of video playback
        video_frame_num = int(i * fps_variation)
        
        # Generate video-like content (moving gradients to simulate motion)
        video_content = np.zeros((vh, vw), dtype=np.uint8)
        
        # Create moving pattern that changes each frame
        for y in range(vh):
            for x in range(vw):
                # Diagonal moving gradient (simulates video motion)
                value = int(128 + 100 * np.sin((x + y + video_frame_num * 15) * 0.05))
                video_content[y, x] = value
        
        # Add some temporal consistency (videos have smooth motion)
        if i > 0 and i % 3 != 0:  # Not a keyframe
            # Blend with previous for temporal smoothness
            video_content = (video_content * 0.8).astype(np.uint8)
        
        # Add slight noise to simulate compression
        noise = np.random.randint(-5, 5, (vh, vw))
        video_content = np.clip(video_content + noise, 0, 255).astype(np.uint8)
        
        # Place video in the frame
        if len(frame.shape) == 3:
            # Color frame - convert to color
            for c in range(frame.shape[2]):
                frame[vy:vy+vh, vx:vx+vw, c] = video_content
        else:
            # Grayscale
            frame[vy:vy+vh, vx:vx+vw] = video_content
        
        frames.append(frame)
    
    return frames