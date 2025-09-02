"""Shape drawing utilities for frame generation."""

from typing import Tuple
import numpy as np


def add_rectangle(
    frame: np.ndarray,
    pos: Tuple[int, int],
    size: Tuple[int, int],
    color: Tuple[int, int, int] = (255, 255, 255),
    filled: bool = True
) -> np.ndarray:
    """Add a rectangle to a frame.
    
    Args:
        frame: Input frame to modify
        pos: (x, y) position of top-left corner
        size: (width, height) of rectangle
        color: Color of the rectangle
        filled: Whether to fill the rectangle
        
    Returns:
        Modified frame
    """
    frame = frame.copy()
    x, y = pos
    w, h = size
    
    # Ensure bounds are within frame
    x_end = min(x + w, frame.shape[1])
    y_end = min(y + h, frame.shape[0])
    x = max(0, x)
    y = max(0, y)
    
    if filled:
        if len(frame.shape) == 2:  # Grayscale
            frame[y:y_end, x:x_end] = color[0] if isinstance(color, tuple) else color
        else:  # Color
            frame[y:y_end, x:x_end] = color
    else:
        # Draw outline only
        thickness = 1
        if len(frame.shape) == 2:
            val = color[0] if isinstance(color, tuple) else color
            frame[y:y+thickness, x:x_end] = val  # Top
            frame[y_end-thickness:y_end, x:x_end] = val  # Bottom
            frame[y:y_end, x:x+thickness] = val  # Left
            frame[y:y_end, x_end-thickness:x_end] = val  # Right
        else:
            frame[y:y+thickness, x:x_end] = color  # Top
            frame[y_end-thickness:y_end, x:x_end] = color  # Bottom
            frame[y:y_end, x:x+thickness] = color  # Left
            frame[y:y_end, x_end-thickness:x_end] = color  # Right
    
    return frame


def add_circle(
    frame: np.ndarray,
    center: Tuple[int, int],
    radius: int,
    color: Tuple[int, int, int] = (255, 255, 255),
    filled: bool = True
) -> np.ndarray:
    """Add a circle to a frame.
    
    Args:
        frame: Input frame to modify
        center: (x, y) center position
        radius: Circle radius
        color: Color of the circle
        filled: Whether to fill the circle
        
    Returns:
        Modified frame
    """
    frame = frame.copy()
    cx, cy = center
    
    # Create circle mask
    y, x = np.ogrid[:frame.shape[0], :frame.shape[1]]
    mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
    
    if not filled:
        # Create ring mask for outline
        inner_mask = (x - cx) ** 2 + (y - cy) ** 2 <= (radius - 1) ** 2
        mask = mask & ~inner_mask
    
    if len(frame.shape) == 2:  # Grayscale
        frame[mask] = color[0] if isinstance(color, tuple) else color
    else:  # Color
        frame[mask] = color
    
    return frame


def add_text_region(
    frame: np.ndarray,
    pos: Tuple[int, int],
    size: Tuple[int, int],
    text_color: int = 200,
    bg_color: int = 50
) -> np.ndarray:
    """Add a simulated text region to frame.
    
    Args:
        frame: Input frame to modify
        pos: (x, y) position of text region
        size: (width, height) of text region
        text_color: Intensity of text lines
        bg_color: Background color of text region
        
    Returns:
        Modified frame with simulated text
    """
    frame = frame.copy()
    x, y = pos
    w, h = size
    
    # Create text region background
    frame = add_rectangle(frame, pos, size, (bg_color, bg_color, bg_color), filled=True)
    
    # Add simulated text lines
    line_height = 2
    line_spacing = 1
    current_y = y + 1
    
    while current_y + line_height < y + h:
        # Random line length to simulate text
        line_width = np.random.randint(w // 2, w - 2)
        if len(frame.shape) == 2:
            frame[current_y:current_y+line_height, x+1:x+1+line_width] = text_color
        else:
            frame[current_y:current_y+line_height, x+1:x+1+line_width] = (text_color, text_color, text_color)
        current_y += line_height + line_spacing
    
    return frame