"""Basic frame generation utilities."""

from typing import Tuple
import numpy as np


def create_blank_frame(
    size: Tuple[int, int] = (20, 20),
    channels: int = 3,
    dtype: np.dtype = np.uint8
) -> np.ndarray:
    """Create a blank (black) frame.
    
    Args:
        size: (height, width) of the frame
        channels: Number of color channels (1 for grayscale, 3 for BGR)
        dtype: Data type for the frame
        
    Returns:
        Blank frame array
    """
    if channels == 1:
        return np.zeros(size, dtype=dtype)
    else:
        return np.zeros((*size, channels), dtype=dtype)


def create_solid_frame(
    size: Tuple[int, int] = (20, 20),
    color: Tuple[int, int, int] = (128, 128, 128),
    channels: int = 3
) -> np.ndarray:
    """Create a solid color frame.
    
    Args:
        size: (height, width) of the frame
        color: BGR color tuple or grayscale value
        channels: Number of color channels
        
    Returns:
        Solid color frame
    """
    frame = create_blank_frame(size, channels)
    
    if channels == 1:
        frame[:, :] = color[0] if isinstance(color, tuple) else color
    else:
        frame[:, :] = color
    
    return frame


def create_gradient_frame(
    size: Tuple[int, int] = (20, 20),
    direction: str = 'horizontal',
    start_val: int = 0,
    end_val: int = 255
) -> np.ndarray:
    """Create a gradient frame.
    
    Args:
        size: (height, width) of the frame
        direction: 'horizontal', 'vertical', or 'diagonal'
        start_val: Starting intensity value
        end_val: Ending intensity value
        
    Returns:
        Gradient frame (grayscale)
    """
    height, width = size
    frame = np.zeros(size, dtype=np.uint8)
    
    if direction == 'horizontal':
        gradient = np.linspace(start_val, end_val, width, dtype=np.uint8)
        frame[:, :] = gradient
    elif direction == 'vertical':
        gradient = np.linspace(start_val, end_val, height, dtype=np.uint8)
        frame[:, :] = gradient.reshape(-1, 1)
    elif direction == 'diagonal':
        x_grad = np.linspace(0, 1, width)
        y_grad = np.linspace(0, 1, height).reshape(-1, 1)
        combined = (x_grad + y_grad) / 2
        frame = (start_val + (end_val - start_val) * combined).astype(np.uint8)
    
    return frame


def create_checkerboard(
    size: Tuple[int, int] = (20, 20),
    square_size: int = 5,
    color1: int = 0,
    color2: int = 255
) -> np.ndarray:
    """Create a checkerboard pattern.
    
    Args:
        size: (height, width) of the frame
        square_size: Size of each square in pixels
        color1: First color intensity
        color2: Second color intensity
        
    Returns:
        Checkerboard pattern frame (grayscale)
    """
    height, width = size
    frame = np.zeros(size, dtype=np.uint8)
    
    for i in range(0, height, square_size):
        for j in range(0, width, square_size):
            if ((i // square_size) + (j // square_size)) % 2 == 0:
                frame[i:i+square_size, j:j+square_size] = color1
            else:
                frame[i:i+square_size, j:j+square_size] = color2
    
    return frame