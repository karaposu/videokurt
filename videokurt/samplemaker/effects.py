"""Visual effects and artifacts for frame generation."""

import numpy as np


def add_noise(
    frame: np.ndarray,
    noise_type: str = 'gaussian',
    intensity: float = 0.1
) -> np.ndarray:
    """Add noise to a frame.
    
    Args:
        frame: Input frame
        noise_type: Type of noise ('gaussian', 'salt_pepper', 'uniform')
        intensity: Noise intensity (0.0-1.0)
        
    Returns:
        Noisy frame
    """
    frame = frame.copy().astype(np.float32)
    
    if noise_type == 'gaussian':
        noise = np.random.randn(*frame.shape) * intensity * 255
        frame = frame + noise
    elif noise_type == 'salt_pepper':
        mask = np.random.random(frame.shape[:2]) < intensity
        if len(frame.shape) == 2:
            frame[mask] = np.random.choice([0, 255], size=np.sum(mask))
        else:
            for c in range(frame.shape[2]):
                frame[mask, c] = np.random.choice([0, 255], size=np.sum(mask))
    elif noise_type == 'uniform':
        noise = (np.random.random(frame.shape) - 0.5) * 2 * intensity * 255
        frame = frame + noise
    
    return np.clip(frame, 0, 255).astype(np.uint8)


def add_compression_artifacts(
    frame: np.ndarray,
    block_size: int = 4,
    quality: float = 0.7
) -> np.ndarray:
    """Simulate compression artifacts.
    
    Args:
        frame: Input frame
        block_size: Size of compression blocks
        quality: Quality factor (lower = more artifacts)
        
    Returns:
        Frame with compression artifacts
    """
    frame = frame.copy()
    height, width = frame.shape[:2]
    
    # Process in blocks
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            block_h = min(block_size, height - i)
            block_w = min(block_size, width - j)
            
            if len(frame.shape) == 2:
                block = frame[i:i+block_h, j:j+block_w]
                # Reduce precision to simulate compression
                avg = np.mean(block)
                quantized = np.round(avg / (256 * (1 - quality))) * (256 * (1 - quality))
                frame[i:i+block_h, j:j+block_w] = quantized
            else:
                for c in range(frame.shape[2]):
                    block = frame[i:i+block_h, j:j+block_w, c]
                    avg = np.mean(block)
                    quantized = np.round(avg / (256 * (1 - quality))) * (256 * (1 - quality))
                    frame[i:i+block_h, j:j+block_w, c] = quantized
    
    return frame.astype(np.uint8)