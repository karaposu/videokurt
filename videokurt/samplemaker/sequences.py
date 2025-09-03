"""Frame sequence and video generation utilities."""

from typing import Tuple, List, Optional, Dict, Any
import numpy as np
from .frames import create_gradient_frame, create_checkerboard, create_solid_frame
from .shapes import add_circle, add_text_region
from .motion import simulate_scroll, simulate_popup
from .effects import add_noise


def create_frames_with_pattern(
    num_frames: int = 10,
    width: int = 100,
    height: int = 100,
    pattern: str = 'moving_line'
) -> List[np.ndarray]:
    """Create frames with specific test patterns.
    
    Args:
        num_frames: Number of frames to generate
        width: Frame width in pixels
        height: Frame height in pixels  
        pattern: Type of pattern ('moving_line', 'moving_circle', 'gradient', 'checkerboard')
        
    Returns:
        List of frames with the specified pattern
    """
    import cv2
    frames = []
    
    if pattern == 'moving_line':
        # Create frames with moving diagonal line (good for motion tests)
        for i in range(num_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            # Draw diagonal line that moves
            offset = int((i / num_frames) * width)
            cv2.line(frame, (offset, 0), (min(offset + 20, width-1), min(20, height-1)), 
                    (255, 255, 255), 2)
            # Add frame number for debugging
            cv2.putText(frame, str(i), (10, height-10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (255, 255, 255), 1)
            frames.append(frame)
            
    elif pattern == 'moving_circle':
        # Create frames with moving circle
        for i in range(num_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            x = int((i / num_frames) * width)
            y = height // 2
            cv2.circle(frame, (x, y), 10, (255, 255, 255), -1)
            frames.append(frame)
            
    elif pattern == 'gradient':
        # Create gradient frames (good for edge detection tests)
        for i in range(num_frames):
            frame = create_gradient_frame((height, width))
            # Convert to 3-channel if needed
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            frames.append(frame)
            
    elif pattern == 'checkerboard':
        # Create checkerboard frames (good for blur tests)
        for i in range(num_frames):
            frame = create_checkerboard((height, width), square_size=10)
            # Convert to 3-channel if needed
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            # Add some variation between frames
            if i % 2 == 1:
                frame = cv2.bitwise_not(frame)
            frames.append(frame)
    else:
        # Default: solid color frames with text
        for i in range(num_frames):
            frame = create_solid_frame((height, width), color=50)
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            cv2.putText(frame, f"Frame {i}", (width//4, height//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            frames.append(frame)
    
    return frames


def create_frame_sequence(
    num_frames: int = 10,
    size: Tuple[int, int] = (20, 20),
    scenario: str = 'activity'
) -> List[np.ndarray]:
    """Create a sequence of frames for testing.
    
    Args:
        num_frames: Number of frames to generate
        size: Size of each frame
        scenario: Type of sequence ('activity', 'idle', 'mixed')
        
    Returns:
        List of frames
    """
    frames = []
    base_frame = create_gradient_frame(size)
    
    if scenario == 'activity':
        # Continuous activity
        for i in range(num_frames):
            frame = base_frame.copy()
            # Add moving object
            x = (i * 2) % size[1]
            frame = add_circle(frame, (x, size[0]//2), 3, color=200)
            frames.append(frame)
            
    elif scenario == 'idle':
        # No changes
        for i in range(num_frames):
            frames.append(base_frame.copy())
            
    elif scenario == 'mixed':
        # Mix of activity and idle
        for i in range(num_frames):
            if i < 3 or i > 7:  # Active periods
                frame = base_frame.copy()
                x = (i * 2) % size[1]
                frame = add_circle(frame, (x, size[0]//2), 3, color=200)
            else:  # Idle period
                frame = base_frame.copy()
            frames.append(frame)
    
    return frames


def create_test_video_frames(
    size: Tuple[int, int] = (50, 50),
    events: Optional[Dict[str, Any]] = None,
    fps: float = 10.0
) -> Dict[str, Any]:
    """Create a comprehensive test video with various events and ground truth.
    
    Args:
        size: Size of frames
        events: Dictionary specifying which events to include
        fps: Frames per second for timing calculations
        
    Returns:
        Dictionary with:
            - frames: List of frame arrays
            - events: List of event dictionaries with timing
            - ground_truth: Frame-by-frame annotations
            - activity_timeline: Binary activity periods
            - fps: Frames per second
            - duration: Total duration in seconds
    """
    default_events = {
        'scene_changes': True,
        'scrolls': True,
        'popups': True,
        'idle_periods': True,
        'noise': False  # Default to false for cleaner ground truth
    }
    
    if events:
        default_events.update(events)
    
    frames = []
    event_list = []
    frame_annotations = []  # Frame-by-frame ground truth
    activity_timeline = []  # Binary activity periods
    
    # Helper to add frames with annotations
    def add_frames(frame_data, count, event_type=None, active=True):
        nonlocal frames, frame_annotations
        start_frame = len(frames)
        
        for i in range(count):
            if callable(frame_data):
                frame = frame_data(i)
            else:
                frame = frame_data.copy()
            frames.append(frame)
            
            # Add frame annotation
            frame_annotations.append({
                'frame_idx': len(frames) - 1,
                'timestamp': (len(frames) - 1) / fps,
                'event_type': event_type,
                'active': active
            })
        
        return start_frame, len(frames) - 1
    
    # Initial idle frames
    base = create_gradient_frame(size)
    start_idx, end_idx = add_frames(base, 5, event_type='idle', active=False)
    activity_timeline.append({
        'active': False,
        'start': start_idx / fps,
        'end': (end_idx + 1) / fps,
        'start_frame': start_idx,
        'end_frame': end_idx
    })
    
    # Scene change
    if default_events['scene_changes']:
        # Single frame of scene change
        new_scene = create_checkerboard(size, square_size=5)
        start_idx, _ = add_frames(new_scene, 1, event_type='scene_change', active=True)
        
        # Event timing
        event_list.append({
            'type': 'scene_change',
            'start': start_idx / fps,
            'end': (start_idx + 1) / fps,
            'start_frame': start_idx,
            'end_frame': start_idx,
            'confidence': 1.0  # Ground truth confidence
        })
        
        # Continue with new scene (active period)
        start_idx, end_idx = add_frames(new_scene, 5, event_type='post_scene_change', active=True)
        activity_timeline.append({
            'active': True,
            'start': (start_idx - 1) / fps,  # Include scene change frame
            'end': (end_idx + 1) / fps,
            'start_frame': start_idx - 1,
            'end_frame': end_idx
        })
    
    # Scroll sequence
    if default_events['scrolls']:
        scroll_base = create_gradient_frame(size, direction='vertical')
        scroll_base = add_text_region(scroll_base, (5, 5), (40, 40))
        
        # Generate scrolling frames
        def scroll_frame_gen(i):
            return simulate_scroll(scroll_base, pixels=i+1, direction='down')
        
        start_idx, end_idx = add_frames(scroll_frame_gen, 10, event_type='scroll', active=True)
        
        event_list.append({
            'type': 'scroll',
            'start': start_idx / fps,
            'end': (end_idx + 1) / fps,
            'start_frame': start_idx,
            'end_frame': end_idx,
            'confidence': 1.0,
            'metadata': {
                'direction': 'down',
                'total_pixels': 10,
                'velocity': 10 * fps  # pixels per second
            }
        })
        
        activity_timeline.append({
            'active': True,
            'start': start_idx / fps,
            'end': (end_idx + 1) / fps,
            'start_frame': start_idx,
            'end_frame': end_idx
        })
    
    # Idle period
    if default_events['idle_periods']:
        idle_frame = create_solid_frame(size, color=(128, 128, 128))
        start_idx, end_idx = add_frames(idle_frame, 15, event_type='idle', active=False)
        
        event_list.append({
            'type': 'idle_wait',
            'start': start_idx / fps,
            'end': (end_idx + 1) / fps,
            'start_frame': start_idx,
            'end_frame': end_idx,
            'confidence': 1.0
        })
        
        activity_timeline.append({
            'active': False,
            'start': start_idx / fps,
            'end': (end_idx + 1) / fps,
            'start_frame': start_idx,
            'end_frame': end_idx
        })
    
    # Popup
    if default_events['popups']:
        base_for_popup = create_gradient_frame(size)
        
        # Frame before popup (inactive)
        pre_start, pre_end = add_frames(base_for_popup, 1, event_type='pre_popup', active=False)
        
        # Add inactive period for pre-popup frame
        activity_timeline.append({
            'active': False,
            'start': pre_start / fps,
            'end': (pre_end + 1) / fps,
            'start_frame': pre_start,
            'end_frame': pre_end
        })
        
        # Popup appears
        with_popup = simulate_popup(base_for_popup, popup_size=(30, 20))
        popup_start, popup_end = add_frames(with_popup, 10, event_type='popup', active=True)
        
        event_list.append({
            'type': 'popup',
            'start': popup_start / fps,
            'end': (popup_end + 1) / fps,
            'start_frame': popup_start,
            'end_frame': popup_end,
            'confidence': 1.0,
            'metadata': {
                'popup_size': (30, 20),
                'position': 'center'
            }
        })
        
        activity_timeline.append({
            'active': True,
            'start': popup_start / fps,
            'end': (popup_end + 1) / fps,
            'start_frame': popup_start,
            'end_frame': popup_end
        })
    
    # Add noise to some frames if requested
    if default_events.get('noise', False):
        noise_start = len(frames) // 4
        noise_end = len(frames) // 2
        for i in range(noise_start, noise_end):
            frames[i] = add_noise(frames[i], 'gaussian', 0.05)
            # Update annotation to indicate noise
            frame_annotations[i]['has_noise'] = True
    
    return {
        'frames': frames,
        'events': event_list,
        'ground_truth': frame_annotations,
        'activity_timeline': activity_timeline,
        'fps': fps,
        'duration': len(frames) / fps,
        'size': size,
        'total_frames': len(frames)
    }