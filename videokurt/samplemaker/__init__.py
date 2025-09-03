"""Sample video and frame generation for VideoKurt testing."""

from .frames import (
    create_blank_frame,
    create_solid_frame,
    create_gradient_frame,
    create_checkerboard
)

from .shapes import (
    add_rectangle,
    add_circle,
    add_text_region
)

from .effects import (
    add_noise,
    add_compression_artifacts
)

from .motion import (
    simulate_scroll,
    simulate_scene_change,
    simulate_popup,
    simulate_video_playback
)

from .sequences import (
    create_frame_sequence,
    create_test_video_frames,
    create_frames_with_pattern
)

__all__ = [
    # Basic frames
    'create_blank_frame',
    'create_solid_frame',
    'create_gradient_frame',
    'create_checkerboard',
    # Shapes
    'add_rectangle',
    'add_circle',
    'add_text_region',
    # Effects
    'add_noise',
    'add_compression_artifacts',
    # Motion
    'simulate_scroll',
    'simulate_scene_change',
    'simulate_popup',
    'simulate_video_playback',
    # Sequences
    'create_frame_sequence',
    'create_test_video_frames',
    'create_frames_with_pattern'
]