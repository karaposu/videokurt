"""Scrolling pattern detection for screen recordings."""

import numpy as np
from typing import Dict, Any, List, Tuple

from ..base import AdvancedFeature


class ScrollingDetection(AdvancedFeature):
    """Detect scrolling patterns in screen recordings."""
    
    FEATURE_NAME = 'scrolling_detection'
    REQUIRED_ANALYSES = ['optical_flow_dense']
    
    def __init__(self, consistency_threshold: float = 0.7,
                 min_scroll_frames: int = 5):
        """
        Args:
            consistency_threshold: How consistent flow must be for scrolling
            min_scroll_frames: Minimum consecutive frames for scroll event
        """
        super().__init__()
        self.consistency_threshold = consistency_threshold
        self.min_scroll_frames = min_scroll_frames
    
    def _compute_advanced(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect scrolling patterns from optical flow.
        
        Returns:
            Dict with scroll events, directions, and speeds
        """
        # Get optical flow
        flow_field = analysis_data['optical_flow_dense'].data['flow_field']
        
        scroll_events = []
        current_scroll = None
        
        for i, flow in enumerate(flow_field):
            # Compute dominant direction
            flow_x = flow[..., 0]
            flow_y = flow[..., 1]
            
            # Average flow vectors
            avg_x = np.mean(flow_x)
            avg_y = np.mean(flow_y)
            
            # Check if flow is consistent (most vectors point same direction)
            if abs(avg_y) > abs(avg_x):  # Vertical motion
                # Check consistency
                if avg_y > 0:
                    consistent_pixels = np.sum(flow_y > 0)
                else:
                    consistent_pixels = np.sum(flow_y < 0)
                
                total_pixels = flow_y.size
                consistency = consistent_pixels / total_pixels
                
                if consistency > self.consistency_threshold:
                    direction = 'down' if avg_y > 0 else 'up'
                    speed = abs(avg_y)
                    
                    if current_scroll and current_scroll['direction'] == direction:
                        # Continue current scroll
                        current_scroll['end_frame'] = i
                        current_scroll['speeds'].append(speed)
                    else:
                        # Save previous scroll if long enough
                        if current_scroll and (current_scroll['end_frame'] - 
                                              current_scroll['start_frame'] >= self.min_scroll_frames):
                            current_scroll['avg_speed'] = np.mean(current_scroll['speeds'])
                            del current_scroll['speeds']  # Clean up
                            scroll_events.append(current_scroll)
                        
                        # Start new scroll
                        current_scroll = {
                            'type': 'scroll',
                            'direction': direction,
                            'start_frame': i,
                            'end_frame': i,
                            'speeds': [speed]
                        }
                else:
                    # No consistent scrolling
                    if current_scroll and (current_scroll['end_frame'] - 
                                          current_scroll['start_frame'] >= self.min_scroll_frames):
                        current_scroll['avg_speed'] = np.mean(current_scroll['speeds'])
                        del current_scroll['speeds']
                        scroll_events.append(current_scroll)
                    current_scroll = None
            
            elif abs(avg_x) > 0.5:  # Horizontal motion (less common)
                # Similar logic for horizontal scrolling
                if avg_x > 0:
                    consistent_pixels = np.sum(flow_x > 0)
                else:
                    consistent_pixels = np.sum(flow_x < 0)
                
                total_pixels = flow_x.size
                consistency = consistent_pixels / total_pixels
                
                if consistency > self.consistency_threshold:
                    direction = 'right' if avg_x > 0 else 'left'
                    speed = abs(avg_x)
                    
                    if current_scroll and current_scroll['direction'] == direction:
                        current_scroll['end_frame'] = i
                        current_scroll['speeds'].append(speed)
                    else:
                        if current_scroll and (current_scroll['end_frame'] - 
                                              current_scroll['start_frame'] >= self.min_scroll_frames):
                            current_scroll['avg_speed'] = np.mean(current_scroll['speeds'])
                            del current_scroll['speeds']
                            scroll_events.append(current_scroll)
                        
                        current_scroll = {
                            'type': 'scroll',
                            'direction': direction,
                            'start_frame': i,
                            'end_frame': i,
                            'speeds': [speed]
                        }
                else:
                    if current_scroll and (current_scroll['end_frame'] - 
                                          current_scroll['start_frame'] >= self.min_scroll_frames):
                        current_scroll['avg_speed'] = np.mean(current_scroll['speeds'])
                        del current_scroll['speeds']
                        scroll_events.append(current_scroll)
                    current_scroll = None
            else:
                # No significant motion
                if current_scroll and (current_scroll['end_frame'] - 
                                      current_scroll['start_frame'] >= self.min_scroll_frames):
                    current_scroll['avg_speed'] = np.mean(current_scroll['speeds'])
                    del current_scroll['speeds']
                    scroll_events.append(current_scroll)
                current_scroll = None
        
        # Don't forget last scroll event
        if current_scroll and (current_scroll['end_frame'] - 
                              current_scroll['start_frame'] >= self.min_scroll_frames):
            current_scroll['avg_speed'] = np.mean(current_scroll['speeds'])
            del current_scroll['speeds']
            scroll_events.append(current_scroll)
        
        # Compute statistics
        total_scroll_frames = sum(e['end_frame'] - e['start_frame'] + 1 for e in scroll_events)
        scroll_directions = {}
        for event in scroll_events:
            direction = event['direction']
            scroll_directions[direction] = scroll_directions.get(direction, 0) + 1
        
        return {
            'scroll_events': scroll_events,
            'num_scroll_events': len(scroll_events),
            'total_scroll_frames': total_scroll_frames,
            'scroll_directions': scroll_directions,
            'dominant_direction': max(scroll_directions.items(), key=lambda x: x[1])[0] if scroll_directions else None
        }