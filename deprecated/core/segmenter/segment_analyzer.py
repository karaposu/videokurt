"""Segment analyzer that groups timeline events into logical segments."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import numpy as np


class SegmentType(Enum):
    """Types of visual patterns detected in video."""
    SCROLLING = "scrolling"           # Continuous scrolling pattern detected
    MIXED_INTERACTION = "mixed_interaction"  # Mixed patterns of movement and changes
    VIDEO_PLAYING = "video_playing"   # Video playback region detected
    IDLE = "idle"                     # No visual changes detected
    CLICKING = "clicking"             # Rapid small area changes (clicks)
    TYPING = "typing"                 # Text input pattern detected
    TRANSITION = "transition"         # Short transition between patterns
    UNKNOWN = "unknown"               # Pattern couldn't be classified


@dataclass
class Segment:
    """A logical segment of video activity."""
    start_frame: int
    end_frame: int
    segment_type: SegmentType
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration_frames(self) -> int:
        """Duration in frames."""
        return self.end_frame - self.start_frame + 1
    
    def to_seconds(self, fps: float) -> tuple:
        """Convert to time in seconds.
        
        Returns:
            (start_seconds, duration_seconds)
        """
        start = self.start_frame / fps
        duration = self.duration_frames / fps
        return start, duration
    
    def to_dict(self, fps: float = 30.0) -> Dict[str, Any]:
        """Convert to dictionary."""
        start_sec, duration_sec = self.to_seconds(fps)
        return {
            'start_frame': self.start_frame,
            'end_frame': self.end_frame,
            'start_seconds': start_sec,
            'duration_seconds': duration_sec,
            'type': self.segment_type.value,
            'confidence': self.confidence,
            'metadata': self.metadata
        }


class SegmentAnalyzer:
    """Analyzes timeline to create logical segments."""
    
    def __init__(self, min_segment_frames: int = 30):
        """Initialize segment analyzer.
        
        Args:
            min_segment_frames: Minimum frames for a segment
        """
        self.min_segment_frames = min_segment_frames
        
    def analyze_timeline(self, timeline: 'Timeline') -> List[Segment]:
        """Analyze timeline and create segments.
        
        Args:
            timeline: Timeline object with events
            
        Returns:
            List of segments
        """
        from ..timeline import EventType
        
        if not timeline.entries:
            # Entire video is idle
            return [Segment(
                start_frame=0,
                end_frame=timeline.total_frames - 1,
                segment_type=SegmentType.IDLE,
                confidence=1.0
            )]
        
        segments = []
        current_segment = None
        
        for entry in timeline.entries:
            segment_type = self._map_event_to_segment_type(entry)
            
            # Check if we should continue current segment or start new one
            if current_segment is None:
                # Start new segment
                current_segment = {
                    'start': entry.frame_start,
                    'end': entry.frame_end,
                    'type': segment_type,
                    'confidence': entry.confidence,
                    'events': [entry]
                }
            elif (self._should_continue_segment(current_segment, entry, segment_type) and
                  entry.frame_start - current_segment['end'] < 30):  # Max 1 second gap at 30fps
                # Continue current segment
                current_segment['end'] = entry.frame_end
                current_segment['confidence'] = max(current_segment['confidence'], entry.confidence)
                current_segment['events'].append(entry)
            else:
                # Save current segment and start new one
                segments.append(self._create_segment(current_segment))
                
                current_segment = {
                    'start': entry.frame_start,
                    'end': entry.frame_end,
                    'type': segment_type,
                    'confidence': entry.confidence,
                    'events': [entry]
                }
        
        # Save last segment
        if current_segment:
            segments.append(self._create_segment(current_segment))
        
        # Fill gaps with idle segments to ensure complete coverage
        segments = self._fill_gaps(segments, timeline.total_frames)
        
        # Merge short segments
        segments = self._merge_short_segments(segments)
        
        # Add transition markers
        segments = self._identify_transitions(segments)
        
        return segments
    
    def analyze_binary(self, binary_timeline: 'BinaryTimeline') -> List[Segment]:
        """Analyze binary timeline and create segments.
        
        Args:
            binary_timeline: BinaryTimeline with active/inactive periods
            
        Returns:
            List of segments
        """
        periods = binary_timeline.get_periods(min_duration_frames=self.min_segment_frames)
        
        segments = []
        for period in periods:
            if period.is_active:
                # Need more analysis to determine specific type
                segment_type = SegmentType.CLICKING
            else:
                segment_type = SegmentType.IDLE
            
            segments.append(Segment(
                start_frame=period.start_frame,
                end_frame=period.end_frame,
                segment_type=segment_type,
                confidence=period.confidence
            ))
        
        return segments
    
    def _map_event_to_segment_type(self, entry) -> SegmentType:
        """Map timeline event to segment type."""
        from ..timeline import EventType
        
        # Check metadata for hints
        if 'activity_type' in entry.metadata:
            activity = entry.metadata['activity_type']
            if activity in ['reading', 'scrolling']:
                return SegmentType.SCROLLING
            elif activity in ['browsing']:
                return SegmentType.MIXED_INTERACTION
            elif activity in ['watching', 'watching_video', 'autoplay']:
                return SegmentType.VIDEO_PLAYING
            elif activity == 'idle':
                return SegmentType.IDLE
        
        # Map based on event type
        if entry.event_type == EventType.USER_SCROLL:
            # Check if continuous scrolling (reading) or intermittent (browsing)
            if entry.confidence > 0.8 and entry.duration_frames > 30:
                return SegmentType.SCROLLING
            return SegmentType.MIXED_INTERACTION
        elif entry.event_type == EventType.VIDEO_PLAYBACK:
            return SegmentType.VIDEO_PLAYING
        elif entry.event_type == EventType.USER_CLICK:
            return SegmentType.CLICKING
        elif entry.event_type == EventType.USER_TYPE:
            return SegmentType.CLICKING
        elif entry.event_type == EventType.IDLE:
            return SegmentType.IDLE
        else:
            return SegmentType.UNKNOWN
    
    def _should_continue_segment(self, current: dict, entry, segment_type: SegmentType) -> bool:
        """Decide if entry should be part of current segment."""
        # Same type continues segment
        if current['type'] == segment_type:
            return True
        
        # Compatible types can be grouped
        compatible = {
            SegmentType.SCROLLING: [SegmentType.MIXED_INTERACTION],
            SegmentType.MIXED_INTERACTION: [SegmentType.SCROLLING, SegmentType.CLICKING],
            SegmentType.CLICKING: [SegmentType.MIXED_INTERACTION],
        }
        
        if current['type'] in compatible:
            return segment_type in compatible[current['type']]
        
        return False
    
    def _create_segment(self, segment_data: dict) -> Segment:
        """Create segment from accumulated data."""
        # Analyze events to refine segment type
        segment_type = segment_data['type']
        
        # Calculate statistics
        metadata = {
            'num_events': len(segment_data['events']),
            'event_types': {}
        }
        
        for event in segment_data['events']:
            event_type = event.event_type.value
            metadata['event_types'][event_type] = metadata['event_types'].get(event_type, 0) + 1
        
        # Refine segment type based on event mix
        if segment_type == SegmentType.MIXED_INTERACTION:
            scroll_count = metadata['event_types'].get('user_scroll', 0)
            click_count = metadata['event_types'].get('user_click', 0)
            
            if scroll_count > click_count * 3:
                segment_type = SegmentType.SCROLLING
            elif click_count > scroll_count * 2:
                segment_type = SegmentType.CLICKING
        
        return Segment(
            start_frame=segment_data['start'],
            end_frame=segment_data['end'],
            segment_type=segment_type,
            confidence=segment_data['confidence'],
            metadata=metadata
        )
    
    def _merge_short_segments(self, segments: List[Segment]) -> List[Segment]:
        """Merge very short segments into neighbors."""
        if len(segments) <= 1:
            return segments
        
        merged = []
        i = 0
        
        while i < len(segments):
            segment = segments[i]
            
            # Check if segment is too short
            if segment.duration_frames < self.min_segment_frames:
                # Try to merge with previous or next
                if merged and i < len(segments) - 1:
                    # Check which neighbor is closer/more compatible
                    prev = merged[-1]
                    next_seg = segments[i + 1]
                    
                    # Merge with previous if compatible
                    if prev.segment_type == segment.segment_type or segment.segment_type == SegmentType.UNKNOWN:
                        prev.end_frame = segment.end_frame
                        prev.metadata['absorbed'] = prev.metadata.get('absorbed', 0) + 1
                        i += 1
                        continue
                
                # Mark as transition if very short
                if segment.duration_frames < 10:
                    segment.segment_type = SegmentType.TRANSITION
            
            merged.append(segment)
            i += 1
        
        return merged
    
    def _fill_gaps(self, segments: List[Segment], total_frames: int) -> List[Segment]:
        """Fill gaps between segments with idle segments."""
        if not segments:
            # Entire timeline is idle
            return [Segment(
                start_frame=0,
                end_frame=total_frames - 1,
                segment_type=SegmentType.IDLE,
                confidence=1.0,
                metadata={'gap_filled': True}
            )]
        
        filled = []
        sorted_segments = sorted(segments, key=lambda s: s.start_frame)
        
        # Check for gap at the beginning
        if sorted_segments[0].start_frame > 0:
            filled.append(Segment(
                start_frame=0,
                end_frame=sorted_segments[0].start_frame - 1,
                segment_type=SegmentType.IDLE,
                confidence=1.0,
                metadata={'gap_filled': True}
            ))
        
        # Add segments and fill gaps between them
        for i, segment in enumerate(sorted_segments):
            filled.append(segment)
            
            # Check for gap after this segment
            if i < len(sorted_segments) - 1:
                next_segment = sorted_segments[i + 1]
                if segment.end_frame + 1 < next_segment.start_frame:
                    filled.append(Segment(
                        start_frame=segment.end_frame + 1,
                        end_frame=next_segment.start_frame - 1,
                        segment_type=SegmentType.IDLE,
                        confidence=1.0,
                        metadata={'gap_filled': True}
                    ))
        
        # Check for gap at the end
        last_segment = sorted_segments[-1]
        if last_segment.end_frame < total_frames - 1:
            filled.append(Segment(
                start_frame=last_segment.end_frame + 1,
                end_frame=total_frames - 1,
                segment_type=SegmentType.IDLE,
                confidence=1.0,
                metadata={'gap_filled': True}
            ))
        
        return filled
    
    def _identify_transitions(self, segments: List[Segment]) -> List[Segment]:
        """Identify and mark transition segments."""
        if len(segments) <= 2:
            return segments
        
        result = []
        
        for i, segment in enumerate(segments):
            # Very short segments between different types are transitions
            if (segment.duration_frames < 15 and 
                0 < i < len(segments) - 1):
                
                prev_type = segments[i-1].segment_type
                next_type = segments[i+1].segment_type
                
                if prev_type != next_type and segment.segment_type == SegmentType.UNKNOWN:
                    segment.segment_type = SegmentType.TRANSITION
                    segment.metadata['transition_from'] = prev_type.value
                    segment.metadata['transition_to'] = next_type.value
            
            result.append(segment)
        
        return result
    
    def get_summary(self, segments: List[Segment], fps: float = 30.0) -> Dict[str, Any]:
        """Get summary statistics of segments.
        
        Args:
            segments: List of segments
            fps: Frames per second
            
        Returns:
            Summary dictionary
        """
        if not segments:
            return {
                'num_segments': 0,
                'total_duration_seconds': 0,
                'segment_types': {}
            }
        
        total_frames = sum(s.duration_frames for s in segments)
        
        # Count by type
        type_counts = {}
        type_durations = {}
        
        for segment in segments:
            seg_type = segment.segment_type.value
            type_counts[seg_type] = type_counts.get(seg_type, 0) + 1
            type_durations[seg_type] = type_durations.get(seg_type, 0) + segment.duration_frames
        
        # Convert durations to seconds
        for seg_type in type_durations:
            type_durations[seg_type] = type_durations[seg_type] / fps
        
        return {
            'num_segments': len(segments),
            'total_duration_seconds': total_frames / fps,
            'segment_types': type_counts,
            'segment_durations': type_durations,
            'average_segment_duration': (total_frames / len(segments)) / fps if segments else 0
        }