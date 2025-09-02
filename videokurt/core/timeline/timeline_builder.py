"""Timeline builder that aggregates detection results."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import numpy as np


class EventType(Enum):
    """Types of events in the timeline."""
    VIDEO_PLAYBACK = "video_playback"
    USER_SCROLL = "user_scroll"
    USER_CLICK = "user_click"
    USER_TYPE = "user_type"
    SCENE_CHANGE = "scene_change"
    IDLE = "idle"
    UNKNOWN = "unknown"


@dataclass
class TimelineEntry:
    """Single entry in the timeline."""
    frame_start: int
    frame_end: int
    event_type: EventType
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration_frames(self) -> int:
        """Get duration in frames."""
        return self.frame_end - self.frame_start + 1
    
    def overlaps_with(self, other: 'TimelineEntry') -> bool:
        """Check if this entry overlaps with another."""
        return not (self.frame_end < other.frame_start or 
                   self.frame_start > other.frame_end)
    
    def merge_with(self, other: 'TimelineEntry') -> 'TimelineEntry':
        """Merge with another overlapping entry."""
        # Take the event with higher confidence
        if self.confidence >= other.confidence:
            primary = self
            secondary = other
        else:
            primary = other
            secondary = self
        
        return TimelineEntry(
            frame_start=min(self.frame_start, other.frame_start),
            frame_end=max(self.frame_end, other.frame_end),
            event_type=primary.event_type,
            confidence=max(self.confidence, other.confidence),
            metadata={
                **secondary.metadata,
                **primary.metadata,
                'merged_from': [
                    {'type': self.event_type.value, 'confidence': self.confidence},
                    {'type': other.event_type.value, 'confidence': other.confidence}
                ]
            }
        )


@dataclass
class Timeline:
    """Complete timeline of events."""
    entries: List[TimelineEntry]
    total_frames: int
    fps: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration_seconds(self) -> float:
        """Get total duration in seconds."""
        return self.total_frames / self.fps if self.fps > 0 else 0
    
    def get_entries_at_frame(self, frame_idx: int) -> List[TimelineEntry]:
        """Get all entries that include the given frame."""
        return [
            entry for entry in self.entries
            if entry.frame_start <= frame_idx <= entry.frame_end
        ]
    
    def get_entries_in_range(self, start: int, end: int) -> List[TimelineEntry]:
        """Get all entries that overlap with the given range."""
        return [
            entry for entry in self.entries
            if not (entry.frame_end < start or entry.frame_start > end)
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert timeline to dictionary."""
        return {
            'total_frames': self.total_frames,
            'fps': self.fps,
            'duration_seconds': self.duration_seconds,
            'num_entries': len(self.entries),
            'entries': [
                {
                    'frame_start': e.frame_start,
                    'frame_end': e.frame_end,
                    'duration_frames': e.duration_frames,
                    'event_type': e.event_type.value,
                    'confidence': e.confidence,
                    'metadata': e.metadata
                }
                for e in self.entries
            ],
            'metadata': self.metadata
        }


class TimelineBuilder:
    """Builds unified timeline from detection results."""
    
    def __init__(self, total_frames: int, fps: float = 30.0):
        """Initialize timeline builder.
        
        Args:
            total_frames: Total number of frames in video
            fps: Frames per second
        """
        self.total_frames = total_frames
        self.fps = fps
        self.entries: List[TimelineEntry] = []
        
    def add_video_regions(self, video_regions: List) -> None:
        """Add video playback regions to timeline.
        
        Args:
            video_regions: List of VideoRegion objects from VideoDetector
        """
        for region in video_regions:
            self.add_entry(
                frame_start=region.frame_start,
                frame_end=region.frame_end,
                event_type=EventType.VIDEO_PLAYBACK,
                confidence=region.confidence,
                metadata={
                    'source': 'video_detector',
                    'region': {
                        'x': region.x,
                        'y': region.y,
                        'width': region.width,
                        'height': region.height
                    },
                    **region.metadata
                }
            )
    
    def add_scroll_events(self, scroll_events: List) -> None:
        """Add scroll events to timeline.
        
        Args:
            scroll_events: List of ScrollEvent objects from ScrollDetector
        """
        for event in scroll_events:
            self.add_entry(
                frame_start=event.frame_start,
                frame_end=event.frame_end,
                event_type=EventType.USER_SCROLL,
                confidence=event.confidence,
                metadata={
                    'source': 'scroll_detector',
                    'direction': event.direction,
                    'distance': event.distance,
                    'speed': event.speed,
                    **event.metadata
                }
            )
    
    def add_activity_states(self, activity_states: List[Tuple[int, int, Any]]) -> None:
        """Add activity classification states to timeline.
        
        Args:
            activity_states: List of (frame_start, frame_end, ActivityState) tuples
        """
        for frame_start, frame_end, state in activity_states:
            # Map activity type to event type
            event_type = EventType.UNKNOWN
            
            if state.activity_type in ['watching_video', 'watching', 'autoplay']:
                event_type = EventType.VIDEO_PLAYBACK
            elif state.activity_type == 'scrolling':
                event_type = EventType.USER_SCROLL
            elif state.activity_type == 'idle':
                event_type = EventType.IDLE
            elif state.activity_type in ['interacting', 'browsing']:
                # Could be click or type, use UNKNOWN for now
                event_type = EventType.UNKNOWN
            
            self.add_entry(
                frame_start=frame_start,
                frame_end=frame_end,
                event_type=event_type,
                confidence=state.confidence,
                metadata={
                    'user_active': state.user_active,
                    'screen_active': state.screen_active,
                    'activity_type': state.activity_type
                }
            )
    
    def add_entry(
        self,
        frame_start: int,
        frame_end: int,
        event_type: EventType,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a single entry to timeline.
        
        Args:
            frame_start: Starting frame index
            frame_end: Ending frame index
            event_type: Type of event
            confidence: Confidence score
            metadata: Additional metadata
        """
        # Handle zero-frame videos
        if self.total_frames == 0:
            return
        
        # Fix frame order if reversed
        if frame_end < frame_start:
            frame_start, frame_end = frame_end, frame_start
        
        # Clamp to valid frame range
        frame_start = max(0, min(frame_start, self.total_frames - 1))
        frame_end = max(frame_start, min(frame_end, self.total_frames - 1))
        
        # Handle invalid confidence values
        if not np.isfinite(confidence):
            confidence = 1.0
        confidence = max(0.0, min(1.0, confidence))
        
        # Store original metadata with any additions
        final_metadata = metadata.copy() if metadata else {}
        if 'source' not in final_metadata and metadata:
            # Preserve source information if available
            for key in ['source', 'direction', 'region']:
                if key in metadata:
                    final_metadata[key] = metadata[key]
        
        entry = TimelineEntry(
            frame_start=frame_start,
            frame_end=frame_end,
            event_type=event_type,
            confidence=confidence,
            metadata=final_metadata
        )
        
        self.entries.append(entry)
    
    def merge_overlapping_entries(self, confidence_threshold: float = 0.5) -> None:
        """Merge overlapping timeline entries.
        
        When entries overlap, the one with higher confidence takes precedence.
        Only merges entries of the same event type to preserve different activities.
        
        Args:
            confidence_threshold: Minimum confidence to keep an entry
        """
        if not self.entries:
            return
        
        # Filter by confidence
        filtered = [e for e in self.entries if e.confidence >= confidence_threshold]
        
        # Group by event type to only merge same types
        by_type = {}
        for entry in filtered:
            if entry.event_type not in by_type:
                by_type[entry.event_type] = []
            by_type[entry.event_type].append(entry)
        
        merged = []
        
        # Merge within each event type
        for event_type, entries in by_type.items():
            # Sort by start frame
            entries.sort(key=lambda e: (e.frame_start, -e.confidence))
            
            type_merged = []
            for entry in entries:
                # Check if it overlaps with any existing merged entry of same type
                overlapping = None
                for existing in type_merged:
                    if entry.overlaps_with(existing):
                        overlapping = existing
                        break
                
                if overlapping:
                    # Merge with existing
                    type_merged.remove(overlapping)
                    merged_entry = entry.merge_with(overlapping)
                    # Preserve source metadata from higher confidence entry
                    if entry.confidence >= overlapping.confidence:
                        if 'source' in entry.metadata:
                            merged_entry.metadata['source'] = entry.metadata['source']
                    elif 'source' in overlapping.metadata:
                        merged_entry.metadata['source'] = overlapping.metadata['source']
                    type_merged.append(merged_entry)
                else:
                    # Add as new entry
                    type_merged.append(entry)
            
            merged.extend(type_merged)
        
        # Sort final result
        merged.sort(key=lambda e: e.frame_start)
        self.entries = merged
    
    def fill_gaps(self, max_gap_frames: int = 5) -> None:
        """Fill small gaps between similar events.
        
        Args:
            max_gap_frames: Maximum gap size to fill
        """
        if len(self.entries) < 2:
            return
        
        filled = []
        prev_entry = None
        
        for entry in sorted(self.entries, key=lambda e: e.frame_start):
            if prev_entry is not None:
                gap = entry.frame_start - prev_entry.frame_end - 1
                
                # Fill gap if small enough and same event type (gap=0 means consecutive)
                if 0 <= gap <= max_gap_frames and prev_entry.event_type == entry.event_type:
                    # Extend previous entry to cover the gap
                    prev_entry = TimelineEntry(
                        frame_start=prev_entry.frame_start,
                        frame_end=entry.frame_end,
                        event_type=prev_entry.event_type,
                        confidence=min(prev_entry.confidence, entry.confidence),
                        metadata={
                            **prev_entry.metadata,
                            'filled_gap': gap
                        }
                    )
                else:
                    filled.append(prev_entry)
                    prev_entry = entry
            else:
                prev_entry = entry
        
        if prev_entry is not None:
            filled.append(prev_entry)
        
        self.entries = filled
    
    def add_idle_periods(self, min_idle_frames: int = 10) -> None:
        """Add idle periods where no events are detected.
        
        Args:
            min_idle_frames: Minimum frames to consider as idle period
        """
        if not self.entries:
            # Entire video is idle
            if self.total_frames >= min_idle_frames:
                self.add_entry(
                    frame_start=0,
                    frame_end=self.total_frames - 1,
                    event_type=EventType.IDLE,
                    confidence=1.0
                )
            return
        
        # Sort entries
        sorted_entries = sorted(self.entries, key=lambda e: e.frame_start)
        idle_entries = []
        
        # Check beginning
        if sorted_entries[0].frame_start >= min_idle_frames:
            idle_entries.append(TimelineEntry(
                frame_start=0,
                frame_end=sorted_entries[0].frame_start - 1,
                event_type=EventType.IDLE,
                confidence=1.0,
                metadata={'location': 'beginning'}
            ))
        
        # Check gaps between entries
        for i in range(len(sorted_entries) - 1):
            gap_start = sorted_entries[i].frame_end + 1
            gap_end = sorted_entries[i + 1].frame_start - 1
            
            if gap_end - gap_start + 1 >= min_idle_frames:
                idle_entries.append(TimelineEntry(
                    frame_start=gap_start,
                    frame_end=gap_end,
                    event_type=EventType.IDLE,
                    confidence=1.0,
                    metadata={'location': 'middle'}
                ))
        
        # Check end
        if self.total_frames - sorted_entries[-1].frame_end - 1 >= min_idle_frames:
            idle_entries.append(TimelineEntry(
                frame_start=sorted_entries[-1].frame_end + 1,
                frame_end=self.total_frames - 1,
                event_type=EventType.IDLE,
                confidence=1.0,
                metadata={'location': 'end'}
            ))
        
        # Add idle entries
        self.entries.extend(idle_entries)
        self.entries.sort(key=lambda e: e.frame_start)
    
    def build(self) -> Timeline:
        """Build the final timeline.
        
        Returns:
            Timeline object with all entries
        """
        # Sort entries by start frame
        self.entries.sort(key=lambda e: e.frame_start)
        
        # Calculate statistics
        stats = self._calculate_statistics()
        
        return Timeline(
            entries=self.entries,
            total_frames=self.total_frames,
            fps=self.fps,
            metadata=stats
        )
    
    def _calculate_statistics(self) -> Dict[str, Any]:
        """Calculate timeline statistics."""
        if not self.entries:
            return {
                'num_events': 0,
                'coverage_ratio': 0.0,
                'event_types': {}
            }
        
        # Count event types
        event_counts = {}
        covered_frames = set()
        
        for entry in self.entries:
            event_type = entry.event_type.value
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
            
            for frame in range(entry.frame_start, entry.frame_end + 1):
                covered_frames.add(frame)
        
        coverage_ratio = len(covered_frames) / self.total_frames if self.total_frames > 0 else 0
        
        return {
            'num_events': len(self.entries),
            'coverage_ratio': coverage_ratio,
            'event_types': event_counts,
            'uncovered_frames': self.total_frames - len(covered_frames)
        }