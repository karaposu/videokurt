"""Scroll detection for screen recordings.

Identifies scrolling patterns which indicate active user interaction,
very different from passive video playback.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
from ..frame_differencer.base import DifferenceResult


@dataclass
class ScrollEvent:
    """Detected scrolling event."""
    direction: str  # 'up', 'down', 'left', 'right'
    distance: int   # pixels scrolled
    speed: float    # pixels per frame
    frame_start: int
    frame_end: int
    confidence: float
    metadata: Dict[str, Any]


class ScrollDetector:
    """Detects scrolling motion in screen recordings.
    
    Scrolling has distinct characteristics:
    - Full width or height motion
    - Directional consistency
    - Content shifting pattern
    - Brief duration (not continuous like video)
    """
    
    def __init__(
        self,
        min_scroll_distance: int = 20,
        confidence_threshold: float = 0.7
    ):
        """Initialize scroll detector.
        
        Args:
            min_scroll_distance: Minimum pixels to qualify as scroll
            confidence_threshold: Minimum confidence to report scroll
        """
        self.min_scroll_distance = min_scroll_distance
        self.confidence_threshold = confidence_threshold
    
    def detect(
        self,
        diff_sequence: List[DifferenceResult],
        frames: Optional[List[np.ndarray]] = None
    ) -> List[ScrollEvent]:
        """Detect scrolling events in frame sequence.
        
        Args:
            diff_sequence: Sequence of frame differences
            frames: Optional original frames for optical flow analysis
            
        Returns:
            List of detected scroll events
        """
        if len(diff_sequence) < 2:
            return []
        
        scroll_events = []
        
        # Process sequence in windows to find scroll patterns
        i = 0
        while i < len(diff_sequence) - 1:
            # Check if current position starts a scroll
            scroll = self._detect_scroll_at_position(diff_sequence, i)
            
            if scroll and scroll.confidence >= self.confidence_threshold:
                scroll_events.append(scroll)
                i = scroll.frame_end  # Skip past this scroll
            else:
                i += 1
        
        return scroll_events
    
    def _detect_scroll_at_position(
        self,
        diff_sequence: List[DifferenceResult],
        start_idx: int
    ) -> Optional[ScrollEvent]:
        """Detect if a scroll starts at given position.
        
        Returns:
            ScrollEvent if scroll detected, None otherwise
        """
        # Look ahead for scroll pattern
        window_size = min(10, len(diff_sequence) - start_idx)
        
        if window_size < 2:
            return None
        
        # Analyze motion pattern in window
        motion_vectors = []
        for i in range(start_idx, start_idx + window_size - 1):
            vector = self._estimate_motion_vector(
                diff_sequence[i].diff_mask,
                diff_sequence[i + 1].diff_mask
            )
            if vector:
                motion_vectors.append(vector)
        
        if not motion_vectors:
            return None
        
        # Check if vectors indicate scrolling
        scroll_info = self._analyze_motion_vectors(motion_vectors)
        
        if scroll_info:
            direction, distance, confidence = scroll_info
            
            return ScrollEvent(
                direction=direction,
                distance=distance,
                speed=distance / len(motion_vectors),
                frame_start=start_idx,
                frame_end=start_idx + len(motion_vectors),
                confidence=confidence,
                metadata={
                    'coverage': self._calculate_coverage(
                        diff_sequence[start_idx:start_idx + window_size]
                    )
                }
            )
        
        return None
    
    def _estimate_motion_vector(
        self,
        diff1: np.ndarray,
        diff2: np.ndarray
    ) -> Optional[Tuple[float, float]]:
        """Estimate motion vector between two difference masks.
        
        Uses cross-correlation to find the best shift that aligns the patterns.
        This works better than center-of-mass for scrolling detection.
        
        Returns:
            (dx, dy) motion vector or None
        """
        # Check if there's enough activity
        if np.sum(diff1) < 1000 or np.sum(diff2) < 1000:  # Increased threshold
            return None
        
        # Check for noise vs structured motion
        # Noise has random scattered pixels, scrolling has structured patterns
        h, w = diff1.shape
        
        # Calculate coverage to filter noise
        coverage1 = np.sum(diff1 > 0) / diff1.size
        coverage2 = np.sum(diff2 > 0) / diff2.size
        
        # If coverage is too low, it's likely noise
        if coverage1 < 0.05 or coverage2 < 0.05:  # Increased threshold
            return None
        
        # Additional check: scrolling should have consistent coverage
        if abs(coverage1 - coverage2) > 0.3:  # Coverage shouldn't vary wildly
            return None
        
        # For scrolling, we expect motion primarily in one direction
        # Check vertical strips for horizontal scroll, horizontal strips for vertical
        
        h, w = diff1.shape
        
        # Try to detect vertical scroll (content moves up/down)
        # Sum along horizontal axis to get vertical profile
        profile1_v = np.sum(diff1, axis=1)
        profile2_v = np.sum(diff2, axis=1)
        
        # Find best vertical shift using cross-correlation
        max_shift = min(50, h // 4)  # Limit search range
        best_shift_v = 0
        best_corr_v = 0
        
        for shift in range(-max_shift, max_shift + 1):
            if shift == 0 or abs(shift) < 5:  # Skip small shifts
                continue
            # Shift profile1 and compare with profile2
            if shift > 0:
                # Shift down
                shifted = np.zeros_like(profile1_v)
                shifted[shift:] = profile1_v[:-shift]
            else:
                # Shift up
                shifted = np.zeros_like(profile1_v)
                shifted[:shift] = profile1_v[-shift:]
            
            # Calculate correlation
            if np.std(shifted) > 0 and np.std(profile2_v) > 0:
                corr = np.corrcoef(shifted, profile2_v)[0, 1]
                if corr > best_corr_v:
                    best_corr_v = corr
                    best_shift_v = shift
        
        # Try to detect horizontal scroll (content moves left/right)
        # Sum along vertical axis to get horizontal profile
        profile1_h = np.sum(diff1, axis=0)
        profile2_h = np.sum(diff2, axis=0)
        
        # Find best horizontal shift
        max_shift = min(50, w // 4)
        best_shift_h = 0
        best_corr_h = 0
        
        for shift in range(-max_shift, max_shift + 1):
            if shift == 0 or abs(shift) < 5:  # Skip small shifts
                continue
            # Shift profile1 and compare with profile2
            if shift > 0:
                # Shift right
                shifted = np.zeros_like(profile1_h)
                shifted[shift:] = profile1_h[:-shift]
            else:
                # Shift left
                shifted = np.zeros_like(profile1_h)
                shifted[:shift] = profile1_h[-shift:]
            
            # Calculate correlation
            if np.std(shifted) > 0 and np.std(profile2_h) > 0:
                corr = np.corrcoef(shifted, profile2_h)[0, 1]
                if corr > best_corr_h:
                    best_corr_h = corr
                    best_shift_h = shift
        
        # Determine which direction has stronger correlation
        # Lower threshold for detection
        min_corr = 0.2  # Lower threshold for gradient patterns
        
        if (best_corr_v > min_corr and abs(best_shift_v) >= 5) or (best_corr_h > min_corr and abs(best_shift_h) >= 5):
            # Good correlation found with significant motion
            if best_corr_v > best_corr_h and abs(best_shift_v) >= 5:
                # Vertical scroll detected
                return (0, best_shift_v)
            elif best_corr_h >= min_corr and abs(best_shift_h) >= 5:
                # Horizontal scroll detected
                return (best_shift_h, 0)
        
        # Fallback to simple edge detection for strong scrolls
        # Check if activity is concentrated at edges (typical for scrolling)
        edge_threshold = 20
        top_edge = float(np.sum(diff2[:edge_threshold, :])) - float(np.sum(diff1[:edge_threshold, :]))
        bottom_edge = float(np.sum(diff2[-edge_threshold:, :])) - float(np.sum(diff1[-edge_threshold:, :]))
        left_edge = float(np.sum(diff2[:, :edge_threshold])) - float(np.sum(diff1[:, :edge_threshold]))
        right_edge = float(np.sum(diff2[:, -edge_threshold:])) - float(np.sum(diff1[:, -edge_threshold:]))
        
        # Strong edge activity indicates scroll direction
        if abs(top_edge - bottom_edge) > 1000:
            # Vertical scroll
            if top_edge > bottom_edge:
                return (0, 15)  # Scrolling down (new content at top)
            else:
                return (0, -15)  # Scrolling up (new content at bottom)
        
        if abs(left_edge - right_edge) > 1000:
            # Horizontal scroll
            if left_edge > right_edge:
                return (15, 0)  # Scrolling right (new content at left)
            else:
                return (-15, 0)  # Scrolling left (new content at right)
        
        return None
    
    def _analyze_motion_vectors(
        self,
        vectors: List[Tuple[float, float]]
    ) -> Optional[Tuple[str, int, float]]:
        """Analyze motion vectors to detect scrolling.
        
        Returns:
            (direction, distance, confidence) or None
        """
        if not vectors:
            return None
        
        vectors = np.array(vectors)
        dx_mean = np.mean(vectors[:, 0])
        dy_mean = np.mean(vectors[:, 1])
        dx_std = np.std(vectors[:, 0])
        dy_std = np.std(vectors[:, 1])
        
        # Determine primary direction
        if abs(dy_mean) > abs(dx_mean):
            # Vertical scroll
            if dy_mean > self.min_scroll_distance / len(vectors):
                direction = 'down'
                distance = int(abs(dy_mean * len(vectors)))
            elif dy_mean < -self.min_scroll_distance / len(vectors):
                direction = 'up'
                distance = int(abs(dy_mean * len(vectors)))
            else:
                return None
            
            # Confidence based on consistency
            confidence = 1.0 / (1.0 + dy_std / (abs(dy_mean) + 0.01))
            
        else:
            # Horizontal scroll
            if dx_mean > self.min_scroll_distance / len(vectors):
                direction = 'right'
                distance = int(abs(dx_mean * len(vectors)))
            elif dx_mean < -self.min_scroll_distance / len(vectors):
                direction = 'left'
                distance = int(abs(dx_mean * len(vectors)))
            else:
                return None
            
            # Confidence based on consistency
            confidence = 1.0 / (1.0 + dx_std / (abs(dx_mean) + 0.01))
        
        # Additional confidence factor: motion should be mostly in one direction
        directionality = max(abs(dx_mean), abs(dy_mean)) / (abs(dx_mean) + abs(dy_mean) + 0.01)
        confidence *= directionality
        
        return (direction, distance, confidence)
    
    def _calculate_coverage(
        self,
        diff_sequence: List[DifferenceResult]
    ) -> float:
        """Calculate what fraction of screen is affected by motion.
        
        Scrolling typically affects large portions of screen.
        
        Returns:
            Coverage ratio 0.0-1.0
        """
        if not diff_sequence:
            return 0.0
        
        # Average coverage across frames
        coverages = []
        for diff_result in diff_sequence:
            mask = diff_result.diff_mask
            coverage = np.sum(mask > 0) / mask.size
            coverages.append(coverage)
        
        return np.mean(coverages)