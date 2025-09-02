"""
Test 04: Ground Truth
Tests ground truth data accuracy and completeness.
Critical for detection accuracy measurement against known truth.


# to run python -m videokurt.smoke_tests.samplemaker.test_04_ground_truth


"""

import sys
import numpy as np
from videokurt.samplemaker import create_test_video_frames


def test_event_timing_accuracy():
    """Test 1: Validate event start/end times and frame indices."""
    print("\nTest 1: test_event_timing_accuracy")
    print("  Validating event timing information...")
    
    passed = True
    
    # Generate test video with known events
    test_data = create_test_video_frames(
        size=(30, 30),
        events={
            'scene_changes': True,
            'scrolls': True,
            'popups': True,
            'idle_periods': True,
            'noise': False
        },
        fps=10.0
    )
    
    events = test_data['events']
    fps = test_data['fps']
    total_frames = test_data['total_frames']
    
    print(f"  Generated {len(events)} events, {total_frames} frames at {fps} fps")
    
    for i, event in enumerate(events):
        # Check required fields
        required_fields = ['type', 'start', 'end', 'start_frame', 'end_frame', 'confidence']
        for field in required_fields:
            if field not in event:
                print(f"  ✗ ERROR: Event {i} missing field '{field}'")
                passed = False
        
        if not passed:
            continue
            
        # Validate timing consistency
        calculated_start = event['start_frame'] / fps
        calculated_end = (event['end_frame'] + 1) / fps
        
        if abs(calculated_start - event['start']) > 0.01:
            print(f"  ✗ ERROR: Event {event['type']} start time mismatch")
            print(f"    Calculated: {calculated_start:.3f}, Stored: {event['start']:.3f}")
            passed = False
        
        if abs(calculated_end - event['end']) > 0.01:
            print(f"  ✗ ERROR: Event {event['type']} end time mismatch")
            print(f"    Calculated: {calculated_end:.3f}, Stored: {event['end']:.3f}")
            passed = False
        
        # Check frame indices are valid
        if event['start_frame'] < 0 or event['start_frame'] >= total_frames:
            print(f"  ✗ ERROR: Event {event['type']} invalid start_frame: {event['start_frame']}")
            passed = False
        
        if event['end_frame'] < 0 or event['end_frame'] >= total_frames:
            print(f"  ✗ ERROR: Event {event['type']} invalid end_frame: {event['end_frame']}")
            passed = False
        
        if event['start_frame'] > event['end_frame']:
            print(f"  ✗ ERROR: Event {event['type']} start_frame > end_frame")
            passed = False
        
        # Check confidence is 1.0 for ground truth
        if event['confidence'] != 1.0:
            print(f"  ✗ ERROR: Ground truth confidence should be 1.0, got {event['confidence']}")
            passed = False
    
    if passed:
        print(f"  ✓ All {len(events)} events have valid timing")
        
        # Report event types found
        event_types = set(e['type'] for e in events)
        print(f"  ✓ Event types: {', '.join(sorted(event_types))}")
    
    return passed


def test_activity_timeline_consistency():
    """Test 2: Ensure binary timeline matches events."""
    print("\nTest 2: test_activity_timeline_consistency")
    print("  Checking activity timeline consistency...")
    
    passed = True
    
    test_data = create_test_video_frames(
        size=(30, 30),
        events={
            'scene_changes': True,
            'scrolls': True,
            'idle_periods': True
        },
        fps=10.0
    )
    
    timeline = test_data['activity_timeline']
    events = test_data['events']
    total_frames = test_data['total_frames']
    
    print(f"  Timeline has {len(timeline)} periods")
    
    # Check timeline coverage
    covered_frames = set()
    for period in timeline:
        # Check required fields
        if 'active' not in period or 'start' not in period or 'end' not in period:
            print(f"  ✗ ERROR: Timeline period missing required fields")
            passed = False
            continue
        
        if 'start_frame' not in period or 'end_frame' not in period:
            print(f"  ✗ ERROR: Timeline period missing frame indices")
            passed = False
            continue
        
        # Check for overlaps
        for frame_idx in range(period['start_frame'], period['end_frame'] + 1):
            if frame_idx in covered_frames:
                print(f"  ✗ ERROR: Frame {frame_idx} appears in multiple timeline periods")
                passed = False
            covered_frames.add(frame_idx)
    
    # Check for gaps
    if len(covered_frames) != total_frames:
        missing = set(range(total_frames)) - covered_frames
        if missing:
            print(f"  ✗ ERROR: Frames not covered by timeline: {sorted(list(missing)[:5])}...")
            passed = False
    else:
        print(f"  ✓ Timeline covers all {total_frames} frames")
    
    # Check that active periods correspond to events
    active_periods = [p for p in timeline if p['active']]
    inactive_periods = [p for p in timeline if not p['active']]
    
    print(f"  Active periods: {len(active_periods)}, Inactive: {len(inactive_periods)}")
    
    # Idle events should correspond to inactive periods
    idle_events = [e for e in events if e['type'] == 'idle_wait']
    for idle_event in idle_events:
        # Find corresponding timeline period
        found = False
        for period in inactive_periods:
            if (abs(period['start'] - idle_event['start']) < 0.1 and 
                abs(period['end'] - idle_event['end']) < 0.1):
                found = True
                break
        
        if not found:
            print(f"  ✗ ERROR: Idle event has no matching inactive period")
            print(f"    Event: {idle_event['start']:.2f}-{idle_event['end']:.2f}")
            passed = False
    
    if passed:
        print(f"  ✓ Timeline periods consistent with events")
    
    # Check chronological order
    for i in range(1, len(timeline)):
        if timeline[i]['start'] < timeline[i-1]['end']:
            print(f"  ✗ ERROR: Timeline periods overlap or out of order")
            passed = False
    
    if passed:
        print(f"  ✓ Timeline in chronological order")
    
    return passed


def test_frame_annotations():
    """Test 3: Verify frame-by-frame annotations."""
    print("\nTest 3: test_frame_annotations")
    print("  Verifying frame-level ground truth annotations...")
    
    passed = True
    
    test_data = create_test_video_frames(
        size=(25, 25),
        events={'scene_changes': True, 'scrolls': True},
        fps=15.0
    )
    
    annotations = test_data['ground_truth']
    frames = test_data['frames']
    fps = test_data['fps']
    
    if len(annotations) != len(frames):
        print(f"  ✗ ERROR: Annotation count ({len(annotations)}) != frame count ({len(frames)})")
        passed = False
    else:
        print(f"  ✓ One annotation per frame ({len(annotations)} annotations)")
    
    # Check each annotation
    for i, ann in enumerate(annotations):
        # Check required fields
        required = ['frame_idx', 'timestamp', 'event_type', 'active']
        for field in required:
            if field not in ann:
                print(f"  ✗ ERROR: Annotation {i} missing field '{field}'")
                passed = False
                break
        
        if not passed:
            break
        
        # Verify frame index matches position
        if ann['frame_idx'] != i:
            print(f"  ✗ ERROR: Annotation {i} has wrong frame_idx: {ann['frame_idx']}")
            passed = False
        
        # Verify timestamp calculation
        expected_timestamp = i / fps
        if abs(ann['timestamp'] - expected_timestamp) > 0.001:
            print(f"  ✗ ERROR: Annotation {i} timestamp incorrect")
            print(f"    Expected: {expected_timestamp:.3f}, Got: {ann['timestamp']:.3f}")
            passed = False
        
        # Check event_type is valid
        valid_types = ['idle', 'scene_change', 'post_scene_change', 'scroll', 
                      'idle_wait', 'popup', 'pre_popup']
        if ann['event_type'] not in valid_types:
            print(f"  ✗ ERROR: Unknown event type: {ann['event_type']}")
            passed = False
        
        # Check active is boolean
        if not isinstance(ann['active'], bool):
            print(f"  ✗ ERROR: 'active' field should be boolean, got {type(ann['active'])}")
            passed = False
    
    if passed:
        # Summarize annotation statistics
        event_types = {}
        active_count = 0
        
        for ann in annotations:
            event_types[ann['event_type']] = event_types.get(ann['event_type'], 0) + 1
            if ann['active']:
                active_count += 1
        
        print(f"  ✓ All annotations valid")
        print(f"  ✓ Active frames: {active_count}/{len(annotations)} ({100*active_count/len(annotations):.1f}%)")
        print(f"  ✓ Event distribution: {dict(event_types)}")
    
    return passed


def test_metadata_completeness():
    """Test 4: Check event metadata contains required fields."""
    print("\nTest 4: test_metadata_completeness")
    print("  Checking event metadata completeness...")
    
    passed = True
    
    test_data = create_test_video_frames(
        size=(40, 40),
        events={
            'scrolls': True,
            'popups': True
        },
        fps=20.0
    )
    
    events = test_data['events']
    
    # Check metadata for specific event types
    for event in events:
        if event['type'] == 'scroll':
            if 'metadata' not in event:
                print(f"  ✗ ERROR: Scroll event missing metadata")
                passed = False
            else:
                metadata = event['metadata']
                required = ['direction', 'total_pixels', 'velocity']
                for field in required:
                    if field not in metadata:
                        print(f"  ✗ ERROR: Scroll metadata missing '{field}'")
                        passed = False
                
                if 'direction' in metadata:
                    if metadata['direction'] not in ['up', 'down', 'left', 'right']:
                        print(f"  ✗ ERROR: Invalid scroll direction: {metadata['direction']}")
                        passed = False
                    else:
                        print(f"  ✓ Scroll event: {metadata['direction']}, "
                              f"{metadata.get('total_pixels', 0)} pixels, "
                              f"{metadata.get('velocity', 0):.1f} px/s")
        
        elif event['type'] == 'popup':
            if 'metadata' not in event:
                print(f"  ✗ ERROR: Popup event missing metadata")
                passed = False
            else:
                metadata = event['metadata']
                required = ['popup_size', 'position']
                for field in required:
                    if field not in metadata:
                        print(f"  ✗ ERROR: Popup metadata missing '{field}'")
                        passed = False
                
                if 'popup_size' in metadata:
                    size = metadata['popup_size']
                    if not (isinstance(size, tuple) and len(size) == 2):
                        print(f"  ✗ ERROR: Invalid popup size format: {size}")
                        passed = False
                    else:
                        print(f"  ✓ Popup event: size {size}, position {metadata.get('position')}")
    
    # Check that all events have confidence = 1.0
    for event in events:
        if event.get('confidence', 0) != 1.0:
            print(f"  ✗ ERROR: Ground truth event should have confidence=1.0")
            passed = False
    
    if passed:
        print(f"  ✓ All event metadata complete and valid")
    
    return passed


def test_timeline_coverage():
    """Test 5: Ensure no gaps or overlaps in timeline."""
    print("\nTest 5: test_timeline_coverage")
    print("  Testing timeline coverage and continuity...")
    
    passed = True
    
    test_data = create_test_video_frames(
        size=(35, 35),
        events={
            'scene_changes': True,
            'scrolls': True,
            'idle_periods': True,
            'popups': True
        },
        fps=12.0
    )
    
    timeline = test_data['activity_timeline']
    total_duration = test_data['duration']
    total_frames = test_data['total_frames']
    
    print(f"  Video duration: {total_duration:.2f}s, {total_frames} frames")
    print(f"  Timeline has {len(timeline)} periods")
    
    # Check continuity
    if len(timeline) > 0:
        # First period should start at 0
        if timeline[0]['start'] != 0:
            print(f"  ✗ ERROR: Timeline doesn't start at 0, starts at {timeline[0]['start']}")
            passed = False
        else:
            print(f"  ✓ Timeline starts at t=0")
        
        # Check for gaps between periods
        for i in range(1, len(timeline)):
            prev_end = timeline[i-1]['end']
            curr_start = timeline[i]['start']
            
            gap = curr_start - prev_end
            if abs(gap) > 0.001:  # Small tolerance for floating point
                print(f"  ✗ ERROR: Gap in timeline between periods {i-1} and {i}")
                print(f"    Period {i-1} ends at {prev_end:.3f}, Period {i} starts at {curr_start:.3f}")
                passed = False
        
        # Last period should end at video duration
        last_end = timeline[-1]['end']
        if abs(last_end - total_duration) > 0.1:
            print(f"  ✗ ERROR: Timeline doesn't cover full duration")
            print(f"    Timeline ends at {last_end:.3f}, video duration is {total_duration:.3f}")
            passed = False
        else:
            print(f"  ✓ Timeline covers full duration")
    
    # Check frame coverage
    frame_coverage = [False] * total_frames
    
    for period in timeline:
        for frame_idx in range(period['start_frame'], period['end_frame'] + 1):
            if frame_idx < total_frames:
                if frame_coverage[frame_idx]:
                    print(f"  ✗ ERROR: Frame {frame_idx} covered by multiple periods")
                    passed = False
                frame_coverage[frame_idx] = True
    
    uncovered = [i for i, covered in enumerate(frame_coverage) if not covered]
    if uncovered:
        print(f"  ✗ ERROR: Frames not covered: {uncovered[:10]}...")
        passed = False
    else:
        print(f"  ✓ All frames covered exactly once")
    
    # Calculate and verify activity ratio
    active_time = sum(p['end'] - p['start'] for p in timeline if p['active'])
    inactive_time = sum(p['end'] - p['start'] for p in timeline if not p['active'])
    
    activity_ratio = active_time / total_duration if total_duration > 0 else 0
    
    print(f"  Active time: {active_time:.2f}s ({activity_ratio:.1%})")
    print(f"  Inactive time: {inactive_time:.2f}s ({(1-activity_ratio):.1%})")
    
    # Verify times add up
    if abs((active_time + inactive_time) - total_duration) > 0.1:
        print(f"  ✗ ERROR: Active + Inactive time doesn't equal total duration")
        passed = False
    else:
        print(f"  ✓ Timeline durations sum correctly")
    
    return passed


def main():
    """Run all tests and report results."""
    print("=" * 60)
    print("Running: test_04_ground_truth.py")
    print("=" * 60)
    
    tests = [
        ("test_event_timing_accuracy", test_event_timing_accuracy),
        ("test_activity_timeline_consistency", test_activity_timeline_consistency),
        ("test_frame_annotations", test_frame_annotations),
        ("test_metadata_completeness", test_metadata_completeness),
        ("test_timeline_coverage", test_timeline_coverage),
    ]
    
    passed_count = 0
    failed_tests = []
    
    for test_name, test_func in tests:
        try:
            if test_func():
                print(f"  PASSED: {test_name}")
                passed_count += 1
            else:
                print(f"  FAILED: {test_name}")
                failed_tests.append(test_name)
        except Exception as e:
            print(f"  EXCEPTION in {test_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            failed_tests.append(test_name)
    
    print("\n" + "=" * 60)
    print(f"SUMMARY: {passed_count}/{len(tests)} tests passed")
    
    if failed_tests:
        print("\nFailed tests:")
        for test_name in failed_tests:
            print(f"  - {test_name}")
        sys.exit(1)
    else:
        print("All tests passed successfully!")
    
    print("=" * 60)


if __name__ == "__main__":
    main()