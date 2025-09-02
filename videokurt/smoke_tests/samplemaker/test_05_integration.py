"""
Test 05: Integration
Tests complete sequence generation and cross-module compatibility.
Critical for real-world usage with multiple events and frame differencing.

# to run python -m videokurt.smoke_tests.samplemaker.test_05_integration

"""

import sys
import numpy as np
from videokurt.samplemaker import (
    create_test_video_frames,
    create_frame_sequence
)


def test_complete_video_generation():
    """Test 1: Validate full test video creation."""
    print("\nTest 1: test_complete_video_generation")
    print("  Testing complete video generation with all event types...")
    
    passed = True
    
    # Generate comprehensive test video
    test_data = create_test_video_frames(
        size=(50, 50),
        events={
            'scene_changes': True,
            'scrolls': True,
            'popups': True,
            'idle_periods': True,
            'noise': False
        },
        fps=15.0
    )
    
    # Check all required fields are present
    required_fields = [
        'frames', 'events', 'ground_truth', 'activity_timeline',
        'fps', 'duration', 'size', 'total_frames'
    ]
    
    for field in required_fields:
        if field not in test_data:
            print(f"  ✗ ERROR: Missing required field '{field}'")
            passed = False
        else:
            print(f"  ✓ Field '{field}' present")
    
    if not passed:
        return False
    
    # Validate data consistency
    frames = test_data['frames']
    total_frames = test_data['total_frames']
    duration = test_data['duration']
    fps = test_data['fps']
    size = test_data['size']
    
    # Check frame count
    if len(frames) != total_frames:
        print(f"  ✗ ERROR: Frame count mismatch: len(frames)={len(frames)}, total_frames={total_frames}")
        passed = False
    else:
        print(f"  ✓ Frame count consistent: {total_frames} frames")
    
    # Check duration calculation
    calculated_duration = total_frames / fps
    if abs(calculated_duration - duration) > 0.01:
        print(f"  ✗ ERROR: Duration mismatch: calculated={calculated_duration:.2f}, stored={duration:.2f}")
        passed = False
    else:
        print(f"  ✓ Duration correct: {duration:.2f}s")
    
    # Check frame dimensions
    for i, frame in enumerate(frames[:5]):  # Check first 5 frames
        if frame.shape != size:
            print(f"  ✗ ERROR: Frame {i} has wrong size: expected {size}, got {frame.shape}")
            passed = False
    
    if passed:
        print(f"  ✓ Frame dimensions correct: {size}")
    
    # Check frame data types
    for i, frame in enumerate(frames[:5]):
        if frame.dtype != np.uint8:
            print(f"  ✗ ERROR: Frame {i} has wrong dtype: {frame.dtype}")
            passed = False
    
    if passed:
        print(f"  ✓ Frame data types correct (uint8)")
    
    # Verify events were created
    if len(test_data['events']) == 0:
        print(f"  ✗ ERROR: No events generated")
        passed = False
    else:
        event_types = set(e['type'] for e in test_data['events'])
        print(f"  ✓ Generated {len(test_data['events'])} events: {', '.join(event_types)}")
    
    return passed


def test_event_sequence_ordering():
    """Test 2: Ensure events occur in correct order."""
    print("\nTest 2: test_event_sequence_ordering")
    print("  Testing event sequence ordering...")
    
    passed = True
    
    test_data = create_test_video_frames(
        size=(40, 40),
        events={
            'scene_changes': True,
            'scrolls': True,
            'idle_periods': True,
            'popups': True
        },
        fps=10.0
    )
    
    events = test_data['events']
    
    # Events should be in chronological order
    for i in range(1, len(events)):
        if events[i]['start'] < events[i-1]['start']:
            print(f"  ✗ ERROR: Events not in chronological order")
            print(f"    Event {i-1} starts at {events[i-1]['start']:.2f}")
            print(f"    Event {i} starts at {events[i]['start']:.2f}")
            passed = False
    
    if passed:
        print(f"  ✓ Events in chronological order")
    
    # Check for event overlaps (except where expected)
    for i in range(len(events)):
        for j in range(i+1, len(events)):
            event1 = events[i]
            event2 = events[j]
            
            # Check if events overlap
            if event1['end'] > event2['start'] and event1['start'] < event2['end']:
                # Some overlaps might be valid (e.g., scene_change is instant)
                if event1['type'] != 'scene_change' and event2['type'] != 'scene_change':
                    print(f"  ✗ ERROR: Events overlap inappropriately")
                    print(f"    {event1['type']}: {event1['start']:.2f}-{event1['end']:.2f}")
                    print(f"    {event2['type']}: {event2['start']:.2f}-{event2['end']:.2f}")
                    passed = False
    
    if passed:
        print(f"  ✓ No inappropriate event overlaps")
    
    # Verify expected event sequence pattern
    event_sequence = [e['type'] for e in events]
    
    # Should have idle at beginning
    if 'idle_wait' not in event_sequence:
        print(f"  WARNING: No idle period in sequence")
    
    # Scene changes should happen quickly (single frame)
    for event in events:
        if event['type'] == 'scene_change':
            duration = event['end'] - event['start']
            if duration > 0.2:  # Should be ~1 frame
                print(f"  ✗ ERROR: Scene change too long: {duration:.2f}s")
                passed = False
    
    if passed:
        print(f"  ✓ Event durations appropriate")
    
    return passed


def test_fps_timing_calculation():
    """Test 3: Verify frame rate affects timing correctly."""
    print("\nTest 3: test_fps_timing_calculation")
    print("  Testing FPS impact on timing calculations...")
    
    passed = True
    
    # Test with different frame rates
    fps_values = [5.0, 10.0, 30.0, 60.0]
    
    for fps in fps_values:
        test_data = create_test_video_frames(
            size=(20, 20),
            events={'scene_changes': True, 'scrolls': True},
            fps=fps
        )
        
        # Check frame timing
        frame_duration = 1.0 / fps
        total_frames = test_data['total_frames']
        expected_duration = total_frames * frame_duration
        actual_duration = test_data['duration']
        
        if abs(expected_duration - actual_duration) > 0.01:
            print(f"  ✗ ERROR: Duration incorrect for {fps} fps")
            print(f"    Expected: {expected_duration:.3f}, Got: {actual_duration:.3f}")
            passed = False
        else:
            print(f"  ✓ {fps:5.1f} fps: duration = {actual_duration:.2f}s for {total_frames} frames")
        
        # Check event timings
        for event in test_data['events']:
            # Event times should be multiples of frame_duration
            start_frames = event['start'] * fps
            end_frames = event['end'] * fps
            
            # Should be close to integer frame numbers
            if abs(start_frames - round(start_frames)) > 0.01:
                print(f"  ✗ ERROR: Event start time not aligned to frames at {fps} fps")
                passed = False
        
        # Check ground truth timestamps
        annotations = test_data['ground_truth']
        for i, ann in enumerate(annotations[:5]):  # Check first 5
            expected_time = i / fps
            if abs(ann['timestamp'] - expected_time) > 0.001:
                print(f"  ✗ ERROR: Annotation timestamp wrong at {fps} fps")
                print(f"    Frame {i}: expected {expected_time:.3f}, got {ann['timestamp']:.3f}")
                passed = False
    
    if passed:
        print(f"  ✓ All FPS values produce correct timings")
    
    return passed


def test_data_structure_integrity():
    """Test 4: Confirm all output fields present and valid."""
    print("\nTest 4: test_data_structure_integrity")
    print("  Testing data structure integrity...")
    
    passed = True
    
    # Test with minimal configuration
    test_data = create_test_video_frames(
        size=(30, 30),
        events={'scene_changes': True},
        fps=10.0
    )
    
    # Check frames are numpy arrays
    frames = test_data['frames']
    for i, frame in enumerate(frames[:5]):
        if not isinstance(frame, np.ndarray):
            print(f"  ✗ ERROR: Frame {i} is not numpy array: {type(frame)}")
            passed = False
    
    if passed:
        print(f"  ✓ All frames are numpy arrays")
    
    # Check events structure
    for i, event in enumerate(test_data['events']):
        if not isinstance(event, dict):
            print(f"  ✗ ERROR: Event {i} is not dictionary")
            passed = False
            continue
        
        # Check event has valid type
        if 'type' not in event or not isinstance(event['type'], str):
            print(f"  ✗ ERROR: Event {i} has invalid type field")
            passed = False
        
        # Check timing fields are numbers
        for field in ['start', 'end']:
            if field not in event or not isinstance(event[field], (int, float)):
                print(f"  ✗ ERROR: Event {i} has invalid {field} field")
                passed = False
    
    if passed:
        print(f"  ✓ Event structures valid")
    
    # Check ground truth structure
    for i, ann in enumerate(test_data['ground_truth'][:5]):
        if not isinstance(ann, dict):
            print(f"  ✗ ERROR: Annotation {i} is not dictionary")
            passed = False
            continue
        
        # Check required fields
        for field in ['frame_idx', 'timestamp', 'event_type', 'active']:
            if field not in ann:
                print(f"  ✗ ERROR: Annotation {i} missing field '{field}'")
                passed = False
    
    if passed:
        print(f"  ✓ Ground truth structures valid")
    
    # Check activity timeline structure
    for i, period in enumerate(test_data['activity_timeline']):
        if not isinstance(period, dict):
            print(f"  ✗ ERROR: Timeline period {i} is not dictionary")
            passed = False
            continue
        
        # Check boolean active field
        if 'active' not in period or not isinstance(period['active'], bool):
            print(f"  ✗ ERROR: Timeline period {i} has invalid 'active' field")
            passed = False
    
    if passed:
        print(f"  ✓ Activity timeline structures valid")
    
    # Test frame sequence function
    sequences = {
        'activity': create_frame_sequence(10, (20, 20), 'activity'),
        'idle': create_frame_sequence(10, (20, 20), 'idle'),
        'mixed': create_frame_sequence(10, (20, 20), 'mixed')
    }
    
    for scenario, frames in sequences.items():
        if len(frames) != 10:
            print(f"  ✗ ERROR: {scenario} sequence has wrong length: {len(frames)}")
            passed = False
        else:
            print(f"  ✓ {scenario} sequence: {len(frames)} frames")
    
    return passed


def test_cross_module_compatibility():
    """Test 5: Test integration with frame differencing module."""
    print("\nTest 5: test_cross_module_compatibility")
    print("  Testing compatibility with frame differencing module...")
    
    passed = True
    
    try:
        from videokurt.core import SimpleFrameDiff
        differencer_available = True
    except ImportError:
        print("  WARNING: Frame differencing module not available, skipping integration test")
        differencer_available = False
    
    if differencer_available:
        # Generate test video
        test_data = create_test_video_frames(
            size=(30, 30),
            events={'scrolls': True, 'idle_periods': True},
            fps=10.0
        )
        
        frames = test_data['frames']
        events = test_data['events']
        timeline = test_data['activity_timeline']
        
        # Create differencer
        diff = SimpleFrameDiff(blur_kernel=3, noise_threshold=5)
        
        # Test differencing during active period
        active_period = next((p for p in timeline if p['active']), None)
        if active_period:
            start_frame = active_period['start_frame']
            end_frame = min(start_frame + 3, active_period['end_frame'])
            
            differences = []
            for i in range(start_frame, end_frame):
                if i + 1 < len(frames):
                    result = diff.compute_difference(frames[i], frames[i+1])
                    differences.append(result.score)
                    
                    if result.score <= 0:
                        print(f"  ✗ ERROR: No difference detected during active period")
                        print(f"    Frames {i}-{i+1}, score: {result.score}")
                        passed = False
            
            if differences and passed:
                avg_diff = np.mean(differences)
                print(f"  ✓ Active period: avg difference = {avg_diff:.3f}")
        
        # Test differencing during idle period
        idle_period = next((p for p in timeline if not p['active']), None)
        if idle_period:
            start_frame = idle_period['start_frame']
            end_frame = min(start_frame + 3, idle_period['end_frame'])
            
            differences = []
            for i in range(start_frame, end_frame):
                if i + 1 < len(frames):
                    result = diff.compute_difference(frames[i], frames[i+1])
                    differences.append(result.score)
                    
                    if result.score > 0.05:  # Should be very small
                        print(f"  ✗ ERROR: Unexpected difference during idle period")
                        print(f"    Frames {i}-{i+1}, score: {result.score}")
                        passed = False
            
            if differences and passed:
                avg_diff = np.mean(differences)
                print(f"  ✓ Idle period: avg difference = {avg_diff:.3f} (near zero)")
        
        # Test with different frame types
        from videokurt.samplemaker import (
            create_blank_frame,
            create_gradient_frame,
            add_circle
        )
        
        frame1 = create_blank_frame((20, 20), channels=1)
        frame2 = add_circle(frame1, (10, 10), 5, color=(255,), filled=True)
        
        result = diff.compute_difference(frame1, frame2)
        
        if result.score <= 0:
            print(f"  ✗ ERROR: Differencer didn't detect added circle")
            passed = False
        else:
            print(f"  ✓ Differencer detects shape changes: score = {result.score:.3f}")
        
        # Test metadata in result
        if 'algorithm' not in result.metadata:
            print(f"  ✗ ERROR: Differencer result missing metadata")
            passed = False
        else:
            print(f"  ✓ Differencer provides metadata: {result.metadata['algorithm']}")
    
    else:
        # Basic compatibility test without differencer
        test_data = create_test_video_frames(size=(25, 25))
        frames = test_data['frames']
        
        # Frames should be compatible format (numpy arrays)
        for frame in frames[:3]:
            if not isinstance(frame, np.ndarray):
                print(f"  ✗ ERROR: Frames not in compatible format")
                passed = False
            if frame.dtype != np.uint8:
                print(f"  ✗ ERROR: Frames not uint8 dtype")
                passed = False
        
        if passed:
            print(f"  ✓ Frame format compatible with CV operations")
    
    return passed


def main():
    """Run all tests and report results."""
    print("=" * 60)
    print("Running: test_05_integration.py")
    print("=" * 60)
    
    tests = [
        ("test_complete_video_generation", test_complete_video_generation),
        ("test_event_sequence_ordering", test_event_sequence_ordering),
        ("test_fps_timing_calculation", test_fps_timing_calculation),
        ("test_data_structure_integrity", test_data_structure_integrity),
        ("test_cross_module_compatibility", test_cross_module_compatibility),
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