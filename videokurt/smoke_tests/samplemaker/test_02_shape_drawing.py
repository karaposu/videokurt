"""
Test 02: Shape Drawing
Tests shape rendering and positioning functionality.
Critical for UI element simulation (popups, buttons, text regions).

# to run python -m videokurt.smoke_tests.samplemaker.test_02_shape_drawing


"""

import sys
import numpy as np
from videokurt.samplemaker import (
    create_blank_frame,
    create_solid_frame,
    add_rectangle,
    add_circle,
    add_text_region
)


def test_rectangle_drawing():
    """Test 1: Validate filled and outlined rectangles."""
    print("\nTest 1: test_rectangle_drawing")
    print("  Drawing rectangles on frames...")
    
    passed = True
    
    # Test filled rectangle
    frame = create_blank_frame(size=(30, 30), channels=1)
    rect_color = 200
    rect_pos = (5, 10)  # x, y
    rect_size = (15, 10)  # width, height
    
    frame_with_rect = add_rectangle(frame, rect_pos, rect_size, color=(rect_color,), filled=True)
    
    # Check rectangle area
    rect_area = frame_with_rect[10:20, 5:20]  # y:y+h, x:x+w
    
    if not np.all(rect_area == rect_color):
        print(f"  ✗ ERROR: Filled rectangle area not uniform, expected {rect_color}")
        print(f"    Unique values in rectangle: {np.unique(rect_area)}")
        passed = False
    else:
        print(f"  ✓ Filled rectangle uniform: all pixels = {rect_color}")
    
    # Check outside rectangle is still black
    outside_sum = np.sum(frame_with_rect) - np.sum(rect_area)
    if outside_sum != 0:
        print(f"  ✗ ERROR: Pixels outside rectangle modified (sum={outside_sum})")
        passed = False
    else:
        print(f"  ✓ Area outside rectangle unchanged")
    
    # Test outlined rectangle
    frame2 = create_blank_frame(size=(30, 30), channels=1)
    frame_with_outline = add_rectangle(frame2, rect_pos, rect_size, color=(rect_color,), filled=False)
    
    # Check that interior is still black
    interior = frame_with_outline[11:19, 6:19]  # Inside the outline
    if np.sum(interior) != 0:
        print(f"  ✗ ERROR: Rectangle outline has filled interior (sum={np.sum(interior)})")
        passed = False
    else:
        print(f"  ✓ Rectangle outline: interior is empty")
    
    # Check that outline exists
    top_edge = frame_with_outline[10, 5:20]
    if not np.all(top_edge == rect_color):
        print(f"  ✗ ERROR: Rectangle outline top edge not drawn correctly")
        passed = False
    else:
        print(f"  ✓ Rectangle outline: edges drawn correctly")
    
    # Test BGR rectangle
    frame_bgr = create_blank_frame(size=(30, 30), channels=3)
    bgr_color = (100, 150, 200)
    frame_bgr_rect = add_rectangle(frame_bgr, rect_pos, rect_size, color=bgr_color, filled=True)
    
    rect_area_bgr = frame_bgr_rect[10:20, 5:20]
    for i, channel in enumerate(['B', 'G', 'R']):
        if not np.all(rect_area_bgr[:, :, i] == bgr_color[i]):
            print(f"  ✗ ERROR: BGR rectangle {channel} channel not uniform")
            passed = False
    
    if passed:
        print(f"  ✓ BGR rectangle: color correct {bgr_color}")
    
    return passed


def test_circle_rendering():
    """Test 2: Ensure circles are drawn with correct radius."""
    print("\nTest 2: test_circle_rendering")
    print("  Drawing circles with various radii...")
    
    passed = True
    
    # Test filled circle
    frame = create_blank_frame(size=(40, 40), channels=1)
    center = (20, 20)
    radius = 10
    circle_color = 180
    
    frame_with_circle = add_circle(frame, center, radius, color=(circle_color,), filled=True)
    
    # Check center point
    if frame_with_circle[center[1], center[0]] != circle_color:
        print(f"  ✗ ERROR: Circle center not filled")
        passed = False
    else:
        print(f"  ✓ Circle center filled correctly")
    
    # Check points at radius distance
    test_points = [
        (center[0] + radius, center[1]),  # Right
        (center[0] - radius, center[1]),  # Left
        (center[0], center[1] + radius),  # Bottom
        (center[0], center[1] - radius),  # Top
    ]
    
    for x, y in test_points:
        if 0 <= x < 40 and 0 <= y < 40:
            if frame_with_circle[y, x] != circle_color:
                print(f"  ✗ ERROR: Point at radius distance ({x},{y}) not filled")
                passed = False
    
    if passed:
        print(f"  ✓ Circle radius points filled correctly")
    
    # Check point outside radius
    outside_point = (center[0] + radius + 2, center[1])
    if frame_with_circle[outside_point[1], outside_point[0]] != 0:
        print(f"  ✗ ERROR: Point outside radius is filled")
        passed = False
    else:
        print(f"  ✓ Points outside radius unchanged")
    
    # Test circle outline
    frame2 = create_blank_frame(size=(40, 40), channels=1)
    frame_with_outline = add_circle(frame2, center, radius, color=(circle_color,), filled=False)
    
    # Center should be empty for outline
    if frame_with_outline[center[1], center[0]] != 0:
        print(f"  ✗ ERROR: Circle outline has filled center")
        passed = False
    else:
        print(f"  ✓ Circle outline: center is empty")
    
    # Edge points should be filled
    edge_filled = False
    for x, y in test_points:
        if 0 <= x < 40 and 0 <= y < 40:
            if frame_with_outline[y, x] == circle_color:
                edge_filled = True
                break
    
    if not edge_filled:
        print(f"  ✗ ERROR: Circle outline edges not drawn")
        passed = False
    else:
        print(f"  ✓ Circle outline: edges drawn")
    
    # Test small circle (radius=1)
    frame3 = create_blank_frame(size=(10, 10), channels=1)
    small_circle = add_circle(frame3, (5, 5), 1, color=(255,), filled=True)
    
    # Should be a small cross pattern
    if small_circle[5, 5] != 255:
        print(f"  ✗ ERROR: Small circle center not filled")
        passed = False
    else:
        print(f"  ✓ Small circle (r=1) drawn correctly")
    
    return passed


def test_text_region_simulation():
    """Test 3: Verify text line generation."""
    print("\nTest 3: test_text_region_simulation")
    print("  Creating simulated text regions...")
    
    passed = True
    
    frame = create_blank_frame(size=(50, 50), channels=1)
    text_pos = (10, 10)
    text_size = (30, 20)
    text_color = 200
    bg_color = 50
    
    frame_with_text = add_text_region(frame, text_pos, text_size, text_color, bg_color)
    
    # Check background color in text region
    text_region = frame_with_text[10:30, 10:40]
    
    # Background should be present
    bg_pixels = np.sum(text_region == bg_color)
    if bg_pixels == 0:
        print(f"  ✗ ERROR: No background color in text region")
        passed = False
    else:
        print(f"  ✓ Text region has background (color={bg_color})")
    
    # Text lines should be present
    text_pixels = np.sum(text_region == text_color)
    if text_pixels == 0:
        print(f"  ✗ ERROR: No text lines in text region")
        passed = False
    else:
        print(f"  ✓ Text lines present (color={text_color}, pixels={text_pixels})")
    
    # Should have horizontal lines (simulated text)
    has_lines = False
    for y in range(text_region.shape[0]):
        row = text_region[y, :]
        if np.any(row == text_color) and np.sum(row == text_color) > 5:
            has_lines = True
            break
    
    if not has_lines:
        print(f"  ✗ ERROR: No horizontal text lines detected")
        passed = False
    else:
        print(f"  ✓ Horizontal text lines detected")
    
    # Check area outside text region is unchanged
    outside_text = frame_with_text.copy()
    outside_text[10:30, 10:40] = 0  # Zero out text region
    
    if np.sum(outside_text) != 0:
        print(f"  ✗ ERROR: Area outside text region modified")
        passed = False
    else:
        print(f"  ✓ Area outside text region unchanged")
    
    # Check that text doesn't overflow bounds
    if np.any(frame_with_text[0:10, :] == text_color):
        print(f"  ✗ ERROR: Text overflowed top boundary")
        passed = False
    if np.any(frame_with_text[30:, :] == text_color):
        print(f"  ✗ ERROR: Text overflowed bottom boundary")
        passed = False
    if np.any(frame_with_text[:, 0:10] == text_color):
        print(f"  ✗ ERROR: Text overflowed left boundary")
        passed = False
    if np.any(frame_with_text[:, 40:] == text_color):
        print(f"  ✗ ERROR: Text overflowed right boundary")
        passed = False
    
    if passed:
        print(f"  ✓ Text contained within specified bounds")
    
    return passed


def test_shape_boundaries():
    """Test 4: Test edge cases near frame boundaries."""
    print("\nTest 4: test_shape_boundaries")
    print("  Testing shapes at frame edges...")
    
    passed = True
    
    frame_size = (20, 20)
    
    # Test rectangle partially outside frame
    frame = create_blank_frame(size=frame_size, channels=1)
    
    # Rectangle that extends beyond right edge
    rect_pos = (15, 5)
    rect_size = (10, 5)  # Would extend to x=25, but frame width is 20
    
    frame_with_rect = add_rectangle(frame, rect_pos, rect_size, color=(255,), filled=True)
    
    # Should be clipped at x=20
    clipped_area = frame_with_rect[5:10, 15:20]
    if not np.all(clipped_area == 255):
        print(f"  ✗ ERROR: Rectangle not drawn up to edge")
        passed = False
    else:
        print(f"  ✓ Rectangle clipped correctly at right edge")
    
    # Should not wrap around or crash
    if np.any(frame_with_rect[:, 0:15] == 255):
        print(f"  ✗ ERROR: Rectangle wrapped around or drew outside bounds")
        passed = False
    else:
        print(f"  ✓ No wraparound or overflow")
    
    # Test circle at corner
    frame2 = create_blank_frame(size=frame_size, channels=1)
    corner_circle = add_circle(frame2, (0, 0), 5, color=(200,), filled=True)
    
    # Should draw partial circle in top-left
    if corner_circle[0, 0] != 200:
        print(f"  ✗ ERROR: Corner circle not drawn at (0,0)")
        passed = False
    else:
        print(f"  ✓ Circle at corner drawn correctly")
    
    # Test negative position (should be handled gracefully)
    frame3 = create_blank_frame(size=frame_size, channels=1)
    neg_rect = add_rectangle(frame3, (-5, -5), (10, 10), color=(150,), filled=True)
    
    # Should draw partial rectangle from (0,0) to (5,5)
    visible_area = neg_rect[0:5, 0:5]
    if not np.all(visible_area == 150):
        print(f"  ✗ ERROR: Negative position rectangle not handled correctly")
        passed = False
    else:
        print(f"  ✓ Negative position handled gracefully")
    
    # Test shape larger than frame
    frame4 = create_blank_frame(size=(10, 10), channels=1)
    large_circle = add_circle(frame4, (5, 5), 20, color=(100,), filled=True)
    
    # Entire frame should be filled
    if not np.all(large_circle == 100):
        print(f"  ✗ ERROR: Large circle doesn't fill entire frame")
        print(f"    Unique values: {np.unique(large_circle)}")
        passed = False
    else:
        print(f"  ✓ Shape larger than frame handled correctly")
    
    return passed


def test_shape_overlays():
    """Test 5: Confirm shapes properly overlay on existing frames."""
    print("\nTest 5: test_shape_overlays")
    print("  Testing shape overlays and layering...")
    
    passed = True
    
    # Start with gradient background
    from videokurt.samplemaker import create_gradient_frame
    base_frame = create_gradient_frame(size=(30, 30), direction='horizontal')
    
    # Add rectangle overlay
    rect_frame = add_rectangle(base_frame, (5, 5), (10, 10), color=(255,), filled=True)
    
    # Rectangle area should be solid white
    rect_area = rect_frame[5:15, 5:15]
    if not np.all(rect_area == 255):
        print(f"  ✗ ERROR: Rectangle didn't overlay properly")
        passed = False
    else:
        print(f"  ✓ Rectangle overlaid on gradient")
    
    # Area outside should still have gradient
    top_strip = rect_frame[0, :]
    if np.all(top_strip == top_strip[0]):  # Should NOT be uniform
        print(f"  ✗ ERROR: Gradient destroyed outside rectangle")
        passed = False
    else:
        print(f"  ✓ Gradient preserved outside rectangle")
    
    # Test multiple overlapping shapes
    frame2 = create_solid_frame(size=(40, 40), color=(50,), channels=1)
    
    # Add circle first
    frame2 = add_circle(frame2, (20, 20), 10, color=(100,), filled=True)
    
    # Add rectangle overlapping circle
    frame2 = add_rectangle(frame2, (15, 15), (15, 15), color=(200,), filled=True)
    
    # Rectangle should overwrite circle where they overlap
    rect_center = frame2[20, 20]
    if rect_center != 200:
        print(f"  ✗ ERROR: Later shape didn't overwrite earlier shape")
        passed = False
    else:
        print(f"  ✓ Shape layering correct (later overwrites earlier)")
    
    # Test outline over filled shape
    frame3 = create_blank_frame(size=(30, 30), channels=1)
    frame3 = add_circle(frame3, (15, 15), 8, color=(100,), filled=True)
    frame3 = add_rectangle(frame3, (10, 10), (10, 10), color=(255,), filled=False)
    
    # Check that outline is visible over filled circle
    outline_pixel = frame3[10, 15]  # Top edge of rectangle
    if outline_pixel != 255:
        print(f"  ✗ ERROR: Outline not visible over filled shape")
        passed = False
    else:
        print(f"  ✓ Outline visible over filled shape")
    
    # Interior should still show circle
    interior_pixel = frame3[15, 15]  # Center
    if interior_pixel != 100:
        print(f"  ✗ ERROR: Outline rectangle filled the interior")
        passed = False
    else:
        print(f"  ✓ Outline preserves interior content")
    
    return passed


def main():
    """Run all tests and report results."""
    print("=" * 60)
    print("Running: test_02_shape_drawing.py")
    print("=" * 60)
    
    tests = [
        ("test_rectangle_drawing", test_rectangle_drawing),
        ("test_circle_rendering", test_circle_rendering),
        ("test_text_region_simulation", test_text_region_simulation),
        ("test_shape_boundaries", test_shape_boundaries),
        ("test_shape_overlays", test_shape_overlays),
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