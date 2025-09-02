"""
Test 01: Frame Generation
Tests basic frame creation functionality of the SampleMaker module.
Critical because all other operations depend on correct frame generation.

# to run python -m videokurt.smoke_tests.samplemaker.test_01_frame_generation
"""

import sys
import numpy as np
from videokurt.samplemaker import (
    create_blank_frame,
    create_solid_frame,
    create_gradient_frame,
    create_checkerboard
)


def test_blank_frame_creation():
    """Test 1: Verify blank frames have correct dimensions and zero values."""
    print("\nTest 1: test_blank_frame_creation")
    print("  Creating blank frames with various configurations...")
    
    passed = True
    
    # Test grayscale blank frame
    frame_gray = create_blank_frame(size=(20, 20), channels=1)
    print(f"  Grayscale frame shape: {frame_gray.shape}")
    
    if frame_gray.shape != (20, 20):
        print(f"  ✗ ERROR: Expected shape (20, 20), got {frame_gray.shape}")
        passed = False
    else:
        print(f"  ✓ Grayscale shape correct: {frame_gray.shape}")
    
    if not np.all(frame_gray == 0):
        print(f"  ✗ ERROR: Frame not all zeros, max value: {np.max(frame_gray)}")
        passed = False
    else:
        print(f"  ✓ All grayscale values are zero: True")
    
    # Test BGR blank frame
    frame_bgr = create_blank_frame(size=(30, 40), channels=3)
    print(f"  BGR frame shape: {frame_bgr.shape}")
    
    if frame_bgr.shape != (30, 40, 3):
        print(f"  ✗ ERROR: Expected shape (30, 40, 3), got {frame_bgr.shape}")
        passed = False
    else:
        print(f"  ✓ BGR shape correct: {frame_bgr.shape}")
    
    if not np.all(frame_bgr == 0):
        print(f"  ✗ ERROR: Frame not all zeros, max value: {np.max(frame_bgr)}")
        passed = False
    else:
        print(f"  ✓ All BGR values are zero: True")
    
    # Test data type
    if frame_bgr.dtype != np.uint8:
        print(f"  ✗ ERROR: Expected dtype uint8, got {frame_bgr.dtype}")
        passed = False
    else:
        print(f"  ✓ Data type is uint8: True")
    
    return passed


def test_solid_frame_colors():
    """Test 2: Ensure solid frames maintain consistent color values."""
    print("\nTest 2: test_solid_frame_colors")
    print("  Creating solid frames with specific colors...")
    
    passed = True
    
    # Test grayscale solid frame
    gray_value = 128
    frame_gray = create_solid_frame(size=(15, 15), color=(gray_value,), channels=1)
    
    if not np.all(frame_gray == gray_value):
        print(f"  ✗ ERROR: Gray values not consistent, expected {gray_value}, got unique: {np.unique(frame_gray)}")
        passed = False
    else:
        print(f"  ✓ Grayscale solid frame uniform: all pixels = {gray_value}")
    
    # Test BGR solid frame
    test_color = (100, 150, 200)  # BGR
    frame_bgr = create_solid_frame(size=(25, 25), color=test_color, channels=3)
    
    for i, channel_name in enumerate(['B', 'G', 'R']):
        channel_values = frame_bgr[:, :, i]
        if not np.all(channel_values == test_color[i]):
            print(f"  ✗ ERROR: {channel_name} channel not uniform, expected {test_color[i]}, got unique: {np.unique(channel_values)}")
            passed = False
        else:
            print(f"  ✓ {channel_name} channel uniform: all pixels = {test_color[i]}")
    
    # Test frame size
    if frame_bgr.shape[:2] != (25, 25):
        print(f"  ✗ ERROR: Frame size incorrect, expected (25, 25), got {frame_bgr.shape[:2]}")
        passed = False
    else:
        print(f"  ✓ Frame dimensions correct: (25, 25)")
    
    return passed


def test_gradient_directions():
    """Test 3: Validate gradient generation in all directions."""
    print("\nTest 3: test_gradient_directions")
    print("  Testing gradient generation in horizontal, vertical, and diagonal directions...")
    
    passed = True
    size = (20, 30)
    
    # Test horizontal gradient
    frame_h = create_gradient_frame(size=size, direction='horizontal', start_val=0, end_val=255)
    
    # Check that values increase horizontally
    left_col = frame_h[:, 0]
    right_col = frame_h[:, -1]
    
    if not np.all(left_col == left_col[0]):  # All values in left column should be same
        print(f"  ✗ ERROR: Horizontal gradient left column not uniform")
        passed = False
    else:
        print(f"  ✓ Horizontal gradient: left column uniform (value={left_col[0]})")
    
    if not np.all(right_col == right_col[0]):  # All values in right column should be same
        print(f"  ✗ ERROR: Horizontal gradient right column not uniform")
        passed = False
    else:
        print(f"  ✓ Horizontal gradient: right column uniform (value={right_col[0]})")
    
    if left_col[0] >= right_col[0]:
        print(f"  ✗ ERROR: Horizontal gradient not increasing left to right")
        passed = False
    else:
        print(f"  ✓ Horizontal gradient increases: {left_col[0]} -> {right_col[0]}")
    
    # Test vertical gradient
    frame_v = create_gradient_frame(size=size, direction='vertical', start_val=0, end_val=255)
    
    top_row = frame_v[0, :]
    bottom_row = frame_v[-1, :]
    
    if not np.all(top_row == top_row[0]):
        print(f"  ✗ ERROR: Vertical gradient top row not uniform")
        passed = False
    else:
        print(f"  ✓ Vertical gradient: top row uniform (value={top_row[0]})")
    
    if not np.all(bottom_row == bottom_row[0]):
        print(f"  ✗ ERROR: Vertical gradient bottom row not uniform")
        passed = False
    else:
        print(f"  ✓ Vertical gradient: bottom row uniform (value={bottom_row[0]})")
    
    if top_row[0] >= bottom_row[0]:
        print(f"  ✗ ERROR: Vertical gradient not increasing top to bottom")
        passed = False
    else:
        print(f"  ✓ Vertical gradient increases: {top_row[0]} -> {bottom_row[0]}")
    
    # Test diagonal gradient
    frame_d = create_gradient_frame(size=size, direction='diagonal', start_val=0, end_val=255)
    
    # Diagonal should increase from top-left to bottom-right
    tl_val = frame_d[0, 0]
    br_val = frame_d[-1, -1]
    
    if tl_val >= br_val:
        print(f"  ✗ ERROR: Diagonal gradient not increasing from top-left to bottom-right")
        passed = False
    else:
        print(f"  ✓ Diagonal gradient increases: top-left({tl_val}) -> bottom-right({br_val})")
    
    return passed


def test_checkerboard_pattern():
    """Test 4: Confirm alternating pattern generation."""
    print("\nTest 4: test_checkerboard_pattern")
    print("  Creating checkerboard with 5x5 squares...")
    
    passed = True
    size = (20, 20)
    square_size = 5
    color1, color2 = 0, 255
    
    frame = create_checkerboard(size=size, square_size=square_size, color1=color1, color2=color2)
    
    # Check alternating pattern
    # Top-left square should be color1
    tl_square = frame[0:square_size, 0:square_size]
    if not np.all(tl_square == color1):
        print(f"  ✗ ERROR: Top-left square not uniform color1={color1}")
        passed = False
    else:
        print(f"  ✓ Top-left square is color1 ({color1})")
    
    # Square to the right should be color2
    tr_square = frame[0:square_size, square_size:2*square_size]
    if not np.all(tr_square == color2):
        print(f"  ✗ ERROR: Top-right square not uniform color2={color2}")
        passed = False
    else:
        print(f"  ✓ Adjacent square is color2 ({color2})")
    
    # Check total unique values
    unique_vals = np.unique(frame)
    if len(unique_vals) != 2:
        print(f"  ✗ ERROR: Expected 2 unique values, got {len(unique_vals)}: {unique_vals}")
        passed = False
    else:
        print(f"  ✓ Exactly 2 unique values: {unique_vals}")
    
    # Verify checkerboard size
    if frame.shape != size:
        print(f"  ✗ ERROR: Frame size incorrect, expected {size}, got {frame.shape}")
        passed = False
    else:
        print(f"  ✓ Frame size correct: {size}")
    
    # Count squares of each color (should be equal for even grid)
    color1_pixels = np.sum(frame == color1)
    color2_pixels = np.sum(frame == color2)
    total_pixels = size[0] * size[1]
    
    if color1_pixels + color2_pixels != total_pixels:
        print(f"  ✗ ERROR: Pixel count mismatch")
        passed = False
    else:
        print(f"  ✓ Pixel counts: color1={color1_pixels}, color2={color2_pixels}, total={total_pixels}")
    
    return passed


def test_frame_dimensions():
    """Test 5: Test various frame sizes and channel configurations."""
    print("\nTest 5: test_frame_dimensions")
    print("  Testing various frame dimensions and configurations...")
    
    passed = True
    
    test_cases = [
        ((10, 10), 1, "Small grayscale"),
        ((100, 50), 3, "Wide BGR"),
        ((50, 100), 3, "Tall BGR"),
        ((1, 1), 1, "Single pixel grayscale"),
        ((1, 1), 3, "Single pixel BGR"),
        ((1000, 1000), 1, "Large grayscale"),
    ]
    
    for size, channels, description in test_cases:
        frame = create_blank_frame(size=size, channels=channels)
        
        expected_shape = size if channels == 1 else (*size, channels)
        
        if frame.shape != expected_shape:
            print(f"  ✗ ERROR: {description} - Expected shape {expected_shape}, got {frame.shape}")
            passed = False
        else:
            print(f"  ✓ {description}: shape {frame.shape} correct")
        
        # Verify memory layout
        if channels == 3:
            if len(frame.shape) != 3:
                print(f"  ✗ ERROR: {description} - BGR frame should be 3D")
                passed = False
        else:
            if len(frame.shape) != 2:
                print(f"  ✗ ERROR: {description} - Grayscale frame should be 2D")
                passed = False
    
    return passed


def main():
    """Run all tests and report results."""
    print("=" * 60)
    print("Running: test_01_frame_generation.py")
    print("=" * 60)
    
    tests = [
        ("test_blank_frame_creation", test_blank_frame_creation),
        ("test_solid_frame_colors", test_solid_frame_colors),
        ("test_gradient_directions", test_gradient_directions),
        ("test_checkerboard_pattern", test_checkerboard_pattern),
        ("test_frame_dimensions", test_frame_dimensions),
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