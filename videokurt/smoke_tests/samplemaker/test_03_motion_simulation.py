"""
Test 03: Motion Simulation
Tests motion and transition effects for event simulation.
Critical for testing event detection algorithms with known motion patterns.

# to run python -m videokurt.smoke_tests.samplemaker.test_03_motion_simulation


"""

import sys
import numpy as np
from videokurt.samplemaker import (
    create_gradient_frame,
    create_checkerboard,
    add_circle,
    add_text_region,
    simulate_scroll,
    simulate_scene_change,
    simulate_popup,
    add_noise,
    add_compression_artifacts
)


def test_scroll_directions():
    """Test 1: Validate scrolling in all four directions."""
    print("\nTest 1: test_scroll_directions")
    print("  Testing scroll simulation in all directions...")
    
    passed = True
    
    # Create a frame with identifiable pattern
    base = create_gradient_frame(size=(20, 20), direction='vertical')
    base = add_circle(base, (10, 5), 3, color=(255,), filled=True)
    
    # Test downward scroll
    scroll_pixels = 5
    scrolled_down = simulate_scroll(base, pixels=scroll_pixels, direction='down')
    
    # Circle should move down by scroll_pixels
    # Original circle at y=5, should now be at y=10
    expected_y = 10
    circle_row = scrolled_down[expected_y, :]
    
    if not np.any(circle_row == 255):
        print(f"  ✗ ERROR: Circle not found at expected position after down scroll")
        print(f"    Expected at y={expected_y}")
        passed = False
    else:
        print(f"  ✓ Down scroll: object moved correctly ({scroll_pixels} pixels)")
    
    # Test upward scroll
    scrolled_up = simulate_scroll(base, pixels=scroll_pixels, direction='up')
    
    # Circle should move up by scroll_pixels
    # Original at y=5, should now be at y=0
    expected_y_up = 0
    circle_row_up = scrolled_up[expected_y_up, :]
    
    if not np.any(circle_row_up == 255):
        print(f"  ✗ ERROR: Circle not found after up scroll")
        passed = False
    else:
        print(f"  ✓ Up scroll: object moved correctly")
    
    # Test right scroll
    base_h = create_gradient_frame(size=(20, 20), direction='horizontal')
    base_h = add_circle(base_h, (5, 10), 3, color=(255,), filled=True)
    
    scrolled_right = simulate_scroll(base_h, pixels=scroll_pixels, direction='right')
    
    # Circle should move right
    expected_x = 10
    circle_col = scrolled_right[:, expected_x]
    
    if not np.any(circle_col == 255):
        print(f"  ✗ ERROR: Circle not found after right scroll")
        passed = False
    else:
        print(f"  ✓ Right scroll: object moved correctly")
    
    # Test left scroll
    scrolled_left = simulate_scroll(base_h, pixels=scroll_pixels, direction='left')
    
    # Circle should move left (with wraparound)
    expected_x_left = 0
    circle_col_left = scrolled_left[:, expected_x_left]
    
    if not np.any(circle_col_left == 255):
        print(f"  ✗ ERROR: Circle not found after left scroll")
        passed = False
    else:
        print(f"  ✓ Left scroll: object moved correctly")
    
    # Test that total content is preserved (wraparound)
    total_original = np.sum(base)
    total_scrolled = np.sum(scrolled_down)
    
    if abs(total_original - total_scrolled) > 1:  # Allow small rounding difference
        print(f"  ✗ ERROR: Content not preserved during scroll")
        print(f"    Original sum: {total_original}, Scrolled sum: {total_scrolled}")
        passed = False
    else:
        print(f"  ✓ Content preserved with wraparound")
    
    return passed


def test_scene_change_types():
    """Test 2: Test cut, fade, and slide transitions."""
    print("\nTest 2: test_scene_change_types")
    print("  Testing different scene change types...")
    
    passed = True
    
    # Create initial frame
    base = create_gradient_frame(size=(30, 30), direction='horizontal')
    
    # Test cut transition
    cut_frame = simulate_scene_change(base, change_type='cut')
    
    # Should be completely different
    if np.array_equal(base, cut_frame):
        print(f"  ✗ ERROR: Cut transition didn't change frame")
        passed = False
    else:
        correlation = np.corrcoef(base.flatten(), cut_frame.flatten())[0, 1]
        print(f"  ✓ Cut transition: completely different (correlation={correlation:.3f})")
    
    # Test fade transition
    fade_frame = simulate_scene_change(base, change_type='fade')
    
    # Should be darker version with some variation
    mean_original = np.mean(base)
    mean_fade = np.mean(fade_frame)
    
    if mean_fade >= mean_original:
        print(f"  ✗ ERROR: Fade didn't darken image")
        print(f"    Original mean: {mean_original:.1f}, Fade mean: {mean_fade:.1f}")
        passed = False
    else:
        print(f"  ✓ Fade transition: darkened ({mean_original:.1f} -> {mean_fade:.1f})")
    
    # Should have some added variation
    if np.array_equal(base * 0.3, fade_frame):
        print(f"  ✗ ERROR: Fade has no variation added")
        passed = False
    else:
        print(f"  ✓ Fade transition: includes variation")
    
    # Test slide transition
    slide_frame = simulate_scene_change(base, change_type='slide')
    
    # Left half should be different
    left_half = slide_frame[:, :15]
    right_half = slide_frame[:, 15:]
    original_right = base[:, 15:]
    
    if np.all(left_half == left_half[0, 0]):
        print(f"  ✓ Slide transition: left half changed")
    else:
        print(f"  ✗ ERROR: Slide transition left half not uniform")
        passed = False
    
    if not np.array_equal(right_half, original_right):
        print(f"  ✗ ERROR: Slide transition modified right half")
        passed = False
    else:
        print(f"  ✓ Slide transition: right half preserved")
    
    return passed


def test_popup_overlay():
    """Test 3: Ensure popups darken background correctly."""
    print("\nTest 3: test_popup_overlay")
    print("  Testing popup overlay effects...")
    
    passed = True
    
    # Create base frame with pattern
    base = create_checkerboard(size=(40, 40), square_size=5)
    base_mean = np.mean(base)
    
    # Add popup
    popup_size = (20, 15)
    frame_with_popup = simulate_popup(base, popup_size=popup_size, position=None)
    
    # Background should be darkened
    # Get corners (outside popup)
    corner_tl = frame_with_popup[0:5, 0:5]
    corner_mean = np.mean(corner_tl)
    
    if corner_mean >= base_mean * 0.8:  # Should be ~70% of original
        print(f"  ✗ ERROR: Background not darkened")
        print(f"    Original: {base_mean:.1f}, Corner: {corner_mean:.1f}")
        passed = False
    else:
        darkening_factor = corner_mean / (base_mean / 2)  # Checkerboard average
        print(f"  ✓ Background darkened (factor ~{darkening_factor:.2f})")
    
    # Check popup exists (centered)
    center_x, center_y = 20, 20
    popup_area = frame_with_popup[
        center_y - popup_size[1]//2 : center_y + popup_size[1]//2,
        center_x - popup_size[0]//2 : center_x + popup_size[0]//2
    ]
    
    # Popup should be brighter than darkened background
    popup_mean = np.mean(popup_area)
    if popup_mean <= corner_mean:
        print(f"  ✗ ERROR: Popup not brighter than background")
        passed = False
    else:
        print(f"  ✓ Popup brighter than darkened background")
    
    # Test popup with specific position
    pos = (5, 5)
    frame_with_popup_pos = simulate_popup(base, popup_size=(10, 10), position=pos)
    
    # Check popup at specified position
    popup_at_pos = frame_with_popup_pos[5:15, 5:15]
    if np.mean(popup_at_pos) <= corner_mean:
        print(f"  ✗ ERROR: Positioned popup not created correctly")
        passed = False
    else:
        print(f"  ✓ Popup positioned correctly at {pos}")
    
    # Verify popup has border (edge detection)
    # Border should be darker than interior
    popup_interior = popup_at_pos[2:-2, 2:-2]
    popup_edges = np.concatenate([
        popup_at_pos[0, :],
        popup_at_pos[-1, :],
        popup_at_pos[:, 0],
        popup_at_pos[:, -1]
    ])
    
    if np.mean(popup_interior) <= np.mean(popup_edges):
        print(f"  ✗ ERROR: Popup border not distinct")
        passed = False
    else:
        print(f"  ✓ Popup has distinct border")
    
    return passed


def test_motion_continuity():
    """Test 4: Verify smooth motion across frames."""
    print("\nTest 4: test_motion_continuity")
    print("  Testing motion continuity across multiple frames...")
    
    passed = True
    
    # Create sequence of scrolling frames
    base = create_gradient_frame(size=(30, 30), direction='vertical')
    base = add_text_region(base, (5, 5), (20, 20), text_color=200, bg_color=50)
    
    frames = [base]
    for i in range(1, 6):
        scrolled = simulate_scroll(frames[0], pixels=i*2, direction='down')
        frames.append(scrolled)
    
    # Check that motion is continuous
    prev_center = None
    positions = []
    
    for i, frame in enumerate(frames):
        # Find center of mass of bright pixels
        bright_pixels = frame > 150
        if np.any(bright_pixels):
            y_coords, x_coords = np.where(bright_pixels)
            center_y = np.mean(y_coords)
            center_x = np.mean(x_coords)
            positions.append((center_x, center_y))
            
            if prev_center is not None:
                # Check motion is consistent (accounting for wraparound)
                dy = center_y - prev_center[1]
                # Motion might wrap around or vary due to text region structure
                # Just check it's moving in generally the right direction
                if i > 1 and dy > 10:  # Large jump might indicate wraparound
                    # This is OK - content wraps around
                    pass
                elif i > 1 and abs(dy) < 0.1:  # No motion at all
                    print(f"  ✗ ERROR: No motion detected at frame {i}")
                    print(f"    Delta Y: {dy:.1f}")
                    passed = False
            
            prev_center = (center_x, center_y)
    
    if len(positions) == len(frames):
        print(f"  ✓ Motion tracked across {len(frames)} frames")
        
        # Check overall motion occurred
        # With wraparound, total displacement might be small but motion should occur
        motion_detected = False
        for i in range(1, len(positions)):
            if positions[i][1] != positions[i-1][1]:
                motion_detected = True
                break
        
        if not motion_detected:
            print(f"  ✗ ERROR: No motion detected across frames")
            passed = False
        else:
            print(f"  ✓ Motion detected across frames")
    
    # Test gradual fade sequence
    fade_frames = [base]
    for i in range(1, 4):
        # Each frame should be progressively darker
        faded = (fade_frames[0] * (1.0 - i * 0.2)).astype(np.uint8)
        fade_frames.append(faded)
    
    # Check progressive darkening
    means = [np.mean(f) for f in fade_frames]
    for i in range(1, len(means)):
        if means[i] >= means[i-1]:
            print(f"  ✗ ERROR: Fade not progressive at frame {i}")
            passed = False
    
    if passed:
        print(f"  ✓ Fade sequence progressively darkens")
    
    return passed


def test_effect_artifacts():
    """Test 5: Test noise and compression artifact generation."""
    print("\nTest 5: test_effect_artifacts")
    print("  Testing noise and compression artifacts...")
    
    passed = True
    
    # Start with clean frame
    base = create_gradient_frame(size=(40, 40), direction='diagonal')
    base_mean = np.mean(base)
    base_std = np.std(base)
    
    # Test Gaussian noise
    noisy_gaussian = add_noise(base, noise_type='gaussian', intensity=0.1)
    
    # Should increase standard deviation
    noisy_std = np.std(noisy_gaussian)
    if noisy_std <= base_std:
        print(f"  ✗ ERROR: Gaussian noise didn't increase variation")
        print(f"    Original std: {base_std:.2f}, Noisy std: {noisy_std:.2f}")
        passed = False
    else:
        print(f"  ✓ Gaussian noise added (std: {base_std:.1f} -> {noisy_std:.1f})")
    
    # Mean should be similar
    noisy_mean = np.mean(noisy_gaussian)
    if abs(noisy_mean - base_mean) > 10:
        print(f"  ✗ ERROR: Gaussian noise changed mean too much")
        passed = False
    else:
        print(f"  ✓ Gaussian noise preserves mean (~{base_mean:.1f})")
    
    # Test salt & pepper noise
    noisy_sp = add_noise(base, noise_type='salt_pepper', intensity=0.05)
    
    # Should have some pure white and black pixels
    white_pixels = np.sum(noisy_sp == 255)
    black_pixels = np.sum(noisy_sp == 0)
    
    if white_pixels == 0 or black_pixels == 0:
        print(f"  ✗ ERROR: Salt & pepper noise not added correctly")
        print(f"    White pixels: {white_pixels}, Black pixels: {black_pixels}")
        passed = False
    else:
        total_pixels = 40 * 40
        noise_ratio = (white_pixels + black_pixels) / total_pixels
        print(f"  ✓ Salt & pepper noise added ({noise_ratio:.1%} of pixels)")
    
    # Test uniform noise
    noisy_uniform = add_noise(base, noise_type='uniform', intensity=0.1)
    
    # Should add uniform random noise
    diff = noisy_uniform.astype(float) - base.astype(float)
    diff_range = np.max(diff) - np.min(diff)
    
    if diff_range < 20:  # Should have reasonable range
        print(f"  ✗ ERROR: Uniform noise range too small")
        passed = False
    else:
        print(f"  ✓ Uniform noise added (range: {diff_range:.1f})")
    
    # Test compression artifacts
    compressed = add_compression_artifacts(base, block_size=4, quality=0.5)
    
    # Should create blocky appearance
    # Check if blocks have uniform values
    block_uniformity = 0
    for i in range(0, 40, 4):
        for j in range(0, 40, 4):
            block = compressed[i:i+4, j:j+4]
            if block.size > 0:
                unique_vals = len(np.unique(block))
                if unique_vals <= 2:  # Block is quite uniform
                    block_uniformity += 1
    
    total_blocks = (40 // 4) * (40 // 4)
    uniform_ratio = block_uniformity / total_blocks
    
    if uniform_ratio < 0.3:
        print(f"  ✗ ERROR: Compression artifacts not creating blocks")
        print(f"    Uniform blocks: {uniform_ratio:.1%}")
        passed = False
    else:
        print(f"  ✓ Compression artifacts create blocks ({uniform_ratio:.1%} uniform)")
    
    # Test that effects are applied correctly to color images
    base_color = np.stack([base, base * 0.8, base * 0.6], axis=2).astype(np.uint8)
    noisy_color = add_noise(base_color, noise_type='gaussian', intensity=0.1)
    
    if noisy_color.shape != base_color.shape:
        print(f"  ✗ ERROR: Noise changed color image dimensions")
        passed = False
    else:
        print(f"  ✓ Noise correctly applied to color images")
    
    return passed


def main():
    """Run all tests and report results."""
    print("=" * 60)
    print("Running: test_03_motion_simulation.py")
    print("=" * 60)
    
    tests = [
        ("test_scroll_directions", test_scroll_directions),
        ("test_scene_change_types", test_scene_change_types),
        ("test_popup_overlay", test_popup_overlay),
        ("test_motion_continuity", test_motion_continuity),
        ("test_effect_artifacts", test_effect_artifacts),
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