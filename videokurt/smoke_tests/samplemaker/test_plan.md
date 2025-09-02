# SampleMaker Smoke Test Plan

## Overview
Comprehensive smoke tests to validate the SampleMaker module without external testing frameworks.
Each test provides verbose output to clearly identify any failures.

## Test Files Structure

### test_01_frame_generation.py
**Focus:** Basic frame creation and properties
**Why Critical:** Foundation for all other operations - if frames aren't created correctly, nothing else works

**Test Cases:**
1. `test_blank_frame_creation` - Verifies blank frames have correct dimensions and zero values
2. `test_solid_frame_colors` - Ensures solid frames maintain consistent color values
3. `test_gradient_directions` - Validates gradient generation in all directions
4. `test_checkerboard_pattern` - Confirms alternating pattern generation
5. `test_frame_dimensions` - Tests various frame sizes and channel configurations

---

### test_02_shape_drawing.py
**Focus:** Shape rendering and positioning
**Why Critical:** UI element simulation depends on accurate shape drawing for popups, buttons, text regions

**Test Cases:**
1. `test_rectangle_drawing` - Validates filled and outlined rectangles
2. `test_circle_rendering` - Ensures circles are drawn with correct radius
3. `test_text_region_simulation` - Verifies text line generation
4. `test_shape_boundaries` - Tests edge cases near frame boundaries
5. `test_shape_overlays` - Confirms shapes properly overlay on existing frames

---

### test_03_motion_simulation.py
**Focus:** Motion and transition effects
**Why Critical:** Event detection algorithms rely on accurate motion simulation for testing

**Test Cases:**
1. `test_scroll_directions` - Validates scrolling in all four directions
2. `test_scene_change_types` - Tests cut, fade, and slide transitions
3. `test_popup_overlay` - Ensures popups darken background correctly
4. `test_motion_continuity` - Verifies smooth motion across frames
5. `test_effect_artifacts` - Tests noise and compression artifact generation

---

### test_04_ground_truth.py
**Focus:** Ground truth data accuracy and completeness
**Why Critical:** Detection accuracy measurement depends on precise ground truth

**Test Cases:**
1. `test_event_timing_accuracy` - Validates event start/end times and frame indices
2. `test_activity_timeline_consistency` - Ensures binary timeline matches events
3. `test_frame_annotations` - Verifies frame-by-frame annotations
4. `test_metadata_completeness` - Checks event metadata contains required fields
5. `test_timeline_coverage` - Ensures no gaps or overlaps in timeline

---

### test_05_integration.py
**Focus:** Complete sequence generation and data consistency
**Why Critical:** Real-world usage involves complete sequences with multiple events

**Test Cases:**
1. `test_complete_video_generation` - Validates full test video creation
2. `test_event_sequence_ordering` - Ensures events occur in correct order
3. `test_fps_timing_calculation` - Verifies frame rate affects timing correctly
4. `test_data_structure_integrity` - Confirms all output fields present and valid
5. `test_cross_module_compatibility` - Tests integration with frame differencing module

## Running the Tests

```bash
# Run all tests
python -m videokurt.smoke_tests.samplemaker.test_01_frame_generation
python -m videokurt.smoke_tests.samplemaker.test_02_shape_drawing
python -m videokurt.smoke_tests.samplemaker.test_03_motion_simulation
python -m videokurt.smoke_tests.samplemaker.test_04_ground_truth
python -m videokurt.smoke_tests.samplemaker.test_05_integration

# Or run the test runner
python -m videokurt.smoke_tests.samplemaker.run_all_tests
```

## Expected Output Format

Each test will output:
```
========================================
Running: test_01_frame_generation.py
========================================

Test 1: test_blank_frame_creation
  Creating blank frame (20x20, 3 channels)...
  ✓ Frame shape correct: (20, 20, 3)
  ✓ All values are zero: True
  ✓ Data type is uint8: True
  PASSED

Test 2: test_solid_frame_colors
  Creating solid frame with color (100, 150, 200)...
  ✓ Frame shape correct: (20, 20, 3)
  ✓ Color values match: B=100, G=150, R=200
  PASSED

[... continues for all tests ...]

========================================
SUMMARY: 5/5 tests passed
========================================
```