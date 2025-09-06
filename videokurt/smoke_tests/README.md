# VideoKurt Smoke Tests

This directory contains smoke test scripts for VideoKurt features.

## Structure

- `feat_test_*.py` - Individual feature test scripts
- `raw_analysis/` - Raw analysis test scripts
- `pure_analysis/` - Pure analysis test scripts  
- `samplemaker/` - Sample video generation scripts

## Running Tests

Each test script can be run individually:

```bash
python videokurt/smoke_tests/feat_test_scrolling.py
python videokurt/smoke_tests/feat_test_motion_magnitude.py
# etc...
```

## Test Categories

### High-Value Features (Screen Recordings)
- `feat_test_scrolling.py` - Scrolling pattern detection
- `feat_test_stability_score.py` - Frame stability measurement
- `feat_test_scene_detection.py` - Scene boundary detection
- `feat_test_dwell_time.py` - Static region heatmaps
- `feat_test_edge_density.py` - Text vs media detection
- `feat_test_spatial_occupancy.py` - Activity distribution

### Motion Analysis
- `feat_test_motion_magnitude.py` - Overall motion intensity
- `feat_test_motion_trajectories.py` - Motion path tracking
- `feat_test_motion_simple.py` - Combined motion analysis

### Activity Detection
- `feat_test_activity_bursts.py` - Intense activity periods
- `feat_test_binary_activity.py` - Simple active/idle detection
- `feat_test_repetition.py` - Periodic pattern detection

### Limited Use (Screen Recordings)
- `feat_test_blob_tracking.py` - Object tracking (poor for UI)
- `feat_test_blob_stability.py` - Object persistence (noisy)
- `feat_test_connected_components.py` - Connected regions (fails often)
- `feat_test_interaction_zones.py` - Object interactions (rare in UI)
- `feat_test_boundary_crossings.py` - Boundary detection (limited)

### Other Features
- `feat_test_structural_similarity.py` - Frame similarity (SSIM)
- `feat_test_perceptual_hashes.py` - Visual fingerprinting
- `feat_test_periodicity_strength.py` - Periodicity measurement
- `feat_test_app_switching.py` - Application switch detection

## Sample Video

All tests use `sample_recording.MP4` by default. Place this file in the project root or modify the path in individual test scripts.

## Notes

- Tests are designed to demonstrate feature usage
- Each test includes interpretation of results
- Some features work better with physical video than screen recordings
- Check individual script comments for feature-specific details