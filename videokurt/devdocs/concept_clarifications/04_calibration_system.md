# Calibration System

## What It Is and Why It Matters
The calibration system is a configuration mechanism that adjusts detection sensitivity and parameters based on video content type (mobile app, desktop, presentation, etc.). It matters because different video contexts have vastly different characteristics - what constitutes a scene change in a presentation differs greatly from a mobile app recording.

## How This Helps the Overall Project
Calibration system enables VideoKurt to:
- Work effectively across diverse video types without code changes
- Provide optimal detection accuracy for each use case
- Reduce false positives through context-aware thresholds
- Simplify user experience with preset configurations
- Enable fine-tuning without modifying core algorithms
- Support new video types through configuration alone

## How This Limits the Overall Project
Calibration system constraints:
- Requires prior knowledge of video content type
- Cannot automatically determine optimal settings
- May need manual tuning for edge cases
- Creates complexity in managing multiple profiles
- Can mask algorithm deficiencies through parameter tweaking
- Makes reproducibility dependent on calibration consistency

## Input Requirements
Calibration system needs:
- Video content type or preset name
- Optional: sample video for auto-calibration
- Optional: user-provided threshold overrides
- Optional: performance vs accuracy preference
- Detection method preferences per event type
- Environmental context (resolution, frame rate)

## Process Description
The calibration process:
1. Load base calibration profile for content type
2. Apply user-specified overrides if provided
3. Adjust thresholds based on video properties:
   - Scale for resolution differences
   - Compensate for frame rate variations
   - Account for compression quality
4. Configure detection methods per event type
5. Set processing parameters (frame skip, regions)
6. Initialize detectors with calibrated values
7. Optional: run validation on sample frames
8. Store active calibration for session

## Output Information
Calibration system outputs:
- Active threshold values for all detections
- Selected detection methods per event type
- Processing parameters (resolution, frame skip)
- Confidence score adjustments
- Minimum duration requirements
- Region of interest specifications
- Performance optimization settings

## Good Expected Outcome
When calibration works well:
- Each video type achieves optimal detection accuracy
- False positive rate stays below 5%
- True positive rate exceeds 95%
- Processing speed meets performance targets
- Users rarely need manual adjustment
- New content types easily supported through profiles

## Bad Unwanted Outcome
When calibration fails:
- Generic settings work poorly across video types
- Users must constantly tune parameters manually
- Preset profiles don't match real-world content
- Calibration becomes excuse for poor algorithms
- Settings drift over time without validation
- Complexity overwhelms users with too many options