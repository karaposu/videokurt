# Processing Modes

## What It Is and Why It Matters
Processing modes are predefined speed/accuracy configurations (fast, balanced, thorough) that trade off analysis quality for performance. It matters because different use cases have different requirements - quick previews need speed, while final analysis needs accuracy, and users need simple ways to express these preferences.

## How This Helps the Overall Project
Processing modes enable VideoKurt to:
- Serve diverse use cases with single codebase
- Provide predictable performance characteristics
- Simplify user decisions with clear presets
- Enable iterative analysis workflows
- Support both real-time and batch processing
- Scale from quick previews to detailed analysis

## How This Limits the Overall Project
Processing mode constraints:
- Forces compromise between speed and accuracy
- Cannot optimize for both simultaneously
- Presets may not match all use cases perfectly
- Creates testing complexity with multiple modes
- May produce inconsistent results across modes
- Requires users to understand tradeoffs

## Input Requirements
Processing modes need:
- Selected mode (fast/balanced/thorough)
- Video resolution and frame rate
- Available computational resources
- Optional: specific accuracy requirements
- Optional: time constraints
- Optional: custom mode parameters

## Process Description
The processing mode configuration:
1. Load mode-specific parameters:
   - Fast: 480p, skip 2 frames, basic detections
   - Balanced: 720p, adaptive skipping, standard detections
   - Thorough: full resolution, all frames, all detections
2. Configure video decoder settings
3. Set frame sampling strategy
4. Enable/disable detection methods
5. Adjust confidence thresholds
6. Configure parallel processing
7. Set memory usage limits
8. Initialize processing pipeline

## Output Information
Processing modes affect:
- Processing speed (frames per second)
- Detection accuracy rates
- Confidence score reliability
- Event detection completeness
- Temporal resolution of timeline
- Memory and CPU usage
- Output data size and detail

## Good Expected Outcome
When processing modes work well:
- Fast mode provides 10x real-time preview
- Balanced mode achieves 95% accuracy at 5x speed
- Thorough mode catches all subtle events
- Users easily choose appropriate mode
- Results scale predictably with mode selection
- Progressive refinement workflows enabled

## Bad Unwanted Outcome
When processing modes fail:
- Fast mode misses critical events
- Thorough mode takes prohibitively long
- Balanced mode satisfies neither speed nor accuracy
- Mode differences cause confusion
- Results vary unpredictably between modes
- No mode matches actual user needs