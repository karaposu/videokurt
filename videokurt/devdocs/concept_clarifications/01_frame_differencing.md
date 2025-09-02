# Frame Differencing

## What It Is and Why It Matters
Frame differencing is the fundamental technique of comparing consecutive video frames at the pixel level to quantify visual changes. It matters because it provides the raw signal for all activity detection - without knowing what changed, we can't determine if anything happened at all.

## How This Helps the Overall Project
Frame differencing enables VideoKurt to:
- Detect any visual change without prior knowledge of content
- Work universally across all video types and applications
- Provide real-time change detection with minimal computation
- Create the foundation data for all higher-level event detection
- Operate without machine learning models or training data

## How This Limits the Overall Project
Frame differencing constraints:
- Cannot distinguish between meaningful and noise-based changes
- Sensitive to video compression artifacts and quality issues
- May trigger false positives from minor lighting changes or camera shake
- Provides no semantic understanding of what the changes represent
- Requires careful threshold tuning to balance sensitivity vs noise

## Input Requirements
Frame differencing needs:
- Two consecutive video frames as pixel arrays
- Color space specification (RGB, grayscale, HSV)
- Resolution information for scaling calculations
- Optional: region of interest masks to focus analysis
- Optional: previous frame history for temporal smoothing

## Process Description
The frame differencing process:
1. Extract two consecutive frames from video stream
2. Convert frames to consistent color space (typically grayscale)
3. Compute absolute pixel-wise difference between frames
4. Apply noise reduction filters (Gaussian blur, morphological operations)
5. Calculate aggregate metrics (sum, mean, maximum difference)
6. Compare metrics against calibrated thresholds
7. Output change magnitude and affected regions

## Output Information
Frame differencing outputs:
- Change score (0.0-1.0 normalized difference value)
- Binary change flag (changed/unchanged based on threshold)
- Change mask (pixel-level map of changed regions)
- Statistical metrics (mean, std deviation, percentiles)
- Bounding boxes of changed regions
- Temporal change velocity (rate of change over time)

## Good Expected Outcome
When frame differencing works well:
- Reliably detects all significant visual changes
- Maintains consistent sensitivity across video duration
- Provides clear signal for activity vs inactivity periods
- Enables accurate event boundary detection
- Processes frames faster than video playback speed
- Produces minimal false positives during static scenes

## Bad Unwanted Outcome
When frame differencing fails:
- Floods system with false positives from compression noise
- Misses gradual changes that occur over many frames
- Triggers constantly on videos with watermarks or overlays
- Cannot differentiate camera movement from content movement
- Produces inconsistent results based on video quality
- Creates computational bottleneck for high-resolution content