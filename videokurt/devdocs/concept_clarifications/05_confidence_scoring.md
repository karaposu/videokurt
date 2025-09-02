# Confidence Scoring

## What It Is and Why It Matters
Confidence scoring is the numerical measurement (0.0-1.0) of detection certainty for each identified event or change. It matters because it allows downstream systems to make informed decisions about which detections to trust, enabling filtering of uncertain results and prioritization of high-confidence events.

## How This Helps the Overall Project
Confidence scoring enables VideoKurt to:
- Communicate detection uncertainty transparently
- Allow users to set quality thresholds for their use case
- Enable gradual degradation rather than binary failure
- Support debugging by identifying weak detections
- Facilitate A/B testing of detection algorithms
- Provide data for continuous improvement

## How This Limits the Overall Project
Confidence scoring constraints:
- Adds complexity to every detection algorithm
- May create false sense of precision in scores
- Requires calibration to maintain consistent meaning
- Can mislead users if scores aren't well-calibrated
- Increases output data size and processing overhead
- Makes binary decisions harder when scores are borderline

## Input Requirements
Confidence scoring needs:
- Raw detection algorithm output
- Calibration data for score normalization
- Detection-specific quality metrics:
  - Template matching: correlation coefficient
  - Optical flow: motion coherence
  - Differencing: change magnitude
- Historical performance data for calibration
- Context about expected detection difficulty

## Process Description
The confidence scoring process:
1. Collect raw metrics from detection algorithm
2. Normalize metrics to 0.0-1.0 range
3. Apply detection-specific weighting factors
4. Account for environmental factors:
   - Video quality adjustments
   - Motion blur penalties
   - Compression artifact compensation
5. Combine multiple signals if available
6. Apply calibration curve for consistency
7. Validate score against known thresholds
8. Attach confidence to detection result

## Output Information
Confidence scoring outputs:
- Normalized confidence value (0.0-1.0)
- Confidence classification (high/medium/low)
- Contributing factor breakdown
- Uncertainty sources identified
- Recommended threshold for filtering
- Comparison to baseline expectations
- Statistical confidence interval

## Good Expected Outcome
When confidence scoring works well:
- High-confidence detections are virtually always correct
- Low-confidence detections appropriately flag uncertainty
- Scores remain consistent across similar videos
- Users can reliably filter by confidence threshold
- Provides clear signal for manual review needs
- Enables accurate cost/quality tradeoffs

## Bad Unwanted Outcome
When confidence scoring fails:
- High confidence assigned to false positives
- Low confidence on obvious correct detections
- Scores vary wildly for similar content
- Users cannot find useful threshold values
- All detections cluster at similar scores
- Confidence doesn't correlate with accuracy