# Binary Activity Timeline

## What It Is and Why It Matters
Binary activity timeline is the core abstraction that reduces complex video content to a simple active/inactive state over time. It matters because it provides the primary filtering mechanism that enables 80-90% cost reduction by identifying which time periods deserve analysis and which can be skipped entirely.


## How This Helps the Overall Project
Binary activity timeline enables VideoKurt to:
- Provide immediate value without complex analysis
- Create clear skip zones for downstream processing
- Simplify decision-making with unambiguous states
- Enable fast first-pass filtering of entire videos
- Reduce API costs by eliminating dead time analysis
- Offer intuitive output that non-technical users understand

## How This Limits the Overall Project
Binary timeline constraints:
- Loses nuance by forcing all activity into two states
- Cannot express varying degrees of importance
- May incorrectly classify subtle but important changes as inactive
- Provides no information about the type of activity
- Creates hard boundaries that may split related events
- Requires downstream systems to handle state transitions

## Input Requirements
Binary timeline generation needs:
- Frame difference scores over time
- Minimum activity threshold value
- Minimum duration for state changes (debouncing)
- Optional: event detection results for validation
- Optional: calibration profile for context-specific thresholds
- Video timestamp information for accurate timing

## Process Description
The binary timeline process:
1. Collect frame difference scores for entire video
2. Apply temporal smoothing to reduce noise
3. Compare smoothed scores against activity threshold
4. Apply minimum duration filter to prevent rapid switching
5. Identify continuous periods of same state
6. Merge adjacent similar states if gap is small
7. Generate timeline entries with start/end timestamps
8. Validate against detected events for consistency

## Output Information
Binary timeline outputs:
- List of time periods with active/inactive state
- Start and end timestamps for each period
- Duration of each period
- Total active vs inactive time
- Activity ratio (active time / total time)
- Number of state transitions
- Longest continuous active/inactive periods

## Good Expected Outcome
When binary timeline works well:
- Accurately identifies all periods of meaningful activity
- Correctly marks loading/idle times as inactive
- Provides clean boundaries between activity states
- Reduces processing needs by 80-90%
- Maintains consistency across similar videos
- Enables reliable cost predictions for analysis

## Bad Unwanted Outcome
When binary timeline fails:
- Marks important subtle changes as inactive
- Fragments continuous activity into many small pieces
- Creates timeline with excessive state switches
- Includes long inactive periods within active zones
- Misclassifies entire video as active due to noise
- Provides no value due to everything being marked active