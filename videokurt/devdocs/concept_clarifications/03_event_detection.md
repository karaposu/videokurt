# Event Detection

## What It Is and Why It Matters
Event detection is the identification of specific predefined visual patterns (scrolls, clicks, scene changes, popups) through specialized computer vision techniques. It matters because it provides structured metadata about what type of mechanical changes occurred, enabling downstream systems to prioritize and contextualize their analysis.

## How This Helps the Overall Project
Event detection enables VideoKurt to:
- Provide rich metadata beyond simple active/inactive states
- Enable targeted analysis of specific interaction types
- Create natural segmentation points for video processing
- Offer actionable insights without semantic understanding
- Support quality assurance and testing workflows
- Generate precise timestamps for important moments

## How This Limits the Overall Project
Event detection constraints:
- Limited to predefined mechanical patterns only
- Cannot detect novel or unexpected event types
- Requires separate detection logic for each event type
- May conflict when multiple events occur simultaneously
- Increases computational cost compared to simple differencing
- Creates maintenance burden as new patterns emerge

## Input Requirements
Event detection needs:
- Video frames or frame sequences
- Event-specific detection parameters
- Calibration thresholds for each event type
- Optional: regions of interest for focused detection
- Optional: template images for pattern matching
- Optional: historical context for temporal events

## Process Description
The event detection process:
1. Apply appropriate detection method per event type:
   - Optical flow for scrolls and swipes
   - Template matching for UI elements
   - Histogram analysis for scene changes
   - Region monitoring for popups
2. Track detection confidence over time
3. Apply event-specific validation rules
4. Determine event boundaries (start/end times)
5. Extract event-specific metadata
6. Resolve conflicts between overlapping events
7. Filter events below confidence threshold
8. Package events with timestamps and metadata

## Output Information
Event detection outputs:
- Event type identifier (scroll, click, scene_change, etc.)
- Start and end timestamps
- Confidence score for detection
- Event-specific metadata:
  - Scroll: direction, velocity, distance
  - Scene change: transition type, magnitude
  - Popup: bounds, overlay type
  - Click: coordinates, target bounds
- Duration of event
- Frame numbers where event occurs

## Good Expected Outcome
When event detection works well:
- Accurately identifies all standard UI interactions
- Provides precise timestamps for each event
- Generates rich metadata for downstream analysis
- Maintains high confidence scores for clear events
- Handles overlapping events gracefully
- Enables targeted frame extraction for specific events

## Bad Unwanted Outcome
When event detection fails:
- Generates many false positive events from noise
- Misses events due to variant implementations
- Provides incorrect event classifications
- Creates conflicting or impossible event sequences
- Significantly slows video processing
- Produces unreliable confidence scores