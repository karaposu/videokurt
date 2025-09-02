# OpenCV Feature Explorations for VideoKurt

## Summary of Findings

We explored several OpenCV features that VideoKurt could use to detect primitive visual patterns. Here's what each method is good for:

## 1. **Farneback Optical Flow** (`01_optical_flow_farneback.py`)
**What it does:** Calculates dense motion vectors for every pixel

**Good for detecting:**
- ✅ `FULL_VERTICAL_SLIDE_UP/DOWN` - Detected scrolling as "UP" motion
- ✅ `FULL_HORIZONTAL_SLIDE_LEFT/RIGHT` - Can detect horizontal scrolling
- ✅ Motion magnitude - Distinguishes between fast and slow scrolling
- ✅ Motion scope - Detected "LOCALIZED" vs "FULL SCREEN" motion

**Results on test video:**
- Successfully detected vertical scrolling (content moving UP = user scrolling down)
- Identified motion was localized to small regions (2.8% of pixels moving)
- Average motion magnitude: 0.29 pixels/frame

## 2. **Lucas-Kanade Optical Flow** (`02_optical_flow_lucas_kanade.py`)
**What it does:** Tracks specific feature points between frames

**Good for detecting:**
- ✅ `UNIFORM_MOTION` vs `SCATTERED_MOTION` - Distinguishes scrolling from UI changes
- ✅ Tracking UI elements as they move
- ✅ Direction of movement for specific regions
- ✅ Detecting when elements appear/disappear

**Results on test video:**
- Detected 22 motion events
- Classified 10 as UNIFORM (scrolling) and 12 as SCATTERED (UI changes)
- Identified VERTICAL_SLIDE_UP as dominant pattern
- Motion intensity: 7.22 pixels/frame average

## 3. **Background Subtraction (MOG2/KNN)** (`03_background_subtraction.py`)
**What it does:** Learns the "background" and detects "foreground" changes

**Good for detecting:**
- ✅ `FULL_CHANGE` - Detected as 80%+ pixel changes
- ✅ `PARTIAL_CHANGE` - Detected as 20-80% changes
- ✅ `LOCALIZED_CHANGE` - Detected as <20% changes
- ✅ `REGION_ANIMATION` - Detected repeated changes in same region

**Results on test video:**
- Detected scrolling as FULL_CHANGE (up to 92% of screen changing)
- Identified REGION_ANIMATION in specific areas
- MOG2 more sensitive than KNN
- Average change: 27.6% of pixels

## 4. **Contour Detection** (`04_contour_detection.py`)
**What it does:** Finds boundaries of changing regions

**Good for detecting:**
- ✅ Shape classification (LINEAR_ELEMENT, CIRCULAR_ELEMENT)
- ✅ Element alignment (HORIZONTAL_ELEMENTS, VERTICAL_ELEMENTS)
- ✅ `SCATTERED_CHANGES` vs single region changes
- ✅ Movement tracking of UI elements

**Results on test video:**
- Detected SCATTERED_CHANGES during scrolling (many small contours)
- Identified HORIZONTAL_ELEMENTS (aligned UI components)
- Tracked vertical movement of elements
- Found persistent regions that could be animations

## Recommendations for VideoKurt

### For Primitive Segment Detection:

1. **Scrolling/Sliding Detection:**
   - Use **Farneback optical flow** for overall motion direction and magnitude
   - Threshold on motion percentage to distinguish FULL vs MINI sliding

2. **Change Magnitude Detection:**
   - Use **Background Subtraction (MOG2)** to measure pixel change percentage
   - Classify into FULL/PARTIAL/LOCALIZED based on thresholds

3. **Region Animation Detection:**
   - Use **Contour Detection** to find persistent changing regions
   - Track bounding boxes across frames to identify animations

4. **UI Element Tracking:**
   - Use **Lucas-Kanade** for tracking specific features
   - Distinguish uniform motion (scrolling) from scattered changes (UI updates)

### Integration Strategy:

```python
def detect_primitive_segments(frames):
    # 1. Optical flow for motion patterns
    flow = cv2.calcOpticalFlowFarneback(...)
    motion_direction = analyze_flow_direction(flow)
    motion_magnitude = analyze_flow_magnitude(flow)
    
    # 2. Background subtraction for change magnitude
    mask = mog2.apply(frame)
    change_percent = calculate_change_percentage(mask)
    
    # 3. Contours for region tracking
    contours = cv2.findContours(mask, ...)
    regions = track_regions(contours, prev_contours)
    
    # 4. Classify into primitive segments
    if motion_magnitude > threshold and motion_direction == "vertical":
        if change_percent > 50:
            return "FULL_VERTICAL_SLIDE_UP"
        else:
            return "MINI_VERTICAL_SLIDE_UP"
    elif change_percent > 80:
        return "FULL_CHANGE"
    # ... etc
```

### Performance Considerations:

- **Farneback**: Computationally expensive but gives dense flow
- **Lucas-Kanade**: Faster, good for tracking specific points
- **MOG2**: Needs 20-30 frames to learn background
- **Contours**: Fast, but sensitive to noise (needs preprocessing)

### Next Steps:

1. Integrate these methods into VideoKurt's primitive segment detection
2. Tune thresholds for each primitive pattern
3. Combine multiple methods for robust detection
4. Add these as options in VideoKurt's configuration