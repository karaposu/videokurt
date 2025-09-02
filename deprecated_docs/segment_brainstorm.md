# Segment Brainstorm: Moving to Primitive Visual Patterns

## Your Proposal is Spot On

You're absolutely right - VideoKurt should detect **primitive visual patterns** rather than trying to interpret user actions. This is more honest, more verifiable, and truly embodies "See Without Understanding".

## Current Problems with Our Segments

Our current segments still carry interpretation:
- `CLICKING` - assumes user clicked (but what if it was an animation?)
- `TYPING` - assumes text input (but what if it was just text appearing?)
- `VIDEO_PLAYING` - assumes video (but what if it was any animated region?)
- `MIXED_INTERACTION` - too vague, what does this even mean visually?

## Better Primitive Segments (Your Proposal)

### Motion-based Primitives
1. **FULL_VERTICAL_SLIDE_UP** - Large portion (>50%) of screen sliding upward
2. **FULL_VERTICAL_SLIDE_DOWN** - Large portion (>50%) of screen sliding downward
3. **FULL_HORIZONTAL_SLIDE_LEFT** - Large portion (>50%) of screen sliding left
4. **FULL_HORIZONTAL_SLIDE_RIGHT** - Large portion (>50%) of screen sliding right
5. **MINI_VERTICAL_SLIDE_UP** - Small region (<20%) sliding upward (dropdown, list)
6. **MINI_VERTICAL_SLIDE_DOWN** - Small region (<20%) sliding downward
7. **MINI_HORIZONTAL_SLIDE_LEFT** - Small region (<20%) sliding left (carousel, tabs)
8. **MINI_HORIZONTAL_SLIDE_RIGHT** - Small region (<20%) sliding right
9. **IDLE** - No visual changes detected

### Change-based Primitives
10. **FULL_SCREEN_CHANGE** - >80% of pixels changed
11. **PARTIAL_SCREEN_CHANGE** - 20-80% of pixels changed
12. **LOCALIZED_CHANGE** - <20% of pixels changed
13. **REGION_ANIMATION** - Repeated changes in same bounded region

### Pattern-based Primitives
14. **UNIFORM_FADE** - Gradual uniform change across screen
15. **RAPID_FLICKER** - Quick repeated changes in same location

## Why This is Better

1. **Purely Visual** - These describe what the video shows, not what we think happened
2. **Verifiable** - We can mathematically verify "30% of pixels changed" 
3. **No Assumptions** - "SCROLL_DOWN" describes motion, not user intent
4. **Composable** - Higher-level interpretations can combine primitives later

## Implementation Approach

```python
class PrimitiveSegmentType(Enum):
    # Motion patterns - Full screen
    FULL_VERTICAL_SLIDE_UP = "full_vertical_slide_up"       # >50% screen sliding up
    FULL_VERTICAL_SLIDE_DOWN = "full_vertical_slide_down"   # >50% screen sliding down
    FULL_HORIZONTAL_SLIDE_LEFT = "full_horizontal_slide_left"   # >50% screen sliding left
    FULL_HORIZONTAL_SLIDE_RIGHT = "full_horizontal_slide_right" # >50% screen sliding right
    
    # Motion patterns - Mini regions
    MINI_VERTICAL_SLIDE_UP = "mini_vertical_slide_up"       # <20% region sliding up
    MINI_VERTICAL_SLIDE_DOWN = "mini_vertical_slide_down"   # <20% region sliding down
    MINI_HORIZONTAL_SLIDE_LEFT = "mini_horizontal_slide_left"   # <20% region sliding left
    MINI_HORIZONTAL_SLIDE_RIGHT = "mini_horizontal_slide_right" # <20% region sliding right
    
    # Change patterns
    IDLE = "idle"                        # No changes
    FULL_CHANGE = "full_change"          # >80% pixels changed
    PARTIAL_CHANGE = "partial_change"    # 20-80% pixels changed
    LOCALIZED_CHANGE = "localized_change" # <20% pixels changed
    
    # Animation patterns
    REGION_ANIMATION = "region_animation" # Repeated changes in fixed region
    UNIFORM_FADE = "uniform_fade"        # Gradual uniform change
```

## Detection Methods

Each primitive can be detected with simple algorithms:

1. **Motion detection**: 
   - Optical flow to detect direction of movement
   - Calculate percentage of screen affected
   - Classify as FULL (>50%) or MINI (<20%)
   
2. **Change magnitude**: 
   - Simple pixel difference percentage
   - Classify into FULL/PARTIAL/LOCALIZED based on thresholds
   
3. **Region tracking**: 
   - Track bounding boxes of changing areas
   - Detect if changes stay within same region (animation)
   
4. **Pattern detection**:
   - Uniform changes across frame (fade)
   - Rapid changes in same pixels (flicker)

## Benefits for Users

Users can then interpret these primitives however they want:
- `FULL_VERTICAL_SLIDE_UP` → "User scrolling through feed"
- `MINI_VERTICAL_SLIDE_UP` → "User scrolling in dropdown/list"
- `PARTIAL_CHANGE` + `LOCALIZED_CHANGE` → "Modal opened then interaction"
- `REGION_ANIMATION` → "Video/GIF/animation playing"
- `FULL_CHANGE` → "Navigation to new page"

But VideoKurt itself makes NO such interpretations.

## Transition Plan

1. Keep current segments temporarily for compatibility
2. Add new primitive detection alongside
3. Gradually migrate to primitives-only
4. Let users/plugins do higher-level interpretation

## Your Examples Mapped

Your examples perfectly map to primitives:
- "scroll downwards" → `FULL_VERTICAL_SLIDE_UP` (content moves up)
- "scroll upwards" → `FULL_VERTICAL_SLIDE_DOWN` (content moves down)
- "user clicked sth and modal opened" → `LOCALIZED_CHANGE` then `PARTIAL_CHANGE`
- "tab changes in same app" → `PARTIAL_CHANGE` (20-80% change)
- "full screen change" → `FULL_CHANGE` (>80% change)
- "dropdown scrolling" → `MINI_VERTICAL_SLIDE_UP/DOWN` (<20% region)

## Conclusion

This approach is:
- **More honest** about what we can actually detect
- **More useful** because it's verifiable
- **More flexible** because users can interpret as needed
- **More aligned** with VideoKurt's core philosophy

You're absolutely right - let's make VideoKurt detect primitive visual patterns, not interpreted user actions. The tool should report what it sees, not what it thinks is happening.

## Next Steps

1. Define exact thresholds (what % is "partial" vs "full"?)
2. Implement primitive detectors
3. Create mapping layer for backward compatibility
4. Document primitive patterns clearly
5. Provide examples of combining primitives for interpretation