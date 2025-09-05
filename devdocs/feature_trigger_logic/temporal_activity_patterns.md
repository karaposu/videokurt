# Temporal Activity Patterns Feature - Trigger Logic

## What This Feature Detects

Temporal activity patterns analyze how activity levels change over time, identifying phases, trends, and patterns in user behavior.

## What CREATES Different Pattern Types

### 1. Steady Patterns (Flat Trend)
- Continuous reading
- Video watching
- Steady scrolling
- Consistent typing
- Static monitoring
- Idle periods

### 2. Burst Patterns (Peaks and Valleys)
- Page navigation (burst → idle → burst)
- Search behavior (type → wait → results → repeat)
- Form filling (input → think → input)
- Code debugging (run → examine → modify)
- Shopping (browse → examine → cart)

### 3. Rising Patterns (Increasing Activity)
- Application startup
- Page loading progressively
- User warming up to task
- Accelerating scrolling
- Building momentum
- Escalating interaction

### 4. Falling Patterns (Decreasing Activity)
- Winding down work
- Page settling after load
- User losing interest
- Slowing scrolling
- Task completion
- Fatigue setting in

### 5. Cyclic Patterns (Regular Repetition)
- Test automation runs
- Slideshow viewing
- Tutorial following
- Data entry routine
- Review cycles
- Polling/refresh loops

## Activity Phases

### High Activity Phase
**Triggers:**
- Rapid clicking/typing
- Fast scrolling
- Multiple window switches
- Intensive interaction
- Animation/video playback

**Indicates:**
- Active work
- Navigation
- Search behavior
- Content creation
- Problem-solving

### Medium Activity Phase
**Triggers:**
- Moderate scrolling
- Occasional clicks
- Steady typing
- Form interactions
- Menu navigation

**Indicates:**
- Normal workflow
- Content review
- Casual browsing
- Routine tasks
- Steady progress

### Low Activity Phase
**Triggers:**
- Minimal interaction
- Rare clicks
- No scrolling
- Cursor movement only
- Static content

**Indicates:**
- Reading/thinking
- Waiting
- Watching
- Idle time
- Break periods

## Common Temporal Patterns

### Document Work
```
[Medium] → [Low] → [Medium] → [High] → [Low]
Writing → Reading → Editing → Formatting → Review
```

### Web Research
```
[High] → [Low] → [High] → [Low]
Search → Read → Navigate → Read
```

### Video Consumption
```
[High] → [Low sustained] → [High]
Find video → Watch → Close/navigate
```

### Development
```
[High] → [Medium] → [Low] → [High]
Code → Test → Debug output → Fix
```

### Communication
```
[Medium] → [High burst] → [Medium]
Read → Reply → Continue
```

## Pattern Change Detection

### Significant Changes Indicate
- **Task switches**: Different activity pattern
- **Context changes**: New workflow phase
- **Attention shifts**: Focus change
- **Interruptions**: Unexpected pattern breaks
- **Completions**: Activity drops to baseline

### Pattern Stability Indicates
- **Focused work**: Consistent pattern
- **Routine tasks**: Predictable cycles
- **Engaged state**: Sustained level
- **Automated behavior**: Regular repetition

## Use Cases

1. **Workflow analysis** - Understanding work patterns
2. **Engagement measurement** - Attention and focus
3. **Task segmentation** - Breaking work into phases
4. **Productivity analysis** - Active vs passive time
5. **Behavior prediction** - Anticipating next actions

## Temporal Metrics

| Metric | Meaning | Use |
|--------|---------|-----|
| Mean activity | Average engagement | Overall activity level |
| Variance | Consistency | How variable the work |
| Trend | Direction | Increasing/decreasing |
| Peak count | Burst frequency | How often intensive |
| Phase duration | Stability | How long in each state |

## Window Size Effects

| Window Size | Captures | Best For |
|-------------|----------|----------|
| Small (10-30 frames) | Micro-patterns | Quick interactions |
| Medium (30-100) | Task patterns | Normal workflows |
| Large (100+) | Session patterns | Overall behavior |

## Advantages

1. **Behavior understanding** - Reveals work patterns
2. **Phase detection** - Automatic segmentation
3. **Trend analysis** - Activity direction
4. **Pattern recognition** - Identifies routines
5. **Flexible granularity** - Adjustable windows

## Limitations

1. **No content awareness** - Just activity amount
2. **Window size sensitive** - Different patterns at different scales
3. **Smoothing effects** - May miss brief events
4. **Relative patterns** - No absolute meaning
5. **Context needed** - Patterns alone insufficient

## Typical Output

```python
{
    'patterns': [
        {'start': 0, 'end': 30, 'trend': 'increasing', 'peaks': 2},
        {'start': 30, 'end': 60, 'trend': 'flat', 'peaks': 0}
    ],
    'pattern_changes': [
        {'frame': 30, 'type': 'trend_change', 'magnitude': 0.6}
    ],
    'activity_phases': [
        {'type': 'high', 'start': 0, 'end': 45},
        {'type': 'low', 'start': 45, 'end': 120}
    ],
    'mean_activity': 0.45,
    'activity_variance': 0.12
}
```

## Key Insight

Temporal activity patterns reveal the "rhythm" of user work - from focused steady-state activities to burst-pause cycles. This helps understand not just what users do, but how they structure their work over time.