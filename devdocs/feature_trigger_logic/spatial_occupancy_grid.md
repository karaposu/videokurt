# Spatial Occupancy Grid Feature - Trigger Logic

## What This Feature Detects

Spatial occupancy grid divides the frame into a grid and measures activity levels in each cell, showing the spatial distribution of movement.

## What CREATES High Occupancy

### 1. Active Grid Cells
- Frequent activity in region
- Continuous movement
- High pixel changes
- Animation hotspots
- Interaction zones
- Dynamic content areas

### 2. Common High-Occupancy Regions

**Center Cells (Main Content)**
- Document editing area
- Video player region
- Main application window
- Content scrolling zone
- Game play area

**Top Cells (Headers)**
- Navigation menus
- Tab switches
- Title bar interactions
- Toolbar usage
- Status updates

**Right Cells (Sidebars)**
- Scrollbars
- Chat panels
- Navigation panes
- Tool palettes
- Notification areas

**Bottom Cells (Footer/Controls)**
- Taskbar interactions
- Player controls
- Status bars
- Command lines
- Input areas

## Grid Pattern Interpretations

### Centered Activity
```
Grid (3x3):
[Low ][Low ][Low ]
[Low ][HIGH][Low ]
[Low ][Low ][Low ]
```
**Indicates**: Focused central work, modal dialog, video watching

### Vertical Strip
```
Grid (3x3):
[Low ][HIGH][Low ]
[Low ][HIGH][Low ]
[Low ][HIGH][Low ]
```
**Indicates**: Scrolling, vertical navigation, sidebar activity

### Horizontal Strip
```
Grid (3x3):
[Low ][Low ][Low ]
[HIGH][HIGH][HIGH]
[Low ][Low ][Low ]
```
**Indicates**: Horizontal scrolling, timeline scrubbing, tab bar

### Scattered
```
Grid (3x3):
[HIGH][Low ][HIGH]
[Low ][HIGH][Low ]
[HIGH][Low ][HIGH]
```
**Indicates**: Multiple windows, scattered interactions, noise

### Top-Heavy
```
Grid (3x3):
[HIGH][HIGH][HIGH]
[Med ][Med ][Med ]
[Low ][Low ][Low ]
```
**Indicates**: Header interactions, menu usage, tab switching

## Common Patterns by Application

### Web Browser
- High: Center (content), top (tabs/URL)
- Low: Edges, corners
- Pattern: T-shaped or centered

### Code Editor
- High: Left-center (code), left (file tree)
- Medium: Top (tabs), bottom (terminal)
- Pattern: Left-weighted

### Video Player
- High: Center (video)
- Low: Everything else during playback
- Pattern: Center-focused

### Spreadsheet
- High: Varies with active cell
- Medium: Headers (row/column labels)
- Pattern: Cross-hair at active cell

## Occupancy Metrics

| Metric | Meaning | Interpretation |
|--------|---------|----------------|
| High concentration | Activity in few cells | Focused work |
| Even distribution | Activity spread out | Scattered attention |
| Edge activity | Perimeter cells active | UI navigation |
| Center activity | Middle cells active | Content focus |
| Zero occupancy | No activity anywhere | Idle/reading |

## Grid Size Effects

| Grid Size | Resolution | Best For |
|-----------|------------|----------|
| 2×2 | Quadrants | Basic regions |
| 3×3 | Nine zones | Standard analysis |
| 4×4 | Detailed | Fine-grained patterns |
| 5×5+ | Very detailed | Specific region tracking |

## Use Cases

1. **Attention analysis** - Where user focuses
2. **UI usage patterns** - Which areas get used
3. **Layout effectiveness** - Dead zones detection
4. **Activity distribution** - Concentrated vs spread
5. **Workflow patterns** - Spatial work patterns

## Advantages

1. **Spatial awareness** - Location-based insights
2. **Simple visualization** - Grid heatmap
3. **Normalized metric** - Compare across videos
4. **Pattern detection** - Spatial signatures
5. **Efficient summary** - Reduces data

## Limitations

1. **Fixed grid** - May not align with UI
2. **Resolution loss** - Averages within cells
3. **No temporal info** - When activity occurred
4. **No semantics** - What caused activity
5. **Grid size dependent** - Different patterns

## Typical Output

```python
{
    'occupancy_grid': [
        [0.1, 0.3, 0.1],
        [0.2, 0.8, 0.2],  # 3×3 grid
        [0.1, 0.4, 0.1]
    ],
    'most_active_cell': (1, 1),  # Center
    'activity_distribution': 'concentrated',
    'statistics': {
        'total_active_cells': 5,
        'activity_variance': 0.24,
        'spatial_entropy': 0.67
    }
}
```

## Visualization

```
Heatmap colors:
■ Dark Red   - Very high activity (>80%)
■ Red        - High activity (60-80%)
■ Orange     - Medium-high (40-60%)
■ Yellow     - Medium (20-40%)
■ Light Blue - Low activity (5-20%)
■ Dark Blue  - Very low (<5%)
```

## Key Insight

Spatial occupancy grids reveal "activity geography" - where things happen on screen. This is invaluable for understanding UI usage patterns and identifying whether users work in focused regions or spread their attention across the interface.