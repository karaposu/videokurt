# VideoKurt Project Description

## What Are We Building?

VideoKurt is a spatiotemporal pattern analysis tool that performs low-level computer vision analysis on video content. It serves as a visual analysis abstraction layer that detects mechanical changes and predefined event patterns in video without understanding semantic meaning.

The system processes video frames through a four-layer architecture:
1. **Raw Analysis Layer** - Direct pixel-level computations (optical flow, edge detection, frame differencing)
2. **Basic Features Layer** - Simple metrics derived from raw data (motion magnitude, binary activity, edge density)
3. **Middle Features Layer** - Pattern extraction with spatial awareness (blob tracking, trajectories, zone activity)
4. **Advanced Features Layer** - Complex visual pattern detection (scene boundaries, scrolling, camera movement)

## What Problem Are We Solving?

### Primary Problem
Video analysis systems typically face a fundamental inefficiency: they either process every frame with expensive semantic analysis (slow and costly) or sample frames blindly (missing important moments). VideoKurt solves this by providing intelligent pre-processing that marks what's worth analyzing.

### Specific Challenges Addressed

1. **LLM Processing Optimization**: When using multimodal LLMs for video understanding, processing every frame is prohibitively expensive. VideoKurt identifies which frames contain meaningful activity, reducing LLM API calls by 80-90%.

2. **Activity Detection Without Semantics**: Many applications need to know WHEN something happens before understanding WHAT happens. VideoKurt marks active periods, idle times, and visual events without requiring semantic understanding.

3. **Technical Pattern Recognition**: Detecting scrolling, scene cuts, UI transitions, and camera movements are mechanical patterns that don't require AI understanding but are crucial for video analysis.

4. **Video Segmentation**: Breaking videos into meaningful segments (shots, scenes, activity periods) provides structure for downstream analysis without semantic interpretation.

## Project Scopes

### Core Scope
- **Mechanical Visual Analysis**: Detect changes, motion, and patterns using computer vision
- **Event Timeline Generation**: Create binary activity timelines and event markers
- **Pattern Classification**: Identify predefined visual patterns (cuts, scrolls, transitions)
- **Feature Extraction**: Provide numerical features for downstream processing

### Extended Scope
- **VideoQuery Integration**: Serve as the foundation layer for semantic video understanding
- **Performance Optimization**: Enable selective frame processing based on activity
- **Calibration System**: Adapt detection sensitivity for different video types

### Out of Scope
- **Semantic Understanding**: No object recognition, text reading, or content interpretation
- **Business Logic**: No domain-specific rules or interpretations
- **Audio Analysis**: Focus purely on visual content
- **Real-time Processing**: Designed for batch analysis, not live streams

## Targeted Users

### Primary Users

1. **AI/ML Engineers**
   - Building video understanding systems with LLMs
   - Need to optimize frame selection for API calls
   - Want pre-computed visual features

2. **Video Analysis Developers**
   - Creating video summarization tools
   - Building content moderation systems
   - Developing video search applications

3. **Screen Recording Analysts**
   - Analyzing user behavior in UI/UX testing
   - Processing tutorial or demo videos
   - Detecting UI patterns and interactions

### Secondary Users

1. **Media Processing Teams**
   - Shot detection for video editing
   - Scene boundary detection for content indexing
   - Camera movement analysis for cinematography

2. **QA/Testing Teams**
   - Automated UI testing with visual verification
   - Detecting loading states and transitions
   - Identifying visual anomalies

3. **Research Teams**
   - Computer vision research requiring feature extraction
   - Behavioral analysis needing activity detection
   - Performance benchmarking of video processing

### Use Case Examples

**For LLM Integration:**
- VideoKurt marks periods of activity and inactivity
- Provides timestamps and event boundaries
- Higher-level systems decide how to use this information

**For UI Testing:**
- Detects mechanical patterns like scrolling and transitions
- Marks when and where visual changes occur
- Testing frameworks interpret these patterns for validation

**For Content Analysis:**
- Identifies scene boundaries and shot changes
- Provides temporal segmentation based on visual patterns
- Content systems use these markers for their own processing

## Value Proposition

VideoKurt provides the "eyes without understanding" layer that enables efficient video analysis by:
- Reducing computational costs by 80-90% when combined with LLMs
- Providing consistent, mechanical pattern detection
- Offering modular, composable analysis components
- Delivering structured temporal segmentation
- Enabling performance optimization through selective processing

The tool excels at answering "when" and "where" questions about visual activity, leaving "what" and "why" questions to higher-level semantic analysis tools built on top of it.