# VideoKurt Features Summary

This document provides a comprehensive overview of all features in VideoKurt, organized by complexity level (Basic, Middle, Advanced). Each feature includes its name, output format, and typical use cases.

## Basic Features

Simple computations on raw analysis data that provide immediate value with minimal processing.

### 1. Binary Activity
**Output:** `np.ndarray[uint8]` - Binary array (0=inactive, 1=active) for each frame  
**Use Cases:**
- Activity timeline generation
- Idle time detection
- Video summarization (skip inactive parts)
- Screen recording analysis

### 2. Motion Magnitude
**Output:** `np.ndarray[float32]` - Average motion magnitude per frame  
**Use Cases:**
- Camera shake detection
- Action intensity measurement
- Motion-based video segmentation
- Sports/action scene analysis

### 3. Motion Direction Histogram
**Output:** `np.ndarray[float32, shape=(n_frames, n_bins)]` - Distribution of motion directions  
**Use Cases:**
- Camera pan/tilt detection
- Dominant motion analysis
- Scrolling direction detection
- Object tracking patterns

### 4. Edge Density
**Output:** `np.ndarray[float32]` - Ratio of edge pixels per frame  
**Use Cases:**
- Scene complexity measurement
- Text/UI detection
- Blur/focus analysis
- Content type classification

### 5. Change Regions
**Output:** `List[List[Tuple]]` - Bounding boxes of changed regions per frame  
**Use Cases:**
- UI element tracking
- Localized motion detection
- Region of interest identification
- Interactive element detection

### 6. Stability Score
**Output:** `np.ndarray[float32]` - Frame stability score (0=unstable, 1=stable)  
**Use Cases:**
- Camera stability analysis
- Static scene detection
- Video quality assessment
- Transition detection

### 7. Repetition Indicator
**Output:** `np.ndarray[float32]` - Repetition strength per frame  
**Use Cases:**
- Loop detection
- Cyclic motion identification
- Pattern recognition
- Animation analysis

### 8. Foreground Ratio
**Output:** `np.ndarray[float32]` - Ratio of foreground pixels (requires background subtraction)  
**Use Cases:**
- Object presence detection
- Scene occupancy analysis
- Activity level measurement
- Security monitoring

### 9. Frame Difference Percentile
**Output:** `Dict` with percentiles (25th, 50th, 75th, 90th) of frame differences  
**Use Cases:**
- Statistical motion analysis
- Outlier detection
- Adaptive thresholding
- Quality metrics

### 10. Dominant Flow Vector
**Output:** `np.ndarray[float32, shape=(n_frames, 2)]` - Main motion direction (x, y)  
**Use Cases:**
- Global motion estimation
- Camera movement tracking
- Scrolling detection
- Stabilization parameters

### 11. Histogram Statistics
**Output:** `Dict` with color histogram statistics (mean, std, skewness, kurtosis)  
**Use Cases:**
- Color consistency analysis
- Lighting change detection
- Scene classification
- White balance assessment

### 12. DCT Energy
**Output:** `np.ndarray[float32]` - DCT coefficient energy per frame  
**Use Cases:**
- Compression quality estimation
- Texture complexity measurement
- Perceptual hashing
- Duplicate frame detection

### 13. Texture Uniformity
**Output:** `np.ndarray[float32]` - Texture uniformity score per frame  
**Use Cases:**
- Solid color detection
- UI vs natural scene classification
- Texture-based segmentation
- Quality assessment

---

## Middle Features

Pattern extraction and structured data analysis with temporal and spatial awareness.

### 1. Blob Tracking
**Output:** `Dict` with tracked blobs, trajectories, and blob count per frame  
**Use Cases:**
- Object tracking
- People counting
- UI element tracking
- Motion pattern analysis

### 2. Blob Stability
**Output:** `Dict` with stability scores and persistent blob information  
**Use Cases:**
- Static object detection
- UI layout analysis
- Scene understanding
- Occlusion handling

### 3. Dwell Time Maps
**Output:** `np.ndarray[float32, shape=(height, width)]` - Heatmap of time spent per location  
**Use Cases:**
- Attention analysis
- UI usability studies
- Hot zone identification
- Movement patterns

### 4. Zone-Based Activity
**Output:** `Dict` with activity levels for predefined zones (quadrants, center, edges)  
**Use Cases:**
- Screen region analysis
- Multi-window detection
- Layout understanding
- Interaction patterns

### 5. Motion Trajectories
**Output:** `List[Dict]` - Tracked point trajectories with positions and velocities  
**Use Cases:**
- Object path analysis
- Gesture recognition
- Movement prediction
- Behavior analysis

### 6. Interaction Zones
**Output:** `Dict` with identified interaction regions and frequency  
**Use Cases:**
- UI hotspot detection
- User behavior analysis
- Click/tap prediction
- Engagement metrics

### 7. Activity Bursts
**Output:** `Dict` with burst events, timing, and intensity  
**Use Cases:**
- Event detection
- Highlight extraction
- Anomaly detection
- Rhythm analysis

### 8. Periodicity Strength
**Output:** `Dict` with period detection and strength scores  
**Use Cases:**
- Repetitive motion detection
- Animation cycle analysis
- Pattern recognition
- Quality control

### 9. Boundary Crossings
**Output:** `Dict` with crossing events and statistics  
**Use Cases:**
- Entry/exit detection
- Zone transition analysis
- Security monitoring
- Sports analytics

### 10. Spatial Occupancy Grid
**Output:** `np.ndarray[float32, shape=(grid_h, grid_w, n_frames)]` - Occupancy over time  
**Use Cases:**
- Space utilization analysis
- Crowd dynamics
- Traffic patterns
- Layout optimization

### 11. Temporal Activity Patterns
**Output:** `Dict` with temporal patterns and activity cycles  
**Use Cases:**
- Workflow analysis
- Behavioral patterns
- Scheduling optimization
- Anomaly detection

### 12. Structural Similarity
**Output:** `np.ndarray[float32]` - SSIM scores between consecutive frames  
**Use Cases:**
- Quality assessment
- Scene change detection
- Compression artifacts
- Video stabilization

### 13. Perceptual Hashes
**Output:** `np.ndarray[uint8, shape=(n_frames, hash_size)]` - Perceptual hash per frame  
**Use Cases:**
- Duplicate detection
- Content matching
- Copyright detection
- Video fingerprinting

### 14. Connected Components
**Output:** `Dict` with component count, sizes, and properties  
**Use Cases:**
- Object segmentation
- Text detection
- UI element counting
- Scene parsing

---

## Advanced Features

Complex visual pattern detection using multiple cues and sophisticated algorithms.

### 1. Scene Detection
**Output:** `Dict` with scene boundaries, types (cut/fade/dissolve), and confidence scores  
**Use Cases:**
- Video editing
- Content indexing
- Chapter generation
- Highlight creation

### 2. Camera Movement
**Output:** `Dict` with movement type (pan/tilt/zoom), direction, and magnitude  
**Use Cases:**
- Cinematography analysis
- Video stabilization
- Motion compensation
- Style classification

### 3. Scrolling Detection
**Output:** `Dict` with scroll events, direction, speed, and content type  
**Use Cases:**
- UI automation testing
- Document analysis
- Reading behavior studies
- Content extraction

### 4. UI Change Detection
**Output:** `Dict` with UI change events, affected regions, and change types  
**Use Cases:**
- Automated testing
- User interaction analysis
- Tutorial generation
- Bug detection

### 5. App Window Switching
**Output:** `Dict` with switch events, window regions, and transition types  
**Use Cases:**
- Workflow analysis
- Multitasking studies
- Productivity measurement
- Screen recording segmentation

### 6. Motion Pattern Classification
**Output:** `Dict` with pattern types (linear/circular/chaotic/static) and confidence  
**Use Cases:**
- Behavior classification
- Activity recognition
- Anomaly detection
- Content categorization

### 7. Shot Type Detection
**Output:** `Dict` with shot types (close-up/medium/wide/extreme) and confidence  
**Use Cases:**
- Film analysis
- Automatic editing
- Content understanding
- Style analysis

### 8. Transition Type Detection
**Output:** `Dict` with transition types (cut/fade/wipe/dissolve) and parameters  
**Use Cases:**
- Video editing analysis
- Effect detection
- Quality assessment
- Content parsing

### 9. Visual Anomaly Detection
**Output:** `Dict` with anomaly events, scores, and affected regions  
**Use Cases:**
- Quality control
- Security monitoring
- Error detection
- Content moderation

### 10. Repetitive Pattern Classification
**Output:** `Dict` with pattern types, periods, and locations  
**Use Cases:**
- Animation detection
- Loading screen identification
- Pattern-based compression
- Content classification

### 11. Motion Coherence Patterns
**Output:** `Dict` with coherence scores and motion groups  
**Use Cases:**
- Object segmentation
- Crowd analysis
- Scene understanding
- Motion-based clustering

### 12. Structural Change Patterns
**Output:** `Dict` with change patterns, frequencies, and classifications  
**Use Cases:**
- Layout change detection
- Content evolution tracking
- Version comparison
- UI testing

---

## Feature Dependencies

### Required Analyses by Feature Level

**Basic Features typically require:**
- `frame_diff` - Most motion-based features
- `edge_canny` - Edge-based features
- `optical_flow_dense/sparse` - Flow-based features
- `background_mog2/knn` - Foreground detection
- `color_histogram` - Color-based features
- `dct_transform` - Frequency domain features
- `texture_descriptors` - Texture-based features

**Middle Features typically require:**
- Basic features as input
- Multiple raw analyses for correlation
- Temporal window processing
- Spatial region analysis

**Advanced Features typically require:**
- Multiple middle features
- Cross-feature correlation
- Machine learning models (when applicable)
- Domain-specific heuristics

---

## Usage Patterns

### For Screen Recording Analysis
```python
vk.add_feature('binary_activity')
vk.add_feature('ui_change_detection')
vk.add_feature('scrolling_detection')
vk.add_feature('app_window_switching')
```

### For Video Content Analysis
```python
vk.add_feature('scene_detection')
vk.add_feature('camera_movement')
vk.add_feature('shot_type_detection')
vk.add_feature('transition_type_detection')
```

### For Motion Analysis
```python
vk.add_feature('motion_magnitude')
vk.add_feature('motion_trajectories')
vk.add_feature('motion_pattern_classification')
vk.add_feature('activity_bursts')
```

### For Quality Assessment
```python
vk.add_feature('stability_score')
vk.add_feature('structural_similarity')
vk.add_feature('visual_anomaly_detection')
```

---

## Performance Considerations

### Lightweight Features (Fast)
- Binary Activity
- Motion Magnitude
- Edge Density
- Stability Score
- Foreground Ratio

### Medium Weight Features
- Blob Tracking
- Zone-Based Activity
- Activity Bursts
- Perceptual Hashes

### Heavyweight Features (Slow)
- Scene Detection
- Motion Pattern Classification
- Visual Anomaly Detection
- UI Change Detection

---

## Feature Selection Guidelines

1. **Start with Basic Features** - They're fast and often sufficient
2. **Add Middle Features** for structured pattern detection
3. **Use Advanced Features** only when specific detection is needed
4. **Consider Dependencies** - Some features auto-include required analyses
5. **Balance Performance** - More features = slower processing
6. **Test Incrementally** - Add features one at a time to measure impact