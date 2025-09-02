# VideoKurt Core Concepts

## Essential Technical Concepts (Ordered by Importance/Dependency)

1. **Frame Differencing** - Pixel-level comparison between video frames to detect visual changes
2. **Binary Activity Timeline** - Classification of time periods as either active or inactive
3. **Event Detection** - Identification of specific visual patterns (scroll, click, scene change)
4. **Calibration System** - Tunable thresholds and parameters for different video contexts
5. **Confidence Scoring** - Numerical certainty measure for each detection (0.0-1.0)
6. **Temporal Segmentation** - Division of video into logical chunks based on activity density
7. **Frame Stepping** - Sampling strategy using frame_step parameter (1=all frames, 2=every 2nd, etc.)
8. **Processing Modes** - Speed/accuracy trade-off configurations (fast/balanced/thorough)