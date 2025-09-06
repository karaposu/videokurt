"""Example showing how to use BinaryActivity feature with VideoKurt."""

# to run python videokurt/smoke_tests/feat_test_binary_activity.py

from videokurt import VideoKurt

# Method 1: Simple usage - auto-configure required analyses
vk = VideoKurt()

# Add binary activity feature - this will automatically configure frame_diff analysis
vk.add_feature('binary_activity', threshold=30.0, activity_threshold=0.1)

# Process video
results = vk.analyze('sample_recording.MP4')

# Access the binary activity timeline
binary_activity = results.features['binary_activity'].data
print(f"Binary activity shape: {binary_activity.shape}")
print(f"Activity detected in {binary_activity.sum()} frames out of {len(binary_activity)}")

# # The binary_activity array is 1 where motion is detected, 0 otherwise
# # You can use it to find active segments:
# import numpy as np

# # Find transitions (start and end of active segments)
# diff = np.diff(np.concatenate(([0], binary_activity, [0])))
# starts = np.where(diff == 1)[0]
# ends = np.where(diff == -1)[0] - 1

# print(f"Found {len(starts)} active segments")
# for start, end in zip(starts, ends):
#     print(f"  Frames {start}-{end}: duration {end-start+1} frames")


# # Method 2: Explicit configuration with custom parameters
# vk2 = VideoKurt()

# # Manually add frame_diff analysis (which BinaryActivity requires)
# vk2.add_analysis('frame_diff')

# # Add binary activity with custom thresholds
# vk2.add_feature('binary_activity', 
#                 threshold=50.0,  # Higher threshold = less sensitive
#                 activity_threshold=0.2)  # 20% of pixels must change

# results2 = vk2.process('path/to/video.mp4')


# # Method 3: Multiple features that share analyses
# vk3 = VideoKurt()

# # Add multiple features that use frame_diff
# vk3.add_feature('binary_activity')
# vk3.add_feature('stability_score')  # Also uses frame_diff
# vk3.add_feature('frame_difference_percentile')  # Also uses frame_diff

# # The frame_diff analysis will only run once and be shared
# results3 = vk3.process('path/to/video.mp4')

# # Access all features
# activity = results3.features['binary_activity'].data
# stability = results3.features['stability_score'].data  
# percentiles = results3.features['frame_difference_percentile'].data

# print(f"Computed {len(results3.features)} features")
# print(f"Using {len(results3.analyses)} raw analyses")


# # Method 4: Batch processing with progress
# vk4 = VideoKurt()
# vk4.add_feature('binary_activity')

# videos = ['video1.mp4', 'video2.mp4', 'video3.mp4']
# all_activities = []

# for video_path in videos:
#     print(f"Processing {video_path}...")
#     results = vk4.process(video_path)
#     activity = results.features['binary_activity'].data
#     all_activities.append(activity)
    
#     # Quick summary
#     activity_ratio = activity.sum() / len(activity)
#     print(f"  {video_path}: {activity_ratio:.1%} active frames")