
# Different ways how VideoKurt is can bed used

videokurt interface works in such way 


from videokurt import VideoKurt


vk = VideoKurt()

vk.add_analysis(analysis_name_here)
vk.add_analysis(another_analysis_name_here)

this way we tell videokurt which analysis we are interested. 

vk.configure(frame_step=5, resolution_scale=0.2, )  



Simple usage - just enable blur with defaults
vk.configure(blur=True)  # Uses kernel_size=13


vk.configure(blur=True, blur_kernel_size=21)  # Stronger blur

VideoKurt provides three preprocessing techniques that can be applied to any analysis:

  1. Downsampling (Temporal Reduction)

  - What: Skip frames to reduce processing load
  - Parameter: frame_step=N (process every Nth frame)
  - Example: frame_step=3 → process frames 0, 3, 6, 9...
  - Use when: Video has high frame rate or redundant frames

  2. Downscaling (Spatial Reduction)

  - What: Reduce resolution of each frame
  - Parameter: resolution_scale=0.X (fraction of original size)
  - Example: resolution_scale=0.5 → 1920×1080 becomes 960×540
  - Use when: High resolution isn't needed for detection

  3. Blur (Detail Reduction)

  - What: Apply Gaussian blur to remove fine details
  - Parameters: blur=True/False, blur_kernel_size=N (odd number)
  - Example: blur=True, blur_kernel_size=13 → smooth out text/noise
  - Use when: Small details interfere with motion detection


vk.configure(
      frame_step=2,         # Half the frames
      resolution_scale=0.5, # Quarter the pixels
      blur=True            # Remove noise
  )
  Result: 8x faster processing with cleaner motion detection


analysis_results= vk.analyze(path_to_the_video)
this is valid usecase, it will take so much space in RAM but we let user do that. 

vk.analyze(path_to_the_video)


and then we should be able to list the features we are interested in 
vk.add_feature(feature_name_here)
vk.add_feature(feature_name_here)

and then do sth like vk.get_features() to get the features 


but also we need a way to directly go for features, and if features are the end result we can dismiss the raw analysis

also we should be able to import each raw analysis and custom configure it and use it. 
i guess 

vk.add_analysis(analysis_name_here)
vk.add_analysis(another_analysis_name_here)


can be also 


vk.add_analysis(analysis_object_here)
in videokurt/models.py  RawAnalysis class for this purpose



it is important we define this interface good. \
  \
  help me clarify things by writing your suggestions in interface_discussion.md\
  \
  \
  but start interface_discussion.md with listing all things the this interface support , if we have that list our job will be 
  easier. \





