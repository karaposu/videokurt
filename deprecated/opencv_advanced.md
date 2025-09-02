Spatiotemporal Pattern Analysis in Video with OpenCV (Python)

OpenCV provides a rich set of tools for low-level spatiotemporal pattern analysis in videos. These tools allow you to detect motion and pixel-level changes between frames, generate masks of moving regions, and highlight visual changes. Key techniques include optical flow (to estimate motion vectors), background subtraction (to segment moving objects), motion saliency detection (to highlight regions of change), and frame differencing (to detect pixel-wise changes). Below, we outline the relevant OpenCV Python APIs, their usage, parameters, and examples for each category, with links to official documentation and tutorials.

Optical Flow for Motion Detection

Optical flow computes the apparent motion of pixels between consecutive video frames. OpenCV supports both sparse optical flow (tracking specific feature points) and dense optical flow (estimating motion at every pixel):

Lucas-Kanade Optical Flow (Sparse): The function cv2.calcOpticalFlowPyrLK implements the iterative Lucas-Kanade method with image pyramids
docs.opencv.org
. You provide a set of 2D points in the first frame, and the function finds their new positions in the next frame. For example:

p1, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, p0, None, **lk_params)


This returns p1 (new point positions), a status array (1 if a point was found, 0 if lost), and an error array
docs.opencv.org
docs.opencv.org
. In practice, you detect good feature points (e.g. using cv2.goodFeaturesToTrack) in the first frame, then track them through subsequent frames using calcOpticalFlowPyrLK. Points with status=1 are successfully tracked; by filtering these you can plot motion trails or compute motion vectors
docs.opencv.org
. This method is suitable for tracking sparse keypoints (e.g. corners) across frames.

Farneback Optical Flow (Dense): The function cv2.calcOpticalFlowFarneback computes a dense optical flow field using Gunnar Farneback’s algorithm
docs.opencv.org
. It takes two consecutive grayscale frames (prev and next) and outputs a flow matrix (of type CV_32FC2) where each pixel has a 2D flow vector (horizontal and vertical displacement)
docs.opencv.org
. Typical usage:

flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 
                                   pyr_scale=0.5, levels=3, winsize=15, 
                                   iterations=3, poly_n=5, poly_sigma=1.2, flags=0)


Key parameters include pyr_scale (image scale < 1 for pyramid levels), levels (number of pyramid layers), winsize (window size for averaging, larger for more robust but smoother flow), iterations (iterations at each pyramid level), poly_n and poly_sigma (size and standard deviation for the pixel neighborhood polynomial expansion)
docs.opencv.org
docs.opencv.org
. The output flow is an array of shape (H, W, 2) containing the flow vector for each pixel. For example, flow[y,x] = (flow_x, flow_y) is the motion at pixel (x, y) from prev to next.

You can post-process the dense flow to analyze motion patterns. A common approach is to convert flow to polar coordinates to get motion magnitude and angle:

mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])


The magnitude mag gives the speed of motion per pixel, and ang gives the direction
docs.opencv.org
. By thresholding the magnitude, you can generate a motion mask – a binary image highlighting where significant motion occurred. Dense optical flow is useful for detecting and visualizing regions of motion, computing motion heatmaps, or identifying moving edges (e.g. for video stabilization and segmentation).

Documentation & Tutorials: The OpenCV tutorial Optical Flow provides an overview and code examples for both Lucas-Kanade tracking and Farneback dense flow
docs.opencv.org
docs.opencv.org
. The official reference for calcOpticalFlowFarneback details all parameters and flags
docs.opencv.org
docs.opencv.org
, and the reference for calcOpticalFlowPyrLK describes its inputs/outputs
docs.opencv.org
docs.opencv.org
.

Background Subtraction Methods

Background subtraction is a technique to model the static background of a scene and detect moving objects as foreground. It produces a foreground mask – a binary image where moving pixels are white (255) and static background pixels are black (0)
docs.opencv.org
. OpenCV’s video module provides two popular background subtraction algorithms in Python:

MOG2 (Gaussian Mixture Model): Create a subtractor using cv2.createBackgroundSubtractorMOG2. This algorithm models each pixel as a mixture of Gaussians to adapt to lighting changes and periodic motion (like tree leaves). For example:

backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)


By default, history=500 frames are used to learn the background model, varThreshold=16 is the threshold on Mahalanobis distance to decide if a pixel is foreground, and detectShadows=True enables shadow detection
docs.opencv.org
docs.opencv.org
. If shadow detection is on, moving shadows are marked in the mask (typically as gray values)
docs.opencv.org
docs.opencv.org
. After creation, you apply the subtractor to each frame:

fgMask = backSub.apply(frame)


The result fgMask is an 8-bit single-channel mask the same size as the frame, with white pixels indicating motion
docs.opencv.org
. MOG2 automatically updates the background model; you can optionally pass a learning rate to apply() if you need to adjust how quickly the model adapts
docs.opencv.org
.

KNN (K-Nearest Neighbors): Create a subtractor with cv2.createBackgroundSubtractorKNN, which uses a KNN approach to model background pixels. Usage is analogous to MOG2:

backSub = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400.0, detectShadows=True)


Here dist2Threshold=400.0 is the threshold on the squared distance to decide foreground (larger means more tolerance for change)
docs.opencv.org
docs.opencv.org
. Like MOG2, history controls the number of frames for background and detectShadows toggles shadow marking. You then call backSub.apply(frame) to get the foreground mask.

Both algorithms yield a foreground mask that can be used for motion segmentation – e.g. finding blobs of moving regions. You can refine the mask with morphological operations (erosion/dilation) to remove noise. The BackgroundSubtractor object can also be tuned (e.g. by adjusting the learning rate in apply or using methods like backSub.setVarThreshold() in MOG2 if needed). The OpenCV tutorial How to Use Background Subtraction provides a complete example of using MOG2/KNN in a video loop
docs.opencv.org
docs.opencv.org
. The official documentation details the parameters for createBackgroundSubtractorMOG2 and createBackgroundSubtractorKNN, including how detectShadows can be disabled for speed
docs.opencv.org
docs.opencv.org
.

Motion Saliency Detection

OpenCV’s saliency module (part of opencv_contrib) contains a motion saliency algorithm that automatically highlights regions of change over time. The primary interface is Bin Wang Apr 2014 motion saliency (named after the authors’ method):

Motion Saliency (BinWangApr2014): Create the detector with cv2.saliency.MotionSaliencyBinWangApr2014_create(). This returns a saliency object with a computeSaliency method. Important: you must initialize the model with the frame dimensions before use. Call saliency.setImagesize(width, height) to set the frame size, then saliency.init() to allocate internal buffers
pyimagesearch.com
docs.opencv.org
. For each new frame (typically grayscale), call:

success, saliencyMap = saliency.computeSaliency(gray_frame)


The result saliencyMap is a single-channel floating-point mask the size of the frame, where salient motion regions are highlighted (the algorithm produces a binary-ish map of moving areas)
docs.opencv.org
. success is a boolean indicating if the computation was successful. In practice, the saliencyMap values are in [0,1] – you can multiply by 255 and convert to uint8 for a binary visualization
pyimagesearch.com
pyimagesearch.com
. This motion saliency technique is essentially a fast self-tuning background subtraction that automatically detects moving regions or changes in a video
docs.opencv.org
. It is useful for scenarios where you want to find regions of movement without manually tweaking parameters; however, it may be sensitive to even slight motions (including noise) as noted by users.

OpenCV’s saliency module also includes static saliency detectors (for still images) and an objectness proposal algorithm, but for purely visual motion segmentation the BinWangApr2014 motion saliency is the relevant one. Make sure you have OpenCV compiled with contrib modules to use cv2.saliency. The official documentation (Saliency API) describes the Motion Saliency algorithm and its usage
docs.opencv.org
docs.opencv.org
, and tutorials like the PyImageSearch guide on OpenCV saliency provide Python examples of using MotionSaliencyBinWangApr2014_create in a video processing loop
pyimagesearch.com
.

Frame Differencing and Pixel-Level Change Detection

The most straightforward way to detect visual changes between frames is frame differencing – computing the absolute difference between successive frames or between a frame and a reference background frame. In OpenCV, you can use cv2.absdiff for this:

Frame differencing: Given two frames frame1 and frame2 (e.g. consecutive frames or a current frame and a running average background), cv2.absdiff(frame1, frame2) produces an image where each pixel is the absolute difference of the two at that location
docs.opencv.org
. In code:

diff = cv2.absdiff(frame1, frame2)


This diff image highlights pixel-level changes – regions that differ will have non-zero intensity. To segment changes, you can convert to grayscale (if not already) and apply a threshold: for example, _, mask = cv2.threshold(diff_gray, thresh, 255, cv2.THRESH_BINARY) will produce a binary mask of pixels that changed by more than a certain amount. Small thresh catches even minor flicker, while larger values focus on significant changes. Frame differencing is useful for detecting fades (gradual illumination changes will show as global differences frame-to-frame) and flicker or sudden scene changes (which produce high difference values). However, it can be noisy – combining it with smoothing or accumulating differences over a few frames can improve robustness.

Frame differencing can be seen as a simple form of background subtraction with no adaptive model – it’s essentially |frame_t - frame_{t-1}|. Despite its simplicity, it’s a valuable tool in an analysis pipeline: for instance, you might first use absdiff to quickly flag where any change occurs, and then refine those regions using optical flow or a background subtractor for a more stable mask.

Additional Notes and Usage Tips

Input Formats: Many of these functions expect grayscale frames for proper operation. For optical flow and motion saliency, convert frames to gray with cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
docs.opencv.org
pyimagesearch.com
. Background subtractors can work with color frames directly (they handle each channel), but often perform just as well on grayscale.

Output Masks and Visualization: The output of background subtractors (fgMask) and motion saliency (saliencyMap) are essentially binary masks of change. You can overlay these masks on the original video (e.g. using cv2.bitwise_and or coloring the mask) to visualize moving regions. For optical flow, you can draw flow vectors (as arrows or lines) or convert the flow field to an HSV image for visualization
docs.opencv.org
docs.opencv.org
 – where hue represents direction and value represents magnitude.

Performance: These methods are suitable for offline processing, but keep in mind computational cost. Dense optical flow (Farneback) is more expensive than Lucas-Kanade tracking of a few points. Background subtractors and frame differencing are relatively fast. Motion saliency is designed to be efficient, but its performance will depend on frame size and the amount of motion. For analyzing long videos, you may need to tune parameters (like the learning rate in background subtractors or the threshold for differences) to balance sensitivity and noise.

By leveraging these OpenCV functions – optical flow for motion vectors, background subtraction for foreground masks, motion saliency for automatic change detection, and frame differencing for simple change segmentation – you can build a robust pipeline for detecting low-level visual primitives in video. Each comes with comprehensive documentation and examples: see OpenCV’s official guides on optical flow
docs.opencv.org
docs.opencv.org
 and background subtraction
docs.opencv.org
docs.opencv.org
, and the API references for the exact function signatures and parameter meanings (e.g. optical flow in the video/tracking module
docs.opencv.org
docs.opencv.org
, background subtractor in the video/motion module
docs.opencv.org
docs.opencv.org
, and saliency in the saliency module
docs.opencv.org
). Using these tools, you can detect visual events like fades, flickers, slides, or region animations by analyzing how pixel intensities and regions change over time, all within the OpenCV Python framework.