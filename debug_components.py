from videokurt import VideoKurt
import traceback

vk = VideoKurt()
vk.configure(frame_step=10, resolution_scale=0.3)
vk.add_feature('connected_components')

try:
    results = vk.analyze('sample_recording.MP4')
    print("Success!")
except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()