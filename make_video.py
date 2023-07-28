import imageio
from PIL import Image
from glob import glob
import numpy as np
subject_id = 387
image_path = '/mnt/workspace/chenyutong/code/humannerf/dataset/zju_mocap/{}/images/frame_*.png'
video_path = '/mnt/workspace/chenyutong/code/humannerf/dataset/zju_mocap/{}/video.mp4'
image_path = image_path.format(subject_id)
video_path = video_path.format(subject_id)

images = []
for filepath in sorted(glob(image_path.format(subject_id))):
    image = Image.open(filepath)
    images.append(np.array(image))
images = np.stack(images, axis=0)
print('Save as', video_path)
imageio.mimwrite(video_path, images, format='mp4', fps=10, quality=8)

