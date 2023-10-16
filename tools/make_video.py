import numpy as np
from PIL import Image
import cv2, os
from glob import glob
import imageio
dirs = 'experiments/human_nerf/zju_mocap/p387/tava/387_l1.0m0.2_4view_mat_len{}_2mlp/latest_inter_1_529/train_render/'
images_np = []
video_name = 'debug_output/interpolate_387.mp4'
for imgname in sorted(os.listdir(dirs.format(1))):
    image = []
    for len_ in [1,2,4]:
        dir_ = dirs.format(len_)
        img = cv2.imread(os.path.join(dir_,imgname)) 
        image.append(img[:,:,[2,1,0]])
    image = np.concatenate(image, axis=1)
    images_np.append(image)
image_stack = np.stack(images_np, axis=0)
imageio.mimwrite(video_name, image_stack, format='mp4', fps=10, quality=8)