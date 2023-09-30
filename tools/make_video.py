import numpy as np
from PIL import Image
import cv2
from glob import glob
inputdir = 'dataset/zju_mocap/387_tava/images/*.png'
video_name = 'dataset/zju_mocap/387_tava/video.mp4'
images_np = []
for imgname in glob(inputdir):
    img = cv2.imread(imgname) 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    maskname = imgname.replace('images','masks')
    mask = cv2.imread(maskname)
    img = cv2.resize(img, None, 
                        fx=0.5,
                        fy=0.5,
                        interpolation=cv2.INTER_LANCZOS4) #H,W,3
    mask = cv2.resize(mask, None, 
                            fx=0.5,
                            fy=0.5,
                            interpolation=cv2.INTER_LINEAR)  
    img = (img*(mask/255.)).astype(np.uint8)
    images_np.append(img)
image_stack = np.stack(images_np, axis=0)
imageio.mimwrite(video_name, image_stack, format='mp4', fps=10, quality=8)