import torch, os, numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
name2cluster = {
    'root': [0],
    'lhip': [1], 'rhip': [2],
    'lknee': [4], 'rknee': [5],
    'lfoot': [7,10], 'rfoot': [8,11],
    'belly': [3], 'spine': [6],
    'chest-inshoulder-neck': [9,12,13,14],
    'head': [15], 
    'lshoulder-elbow': [16,18], 'rshoulder-elbow': [17,19],
    'lwrist-hand': [20,22], 'rwrist-hand': [21,23]}
name2res = {}
for name, ls in name2cluster.items():
    name2res[name] = {}


outputdir = 'experiments/human_nerf/zju_mocap/p387/tava/387_l1.0m0.2_4view_last2/latest'
res = torch.load(f'{outputdir}/name-2-3d.bin')

for i, (k, res_) in tqdm(enumerate(res.items())):
    img = np.zeros((512,512,3), dtype=np.uint8)
    xyz, rgb_on_image, weight_max, pos_on_image, lbs = res_.split([3,3,1,2,1],dim=1)
    for rgb, pos in zip(rgb_on_image.cpu(), pos_on_image.cpu()):
        rgb = torch.clip(rgb*255,0,255).int()
        r,c = int(pos[0].item()),int(pos[1].item())
        img[r,c,0], img[r,c,1], img[r,c,2] = rgb[2].item()//4, rgb[1].item()//4, rgb[0].item()//4  

    for name, ls in name2cluster.items():
        #lbs N, 1
        mask = torch.zeros_like(lbs)
        for i in ls:
            mask = mask+(lbs==i)
        if mask.max()==0:
            name2res[name][k] = None
            continue
        mask = mask.squeeze(1).bool() #M
        pos_mask = pos_on_image[mask] #M, 2
        
        #dilute
        #pos_on_image (N,1,2) - (1, M,2) (N,M,2) (N,M)
        dis = torch.sum(torch.abs(pos_on_image[:,None,:]-pos_mask[None,...]),axis=-1) #(N,M)
        min_dis = torch.min(dis, axis=1)[0] #N
        mask = min_dis<10

        res_seg = res_[mask]
        name2res[name][k] = res_seg

for name, res_ in name2res.items():
    torch.save(res_, os.path.join(outputdir, f'name-2-3d.{name}.bin'))
print("Save as ", os.path.join(outputdir, f'name-2-3d.{name}.bin'))
        # img_ = np.copy(img)
        # for rgb, pos in zip(rgb_on_image[mask].cpu(), pos_on_image[mask].cpu()):
        #     rgb = torch.clip(rgb*255,0,255).int()
        #     r,c = int(pos[0].item()),int(pos[1].item())
        #     img_[r,c,0], img_[r,c,1], img_[r,c,2] = rgb[2], rgb[1], rgb[0]
        # cv2.imwrite( f'debug_output/{k}_{name}.png',img_)
        #import ipdb; ipdb.set_trace()
        #debug


