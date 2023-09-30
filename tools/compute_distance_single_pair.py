import numpy as np
import os, pickle
from tqdm import tqdm
from time import time
import torch
import sys, cv2
def find_nearest_pair_gpu(pts1, pts2):
    start = time()
    dist = torch.linalg.norm(pts1[:,None,:]-pts2[None,:,:], axis=-1) # N1*N2 
    #print('Compute diff', time()-start); start = time()

    min0, min1 = torch.argmin(dist, axis=1), torch.argmin(dist, axis=0) # N1*logN2+N2*logN1
    #print('Compute argmin', time()-start); start = time()

    min01 = min1[min0]
    #print('gather', time()-start); start = time()
    pair_0 = (min01==torch.arange(pts1.shape[0], device=dist.device)).nonzero()#[0]
    #print('nonzero', time()-start); start = time()
    pair_1 = min0[pair_0]
    #print('gather', time()-start); start = time()
    return pair_0, pair_1, dist

def compute_distance_gpu(name0, name1, dist_thresh=0.002, valid_weight_threshold=0.3, 
        output_errormap_path=None, ):
    # xyzs0, xyzs1 = framename2info[name0]['xyzs'], framename2info[name1]['xyzs']
    # rgbs0, rgbs1 = framename2info[name0]['rgbs'], framename2info[name1]['rgbs']
    mask0 = framename2info[name0][:,-1]>valid_weight_threshold
    mask1 = framename2info[name1][:,-1]>valid_weight_threshold
    xyzs0, rgbs0, ws0 = framename2info[name0][mask0].split([3,3,1],dim=1)
    xyzs1, rgbs1, ws1 = framename2info[name1][mask1].split([3,3,1],dim=1)
    #poss0, poss1 = framename2info[name0]['poss'], framename2info[name1]['poss']
    #ind0, ind1 = framename2info[name0]['ind'], framename2info[name1]['ind']
    start = time()
    pair_0, pair_1, dist = find_nearest_pair_gpu(xyzs0,xyzs1)
    #print('Find nn ', time()-start)
    
    start = time()
    rgb_errors = torch.tensor([torch.linalg.norm(rgbs0[p0]-rgbs1[p1]) for p0,p1 in zip(pair_0, pair_1)], device=mask0.device) #
    nearest_dist = torch.tensor([dist[p0,p1] for p0,p1 in zip(pair_0, pair_1)], device=mask0.device)
    #print(nearest_dist.min())
    distance = torch.sum(rgb_errors*(nearest_dist<dist_thresh))
    #print('Compute dist ', time()-start)
    start = time()   
    
    if output_errormap_path is not None:
        os.makedirs(output_errormap_path, exist_ok=True)
        mask = nearest_dist<dist_thresh
        e = np.clip((rgb_errors[mask]).cpu().numpy()*255, 0, 255).astype(np.uint8)
        e = cv2.applyColorMap(e, cv2.COLORMAP_JET)[:,0,:] #N,3
        xyzs0 = [xyzs0[p0][0].cpu().numpy() for p0 in pair_0[mask]]
        xyzs1 = [xyzs1[p1][0].cpu().numpy() for p1 in pair_1[mask]]

        import ipdb; ipdb.set_trace()
        with open(os.path.join(output_errormap_path,f'{name0}-{name1}.obj'),'w') as f:
            for xyzs in [xyzs0, xyzs1]:
                for xyz, bgr in zip(xyzs, e):
                    bgr = bgr/255
                    f.writelines(f'v {xyz[0]:.7f} {xyz[1]:.7f} {xyz[2]:.7f} {bgr[2]:.7f} {bgr[1]:.7f} {bgr[0]:.7f}\n')
        print('Output as ',os.path.join(output_errormap_path,f'{name0}-{name1}.obj'))
    return distance

start = time()
DIRNAME = 'experiments/human_nerf/zju_mocap/p387/tava/387_l1.0m0.2_4view_last2/latest/'
framename2info = torch.load(f'{DIRNAME}/name-2-3d.bin')
print('Load info file ', time()-start); start = time()

framenames = [f for f in sorted(framename2info.keys())]
N = len(framenames)
i,j=370,0
name0, name1=framenames[i], framenames[j]
hyper_param = {'valid_weight_threshold':0.3, 'dist_thresh':0.002}
D = np.zeros([N,N], dtype=np.float32)
d = compute_distance_gpu(name0=name0, name1=name1, 
        output_errormap_path=DIRNAME+f"/distance_mat_{hyper_param['valid_weight_threshold']:.2f}-{hyper_param['dist_thresh']:.2f}", 
        **hyper_param)
D[i,j] = d.item()
D[j,i] = d.item() 

for k in [i,j]:
    name = framenames[k]
    mask = framename2info[name][:,-1]>hyper_param['valid_weight_threshold']
    xyzs, rgbs, ws = framename2info[name][mask].split([3,3,1],dim=1)
    with open(f'debug_{k}.obj','w') as f:
        for xyz, rgb in zip(xyzs, rgbs):
            f.writelines(f'v {xyz[0]:.7f} {xyz[1]:.7f} {xyz[2]:.7f} {rgb[0]:.7f} {rgb[1]:.7f} {rgb[2]:.7f}\n')







# for name in framename2info:
#     mask = framename2info[name][:,-1]>valid_weight_threshold
#     xyzs, rgbs, ws = framename2info[name][mask].split([3,3,1],dim=1)
#     with open(f'debug-{name}.obj','w') as f:
#         for xyz,rgb in zip(xyzs, rgbs):
#             rgb = rgb.clip(0,1)
#             f.writelines(f'v {xyz[0]:.7f} {xyz[1]:.7f} {xyz[2]:.7f} {rgb[0]:.7f} {rgb[1]:.7f} {rgb[2]:.7f}\n')