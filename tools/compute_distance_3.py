import numpy as np
import os, pickle
from tqdm import tqdm
from time import time
import torch
import sys
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

def compute_distance_gpu(name0, name1, dist_thresh=0.002, valid_weight_threshold=0.3):
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
    return distance

start = time()
DIRNAME = 'experiments/human_nerf/zju_mocap/p387/tava/387_l1.0m0.2_4view_last2/latest/'
framename2info = torch.load(f'{DIRNAME}/name-2-3d.bin')
print('Load info file ', time()-start); start = time()

framenames = [f for f in sorted(framename2info.keys())]
N = len(framenames)
D = np.zeros([N,N], dtype=np.float32)

chunk_id, chunk_n = 3, 8 #int(sys.argv[1]), int(sys.argv[2])
chunk_size = N//chunk_n

idx = np.arange(chunk_id, N, chunk_n)
if chunk_id==chunk_n-1:
    idx = np.concatenate([idx,np.arange(idx[-1]+1, N)],axis=0)
#hyper-param
hyper_param = {'valid_weight_threshold':0.3, 'dist_thresh':0.002}



chunk_id, chunk_n = int(sys.argv[1]), int(sys.argv[2])
N3 = len(idx) #!!
chunk_size = N3//chunk_n
idx_3 = np.arange(chunk_id, N3, chunk_n)
if chunk_id==chunk_n-1:
    idx_3 = np.concatenate([idx_3,np.arange(idx_3[-1]+1, N3)],axis=0)
#hyper-param
hyper_param = {'valid_weight_threshold':0.3, 'dist_thresh':0.002}

idx = idx[idx_3]

print(f"{chunk_id}/{chunk_n} {idx[:3]}-{idx[-3:]} num={len(idx)}")

for i  in tqdm(idx):
    for j in tqdm(range(i+1, N)):
        d = compute_distance_gpu(name0=framenames[i], name1=framenames[j], **hyper_param)
        D[i,j] = d.item()
        D[j,i] = d.item() 

outputfile = os.path.join(DIRNAME, 'distance_mat_3',
    f"distance_mat_{hyper_param['valid_weight_threshold']:.2f}-{hyper_param['dist_thresh']:.2f}.{chunk_id}-{chunk_n}.npy")
os.makedirs(os.path.dirname(outputfile), exist_ok=True)
print('Save as '+ outputfile)
np.save(outputfile, D)



# for name in framename2info:
#     mask = framename2info[name][:,-1]>valid_weight_threshold
#     xyzs, rgbs, ws = framename2info[name][mask].split([3,3,1],dim=1)
#     with open(f'debug-{name}.obj','w') as f:
#         for xyz,rgb in zip(xyzs, rgbs):
#             rgb = rgb.clip(0,1)
#             f.writelines(f'v {xyz[0]:.7f} {xyz[1]:.7f} {xyz[2]:.7f} {rgb[0]:.7f} {rgb[1]:.7f} {rgb[2]:.7f}\n')