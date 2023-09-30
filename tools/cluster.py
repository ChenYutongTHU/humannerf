import numpy as np
import torch, os
from tqdm import tqdm
import pickle

D_file = 'experiments/human_nerf/zju_mocap/p387/tava/387_l1.0m0.2_4view_last2/latest/distance_mat/distance_mat_0.30-0.00.npy'
image_dir = 'dataset/zju_mocap/387_tava/images'
output_file = D_file.replace('.npy','.cluster.pkl')
N_cluster = 4


names = sorted(os.listdir(image_dir))
d_np = np.load(D_file)
D = torch.tensor(d_np, device='cuda:0') #N,N
N = D.shape[0]
M = N//N_cluster #one-fourth

# print('DEBUG mode!! Please comment the following lines')
# D = torch.where(D==0, float("inf") ,D)

clustered = []

results = []
for n in range(N_cluster):
    print(f'Create cluster-{n}')
    seeds = [[i for i in range(N) if not i in clustered][0]]
    print('Seeds: ',[names[s] for s in seeds])
    M_to_add = M-len(seeds)
    #init
    dist2cluster,_ = torch.max(torch.stack([D[s,:] for s in seeds],dim=0),dim=0) #n, N (N,) #MAX!
    #import ipdb; ipdb.set_trace()
    dist2cluster[seeds] = float("inf")
    dist2cluster[clustered] = float("inf")

    total_dist = []
    #add new sample one-by-one
    for step in range(M_to_add):
        i = torch.argmin(dist2cluster) #N,  MIN!
        seeds.append(i.item())
        total_dist.append(dist2cluster[i].item())

        dist2new = D[i,:]
        #import ipdb; ipdb.set_trace()
        dist2cluster, _ = torch.max(torch.stack([dist2new, dist2cluster], dim=0), dim=0) #2,N -> N #MAX!
        dist2cluster[seeds] = float("inf")

    print(f'Finish cluster-{n} total-dist={np.sum(total_dist):.2f}\n')
    C = {'names':[names[s] for s in seeds], 'dist':total_dist}
    clustered.extend(seeds)
    results.append(C)
    # print('+', names[i], i, total_dist[-1])
    # import ipdb; ipdb.set_trace()
with open(output_file,'wb') as f:
    pickle.dump(results, f)









#init
