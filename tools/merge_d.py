import numpy as np
import os

# filepath = 'experiments/human_nerf/zju_mocap/p387/tava/387_l1.0m0.2_4view_last2/latest/' + \
#     'distance_mat_3/distance_mat_0.30-0.00.{}-16.npy'
# D = []
# for i in range(16):
#     try:
#         d = np.load(filepath.format(i))
#         D.append(d) #N,N
#     except:
#         print(f'Error {i}')
# D = sum(D) #N,N
# np.save(filepath.replace('.{}-16',''),D)
# print(D)
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

from tqdm import tqdm
for seg in tqdm(name2cluster):
    filepath = 'experiments/human_nerf/zju_mocap/p387/tava/387_l1.0m0.2_4view_last2/latest/' + \
        'distance_mat_seg/'+seg+'/distance_mat_0.30-0.00.{}-8.npy'
    D = []
    for i in range(8):
        try:
            d = np.load(filepath.format(i))
            D.append(d) #N,N
            #import ipdb; ipdb.set_trace()
        except:
            print(f'Error {seg}-{i}')
    D = sum(D) #N,N
    np.save(filepath.replace('.{}-8',''),D)
    print(D)
'experiments/human_nerf/zju_mocap/p387/tava/387_l1.0m0.2_4view_last2/latest/distance_mat/'

