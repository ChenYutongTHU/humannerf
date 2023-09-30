import pickle, torch, numpy as np
from matplotlib import pyplot as plt
import os
from PIL import Image
from collections import defaultdict
import cv2
from time import time

def find_nearest_pair_gpu_import(pts1, pts2):
    start = time()
    dist = torch.linalg.norm(pts1[:,None,:]-pts2[None,:,:], axis=-1) # N1*N2 
    #print('Compute diff', time()-start); start = time()

    #print('Compute argmin', time()-start); start = time()
    dist01, min0 = torch.min(dist, axis=1)
    dist10, min1 = torch.min(dist, axis=0)

    min01 = min1[min0]
    #print('gather', time()-start); start = time()
    pair_0 = (min01==torch.arange(pts1.shape[0], device=dist.device)).nonzero()#[0]
    #print('nonzero', time()-start); start = time()
    pair_1 = min0[pair_0]
    #print('gather', time()-start); start = time()
    return pair_0, pair_1, dist01, dist10, dist

def compute_distance_gpu(name0, name1, dist_thresh=0.002, valid_weight_threshold=0.3):
    # xyzs0, xyzs1 = framename2info[name0]['xyzs'], framename2info[name1]['xyzs']
    # rgbs0, rgbs1 = framename2info[name0]['rgbs'], framename2info[name1]['rgbs']
    if framename2info[name0] is None or framename2info[name1] is None:
        return 0
    mask0 = framename2info[name0][:,6]>valid_weight_threshold 
    mask1 = framename2info[name1][:,6]>valid_weight_threshold
    xyzs0, rgbs0, ws0, uvs0, _ = framename2info[name0][mask0].split([3,3,1,2,1],dim=1)
    xyzs1, rgbs1, ws1, uvs1, _ = framename2info[name1][mask1].split([3,3,1,2,1],dim=1)
    #poss0, poss1 = framename2info[name0]['poss'], framename2info[name1]['poss']
    #ind0, ind1 = framename2info[name0]['ind'], framename2info[name1]['ind']

    pair_0, pair_1, dist01, dist10, dist = find_nearest_pair_gpu_import(xyzs0,xyzs1)
    #print('Find nn ', time()-start)
    

    rgb_errors = torch.tensor([torch.linalg.norm(rgbs0[p0]-rgbs1[p1]) for p0,p1 in zip(pair_0, pair_1)], device=mask0.device) #
    nearest_dist = torch.tensor([dist[p0,p1] for p0,p1 in zip(pair_0, pair_1)], device=mask0.device)
    #print(nearest_dist.min())
    distance = torch.sum(rgb_errors*(nearest_dist<dist_thresh))
    #print('Compute dist ', time()-start)

    results = {'distance':distance, 
            'xyz_pairs':[[xyzs0[p0], xyzs1[p1]] for p0,p1 in zip(pair_0, pair_1)],
            'rgb_pairs':[[rgbs0[p0], rgbs1[p1]] for p0,p1 in zip(pair_0, pair_1)],
            'pair_0':pair_0, 'pair_1':pair_1,
            'dist01':dist01, 
            'dist10':dist10, 'dist':dist}
    return results

seg='chest-inshoulder-neck'
D_file = f'experiments/human_nerf/zju_mocap/p387/tava/387_l1.0m0.2_4view_last2/latest/distance_mat_seg/{seg}/distance_mat_0.30-0.00.npy'
image_dir = 'dataset/zju_mocap/387_tava/images'
output_file = D_file.replace('.npy','.cluster.pkl')
names = sorted(os.listdir(image_dir))
d_np = np.load(D_file)

DIRNAME = 'experiments/human_nerf/zju_mocap/p387/tava/387_l1.0m0.2_4view_last2/latest/'
framename2info = torch.load(f'{DIRNAME}/name-2-3d.{seg}.bin')

def get_frame_view(n):
    n = n.split('.png')[0]
    frame, view = n.split('_view_')
    frame = frame.split('frame_')[1]
    return int(frame), int(view)

def find_nearest_pair_gpu(pts1, pts2):
    dist = torch.linalg.norm(pts1[:,None,:]-pts2[None,:,:], axis=-1) # N1*N2 
    #print('Compute diff', time()-start); start = time()

    #print('Compute argmin', time()-start); start = time()
    dist01, min0 = torch.min(dist, axis=1)
    dist10, min1 = torch.min(dist, axis=0)
    min01 = min1[min0]
    #print('gather', time()-start); start = time()
    pair_0 = (min01==torch.arange(pts1.shape[0], device=dist01.device)).nonzero()#[0]
    #print('nonzero', time()-start); start = time()
    pair_1 = min0[pair_0]
    #print('gather', time()-start); start = time()
    return pair_0, pair_1, dist, dist01, dist10

def write_obj(xyzs, outputfile):

def warp(f0,v0,f1,v1):
    name0 = f'frame_{f0:06d}_view_{v0:02d}.png'
    name1 = f'frame_{f1:06d}_view_{v1:02d}.png'
    i0 = names.index(name0)
    i1 = names.index(name1)
    pc = d_np[i0,i1]

    valid_weight_threshold = 0
    dist_thresh = 1

    dirname = 'experiments/human_nerf/zju_mocap/p387/tava/387_l1.0m0.2_4view_last2/latest/train_render_3d_bkup/frame_{:06d}_view_{:02d}-rays.pkl'
    infos = []
    rgb_error_thresh = 0
    imgs = []
    for frame, view in ([f0,v0],[f1,v1]):

        # rays = pickle.load(open(dirname.format(frame,view),'rb'))
        # mask = np.max(rays['weights_on_rays'],axis=1)>valid_weight_threshold
        # xyz_on_image = np.sum(rays['xyz_on_rays']*rays['weights_on_rays'][...,None], axis=1)
        # xyz_on_image = xyz_on_image[mask]
        # rgb_predicted = np.sum(rays['rgb_on_rays']*rays['weights_on_rays'][...,None], axis=1)
        # rgb_predicted = rgb_predicted[mask]
        # pos_on_img = rays['pos_on_image'][mask]
        # rgb_on_img = rays['rgb_on_image'][mask]

        #
        name = f'frame_{frame:06d}_view_{view:02d}'
        xyzs0, rgbs0, ws0, uvs0, _ = framename2info[name].split([3,3,1,2,1],dim=1)
        mask = framename2info[name][:,6]>valid_weight_threshold 
        uvs0, rgbs0 = uvs0[mask], rgbs0[mask]
        xyzs0 = xyzs0[mask]
        img = np.zeros([512,512,3], dtype=np.uint8)
        for (r,c), rgb in zip(uvs0.cpu().numpy(), rgbs0.cpu().numpy()):
            img[int(r),int(c)] = (np.clip(rgb,0,1)*255).astype(np.uint8)
        imgs.append(img)

        infos.append({'xyzs':torch.tensor(xyzs0,device='cuda:0'), 
                    'rgbs': torch.tensor(rgbs0, device='cuda:0'), 
                    'rcs':uvs0.cpu().numpy().astype(np.int32)})
        


    # pair_0, pair_1, dist, dist01, dist10 = find_nearest_pair_gpu(
    #     infos[0]['xyzs'],
    #     infos[1]['xyzs'])
    results = compute_distance_gpu(name0[:-4], name1[:-4], dist_thresh=dist_thresh, valid_weight_threshold=valid_weight_threshold)
    pc = results['distance']
    pair_0, pair_1 = results['pair_0'], results['pair_1']
    dist01, dist10, dist = results['dist01'], results['dist10'],  results['dist']
    

    img01 = np.zeros([512,512,3], dtype=np.uint8)
    img10 = np.zeros([512,512,3], dtype=np.uint8)
    img0 = np.zeros([512,512,3], dtype=np.uint8)
    img1 = np.zeros([512,512,3], dtype=np.uint8)
    error01 = np.zeros([512,512,3], dtype=np.uint8)
    error10 = np.zeros([512,512,3], dtype=np.uint8)
    rgb_errors = np.array([torch.linalg.norm(infos[0]['rgbs'][p0.item()]-infos[1]['rgbs'][p1.item()]).item() \
            for p0, p1 in zip(pair_0, pair_1)])
    rgb_errors = rgb_errors*(rgb_errors>rgb_error_thresh)
    rgb_errors = np.clip(rgb_errors[...,None]*255, 0, 255).astype(np.uint8)
    rgb_errors = cv2.applyColorMap(rgb_errors, cv2.COLORMAP_JET)[:,0,:] #N,3


    max_dist = 0.02
    for i ,dd in ([0, dist01.cpu().numpy()],[1,dist10.cpu().numpy()]):
        dist_map = np.zeros([512,512,3], dtype=np.float32)
        mask = np.zeros([512,512,3], dtype=np.uint8)
        for (r,c), d in zip(infos[i]['rcs'], dd):
            dist_map[r,c] = np.clip(d.item(), 0, max_dist) #1-1cm
            mask[r,c] = 1
        dist_map = (dist_map/max_dist*255).astype(np.uint8)
        dist_map = cv2.applyColorMap(dist_map, cv2.COLORMAP_JET)[:,:,[2,1,0]]
        if i==0:
            dist01_map = dist_map*mask
        else:
            dist10_map = dist_map*mask

    for p0, p1, e in zip(pair_0, pair_1, rgb_errors): #warp img0 to img1
        r1,c1 = infos[1]['rcs'][p1.item()]
        r0,c0 = infos[0]['rcs'][p0.item()]
        rgb0 = infos[0]['rgbs'][p0.item()].cpu().numpy()
        rgb1 = infos[1]['rgbs'][p1.item()].cpu().numpy()
        if dist[p0,p1]<dist_thresh:
            #print(f'{infos[0]["rcs"][p0]}->{infos[1]["rcs"][p1]}')
            img01[r1,c1] = (np.clip(rgb0,0,1)*255).astype(np.uint8)
            img10[r0,c0] = (np.clip(rgb1,0,1)*255).astype(np.uint8)
            # img0[r0,c0] = (np.clip(rgb0,0,1)*255).astype(np.uint8)
            # img1[r1,c1] = (np.clip(rgb1,0,1)*255).astype(np.uint8)
            error01[r1,c1] = e[[2,1,0]]
            error10[r0,c0] = e[[2,1,0]]
    stack_img01 = np.concatenate([imgs[0], imgs[1], img01, error01, dist10_map], axis=1)
    stack_img10 = np.concatenate([imgs[1], imgs[0], img10, error10, dist01_map], axis=1)
    o01 = f'debug_output/f{f0}v{v0}-f{f1}v{v1}-w{valid_weight_threshold}-d{dist_thresh}-re{rgb_error_thresh}-pc{pc:.0f}.png'
    Image.fromarray(stack_img01).save(o01)
    o10 = f'debug_output/f{f1}v{v1}-f{f0}v{v0}-w{valid_weight_threshold}-d{dist_thresh}-re{rgb_error_thresh}-pc{pc:.0f}.png'
    Image.fromarray(stack_img10).save(o10)
    print(o01)
    print(o10)




warp(653,18,6,18)
warp(653,18,289,6)
'''
for view in [0,6,12,18]:
    warp(1,view,547,view)

warp(218, 0, 389, 12)
warp(218, 6, 389, 18)
warp(218, 12, 389, 0)
warp(218, 18, 389, 6)

warp(281, 0, 344, 18)
warp(281, 6, 344, 0)
warp(281, 12, 344, 6)
warp(281, 18, 344, 12)

warp(202, 0, 481, 6)
warp(202, 6, 481, 12)
warp(202, 12, 481, 18)
warp(202, 18, 481, 0)
'''