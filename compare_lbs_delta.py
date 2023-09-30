from collections import defaultdict
import os, cv2, numpy as np
from collections import OrderedDict
import pickle
from tqdm import tqdm

def map_offset_to_color(offset, min=-0.02, max=0.02): #
    offset = np.clip((offset-min)/(max-min),0,1)
    return offset

name2filepath=OrderedDict({
    'only_lbs':'experiments/human_nerf/zju_mocap/p387/tava/387_l1.0m0.2_4view_last2/latest_only_lbs/movement',
    'full':'experiments/human_nerf/zju_mocap/p387/tava/387_l1.0m0.2_4view_last2/latest/movement'
})
output_dir = 'experiments/human_nerf/zju_mocap/p387/tava/387_l1.0m0.2_4view_last2/cmp_lbs_delta/movement'
filename = 'experiments/human_nerf/zju_mocap/p387/tava/387_l1.0m0.2_4view_last2/latest_/movement_3d/{}-rays.pkl'

os.makedirs(output_dir, exist_ok=True)
metric='psnr'
name2scores = defaultdict(list)
name2imgs = defaultdict(lambda:[None,None,None, None])
for i,(name, filepath) in enumerate(name2filepath.items()):
    lines = open(filepath+'-metrics.perimg.txt').readlines()
    for l in lines:
        l = l.strip()
        try:
            frame_name, scores = l.split(':')
            for s in scores.split(' ')[1:]:
                m, v = s.split('-')
                v = float(v)
                if m==metric:
                    name2scores[frame_name].append(v)
        except:
            continue
        img = cv2.imread(os.path.join(filepath, frame_name+'.png'))
        name2imgs[frame_name][i] = img[:,:512]
        name2imgs[frame_name][-1] = img[:,512:]

        if name=='full':
            res = pickle.load(open(filename.format(frame_name),'rb'))
            offsets = np.sum(res['offset_on_rays']*res['weights_on_rays'][...,None],axis=1)
            img = np.zeros((512,512,3), np.float32)
            cs= map_offset_to_color(offsets)
            for pos, c, rgb in zip(res['pos_on_image'], cs, res['rgb_on_image']):
                if np.sum(rgb)>0:
                    img[pos[0],pos[1]] = c
            name2imgs[frame_name][-2] = (img*255).astype(np.uint8)
            s1, s2 = name2scores[name]
            new_img = np.concatenate(name2imgs[frame_name],axis=1)
            cv2.imwrite(os.path.join(output_dir, f'{name}_lbs-{s1:.1f}_full-{s2:.1f}.png'),new_img)


'''
for name, imgs in name2imgs.items():

    mask = ((imgs[0]>0)+(imgs[1]>0)).astype(np.bool)
    delta = np.mean(np.abs(imgs[1]-imgs[0]), axis=2,keepdims=True).astype(np.uint8) #
    delta = cv2.applyColorMap(delta, cv2.COLORMAP_JET)
    delta = delta*mask
    new_img = np.concatenate([imgs[0],imgs[1],delta,imgs[-1]],axis=1)
'''