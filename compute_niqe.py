from basicsr.metrics.niqe import calculate_niqe
import os, cv2
from tqdm import tqdm
from collections import defaultdict
import numpy as np

name2head = defaultdict(list)
for split in ['movement', 'novelview_all']:
    # for image_dir in ['experiments/human_nerf/zju_mocap/p387/tava/387_l1.0m0.2_4view_last2/latest',
    #     ]+[f'experiments/human_nerf/zju_mocap/p387/tava/387_4view_mh-view/latest_h{i}' for i in range(4)]:
    #for image_dir in [f'experiments/human_nerf/zju_mocap/p387/tava/387_l1.0m0.2_1view_camera{i}/latest' for i in [0,6,12,18]]:
    for image_dir in ['experiments/human_nerf/zju_mocap/p387/tava/387_l1.0m0.2_4view_last2_mse/latest']+ \
        [f'experiments/human_nerf/zju_mocap/p387/tava/387_4view_mh-view_mse/latest_h{i}' for i in range(4)]:
        score_perimg_f = open(os.path.join(image_dir, f'{split}-niqe.perimg.txt'),'a')
        score_ave_f = open(os.path.join(image_dir, f'{split}-niqe.average.txt'),'a')
        scores = []
        for filename in tqdm(sorted(os.listdir(os.path.join(image_dir, split)))):
            name = filename.split('.')[0]
            if split == 'novelview_all':
                #exclude train view
                camera = int(name.split('-')[0].split('B')[1])-1
                if camera in [0,6,12,18]:
                    continue
            img = cv2.imread(os.path.join(image_dir, split, filename))
            niqe = calculate_niqe(img, crop_border=0)
            score_perimg_f.writelines(f'{name}: niqe-{niqe:.4f}\n')
            scores.append(niqe)

            if 'mh-view' in image_dir or 'camera' in image_dir:
                name2head[name].append(niqe)
            #print(f'{name}: niqe-{niqe:.4f}\n')
        score_perimg_f.close()
        ave = np.mean(scores)
        score_ave_f.writelines(f'{ave:.4f}\n')
        score_ave_f.close()
        print(image_dir, split, ave, len(scores))

    name2best = {}
    for name, vlist in name2head.items():
        name2best[name] = min(vlist)
    ave_best = np.mean([b for b in name2best.values()])
    print(f'multi-head split:{split} argbest:{ave_best:.4f}')
