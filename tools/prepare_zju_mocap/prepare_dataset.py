import os
import sys

from shutil import copyfile

import pickle
import yaml
import numpy as np
from tqdm import tqdm

from pathlib import Path
sys.path.append(str(Path(os.getcwd()).resolve().parents[1]))

from third_parties.smpl.smpl_numpy import SMPL
from core.utils.file_util import split_path
from core.utils.image_util import load_image, save_image, to_3ch_image
from collections import OrderedDict

from absl import app
from absl import flags
FLAGS = flags.FLAGS

flags.DEFINE_string('cfg',
                    '387.yaml',
                    'the path of config file')

MODEL_DIR = '../../third_parties/smpl/models'


def parse_config():
    config = None
    with open(FLAGS.cfg, 'r') as file:
        config = yaml.full_load(file)

    return config


def prepare_dir(output_path, name):
    out_dir = os.path.join(output_path, name)
    os.makedirs(out_dir, exist_ok=True)

    return out_dir


def get_mask(subject_dir, img_name):
    msk_path = os.path.join(subject_dir, 'mask',
                            img_name)[:-4] + '.png'
    msk = np.array(load_image(msk_path))[:, :, 0]
    msk = (msk != 0).astype(np.uint8) # binarize 1 or 0

    msk_path = os.path.join(subject_dir, 'mask_cihp',
                            img_name)[:-4] + '.png'
    msk_cihp = np.array(load_image(msk_path))[:, :, 0]
    msk_cihp = (msk_cihp != 0).astype(np.uint8)

    msk = (msk | msk_cihp).astype(np.uint8) #union of two masks
    msk[msk == 1] = 255

    return msk


def main(argv):
    del argv  # Unused.

    cfg = parse_config()
    subject = cfg['dataset']['subject']
    sex = cfg['dataset']['sex']
    max_frames = cfg['max_frames']

    dataset_dir = cfg['dataset']['zju_mocap_path']
    subject_dir = os.path.join(dataset_dir, f"CoreView_{subject}")
    smpl_params_dir = os.path.join(subject_dir, "new_params")

    multi_select_view = cfg['training_view']
    if type(multi_select_view)==int:
        multi_select_view = [multi_select_view]
    elif type(multi_select_view)==str and multi_select_view=='all':
        multi_select_view = list(range(0,23))
    multi_select_view = sorted(multi_select_view)

    if 'train_split_file' in cfg:
        with open(cfg['train_split_file'], mode="r") as fp:
            frame_list = np.loadtxt(fp, dtype=int).tolist()
    else:
        frame_list = list(range(max_frames))
    
    if cfg.get('skip',-1)>0:
        frame_list = frame_list[::cfg['skip']]

    anno_path = os.path.join('data/zju/CoreView_387/annots.npy')
    annots = np.load(anno_path, allow_pickle=True).item()

    #load cameras
    cams = annots['cams']
    Ks, Ds, Es = {}, {}, {}
    for i, select_view in enumerate(multi_select_view):
        cam_Ks = np.array(cams['K'])[select_view].astype('float32')
        cam_Rs = np.array(cams['R'])[select_view].astype('float32')
        cam_Ts = np.array(cams['T'])[select_view].astype('float32') / 1000.
        cam_Ds = np.array(cams['D'])[select_view].astype('float32')

        K = cam_Ks     #(3, 3)
        D = cam_Ds[:, 0]
        E = np.eye(4)  #(4, 4)
        cam_T = cam_Ts[:3, 0]
        E[:3, :3] = cam_Rs
        E[:3, 3]= cam_T

        Ks[select_view], Ds[select_view], Es[select_view] = K, D, E

    
    # load image paths
    img_paths = []
    for frame_id in frame_list:
        imgs_this_frame = annots['ims'][frame_id]['ims']      
        img_paths.extend([imgs_this_frame[v] for v in multi_select_view])
    # import ipdb; ipdb.set_trace()
    img_paths = np.array(img_paths) #n_frame*n_view


    output_path = os.path.join(cfg['output']['dir'], 
                               subject if 'name' not in cfg['output'].keys() else cfg['output']['name'])
    os.makedirs(output_path, exist_ok=True)
    out_img_dir  = prepare_dir(output_path, 'images')
    out_mask_dir = prepare_dir(output_path, 'masks')

    # copy config file
    copyfile(FLAGS.cfg, os.path.join(output_path, 'config.yaml'))

    smpl_model = SMPL(sex=sex, model_dir=MODEL_DIR)

    cameras = OrderedDict()
    mesh_infos = OrderedDict()
    all_betas = []
    for idx, ipath in enumerate(tqdm(img_paths)):
        frame_id, select_view = idx//len(multi_select_view), idx%len(multi_select_view)
        frame_id, select_view = frame_list[frame_id], multi_select_view[select_view] #!!
        if cfg["v2"]:
            out_name = ipath
        else:
            if len(multi_select_view)==1:
                out_name = 'frame_{:06d}'.format(frame_id)
            else:
                out_name = 'frame_{:06d}_view_{:02d}'.format(frame_id, select_view)

        img_path = os.path.join(subject_dir, ipath)
    
        # load image
        # img = np.array(load_image(img_path))

        if subject in ['313', '315']:
            _, image_basename, _ = split_path(img_path)
            start = image_basename.find(')_')
            smpl_idx = int(image_basename[start+2: start+6])
        else:
            smpl_idx = frame_id #!!!

        # load smpl parameters
        smpl_params = np.load(
            os.path.join(smpl_params_dir, f"{smpl_idx}.npy"),
            allow_pickle=True).item()

        betas = smpl_params['shapes'][0] #(10,)
        poses = smpl_params['poses'][0]  #(72,)
        Rh = smpl_params['Rh'][0]  #(3,)
        Th = smpl_params['Th'][0]  #(3,)

        all_betas.append(betas)

        # write camera info
        cameras[out_name] = {
                'intrinsics': Ks[select_view],
                'extrinsics': Es[select_view],
                'distortions': Ds[select_view]
        }

        # write mesh info
        _, tpose_joints = smpl_model(np.zeros_like(poses), betas)
        _, joints = smpl_model(poses, betas)
        mesh_infos[out_name] = {
            'Rh': Rh,
            'Th': Th,
            'poses': poses,
            'joints': joints, 
            'tpose_joints': tpose_joints
        }

        # load and write mask
        mask = get_mask(subject_dir, ipath)
        if cfg["v2"]:
            pass
        else:
            save_image(to_3ch_image(mask), 
                    os.path.join(out_mask_dir, out_name+'.png'))

        if cfg["v2"]:
            pass
        else:
            # write image
            out_image_path = os.path.join(out_img_dir, '{}.png'.format(out_name))
            save_image(img, out_image_path)

    # write camera infos
    with open(os.path.join(output_path, 'cameras.pkl'), 'wb') as f:   
        pickle.dump(cameras, f)
        
    # write mesh infos
    with open(os.path.join(output_path, 'mesh_infos.pkl'), 'wb') as f:   
        pickle.dump(mesh_infos, f)

    # write canonical joints
    avg_betas = np.mean(np.stack(all_betas, axis=0), axis=0)
    smpl_model = SMPL(sex, model_dir=MODEL_DIR)
    _, template_joints = smpl_model(np.zeros(72), avg_betas)
    with open(os.path.join(output_path, 'canonical_joints.pkl'), 'wb') as f:   
        pickle.dump(
            {
                'joints': template_joints,
            }, f)

if __name__ == '__main__':
    app.run(main)
