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
from prepare_dataset import prepare_dir
from absl import app
from absl import flags
FLAGS = flags.FLAGS


flags.DEFINE_string('pose',
                    '313',
                    'the path of config file')

flags.DEFINE_string('shape',
                    '387',
                    'the path of config file')

MODEL_DIR = '../../third_parties/smpl/models'


def parse_config():
    with open(FLAGS.pose+'.yaml', 'r') as file:
        pose_config = yaml.full_load(file)
    with open(FLAGS.shape+'.yaml', 'r') as file:
        shape_config = yaml.full_load(file)    
    return shape_config, pose_config

def main(argv):
    del argv  # Unused.

    shape_cfg, pose_cfg = parse_config()
    shape_subject = shape_cfg['dataset']['subject']
    sex = shape_cfg['dataset']['sex']
    pose_subject = pose_cfg['dataset']['subject']
    max_frames = min(pose_cfg['max_frames'], shape_cfg['max_frames'])


    shape_dataset_dir = shape_cfg['dataset']['zju_mocap_path']
    shape_subject_dir = os.path.join(shape_dataset_dir, f"CoreView_{shape_subject}")
    shape_smpl_params_dir = os.path.join(shape_subject_dir, "new_params")

    pose_dataset_dir = pose_cfg['dataset']['zju_mocap_path']
    pose_subject_dir = os.path.join(pose_dataset_dir, f"CoreView_{pose_subject}")
    pose_smpl_params_dir = os.path.join(pose_subject_dir, "new_params")


    shape_anno_path = os.path.join(shape_subject_dir, 'annots.npy')
    shape_annots = np.load(shape_anno_path, allow_pickle=True).item()

    pose_anno_path = os.path.join(pose_subject_dir, 'annots.npy')
    pose_annots = np.load(pose_anno_path, allow_pickle=True).item()
    
    select_view = pose_cfg['training_view']
    # load cameras
    cams = pose_annots['cams']
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

    # load image paths
    shape_img_path_frames_views = shape_annots['ims']
    shape_img_paths = np.array([
        np.array(multi_view_paths['ims'])[select_view] \
            for multi_view_paths in shape_img_path_frames_views
    ])
    if max_frames > 0:
        shape_img_paths = shape_img_paths[:max_frames]

    pose_img_path_frames_views = pose_annots['ims']
    pose_img_paths = np.array([
        np.array(multi_view_paths['ims'])[select_view] \
            for multi_view_paths in pose_img_path_frames_views
    ])
    if max_frames > 0:
        pose_img_paths = pose_img_paths[:max_frames]

    output_path = os.path.join(shape_cfg['output']['dir'], 
                               subject if 'name' not in shape_cfg['output'].keys() else shape_cfg['output']['name'])
    os.makedirs(output_path, exist_ok=True)
    out_img_dir  = prepare_dir(output_path, f'images_pose{pose_subject}')

    smpl_model = SMPL(sex=sex, model_dir=MODEL_DIR)
    cameras = {}
    mesh_infos = {}
    all_betas = []

    for idx, (pose_ipath,  shape_ipath) in enumerate(tqdm(zip(pose_img_paths,shape_img_paths))):
        out_name = 'frame_{:06d}'.format(idx)
        img_path = os.path.join(pose_subject_dir, pose_ipath)
        # load image
        img = np.array(load_image(img_path))

        if pose_subject in ['313', '315']:
            smpl_idx = idx+1
            # pose_img_path = os.path.join(pose_subject_dir, pose_ipath)
            # _, image_basename, _ = split_path(pose_img_path)
            # start = image_basename.find(')_')
            # smpl_idx = int(image_basename[start+2: start+6])
        else:
            smpl_idx = idx
        # load smpl parameters
        pose_smpl_params = np.load(
            os.path.join(pose_smpl_params_dir, f"{smpl_idx}.npy"),
            allow_pickle=True).item()   

        if shape_subject in ['313', '315']:
            smpl_idx = idx+1
            # shape_img_path = os.path.join(shape_subject_dir, shape_ipath)
            # _, image_basename, _ = split_path(shape_img_path)
            # start = image_basename.find(')_')
            # smpl_idx = int(image_basename[start+2: start+6])
        else:
            smpl_idx = idx
        # load smpl parameters
        shape_smpl_params = np.load(
            os.path.join(shape_smpl_params_dir, f"{smpl_idx}.npy"),
            allow_pickle=True).item()   

        betas = shape_smpl_params['shapes'][0] #(10,)
        poses = pose_smpl_params['poses'][0]  #(72,)
        Rh = pose_smpl_params['Rh'][0]  #(3,)
        Th = pose_smpl_params['Th'][0]  #(3,)

        # write camera info
        cameras[out_name] = {
                'intrinsics': K,
                'extrinsics': E,
                'distortions': D
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

        out_image_path = os.path.join(out_img_dir, '{}.png'.format(out_name))
        save_image(img, out_image_path)
    # write mesh infos
    with open(os.path.join(output_path, f'mesh_infos_pose{pose_subject}.pkl'), 'wb') as f:   
        pickle.dump(mesh_infos, f)

    with open(os.path.join(output_path, f'cameras_pose{pose_subject}.pkl'), 'wb') as f:   
        pickle.dump(cameras, f)

if __name__ == '__main__':
    app.run(main)