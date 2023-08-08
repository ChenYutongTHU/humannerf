import os

import torch
import numpy as np
from tqdm import tqdm

from core.data import create_dataloader
from core.nets import create_network
from core.utils.train_util import cpu_data_to_gpu
from core.utils.image_util import ImageWriter, to_8b_image, to_8b3ch_image
from core.utils.metrics_util import MetricsWriter
from configs import cfg, args
from collections import defaultdict
EXCLUDE_KEYS_TO_GPU = ['frame_name',
                       'img_width', 'img_height', 'ray_mask']


def load_network():
    model = create_network()
    ckpt_path = os.path.join(cfg.logdir, f'{cfg.load_net}.tar')
    ckpt = torch.load(ckpt_path, map_location='cuda:0')
    model.load_state_dict(ckpt['network'], strict=False)
    print('load network from ', ckpt_path)
    return model.cuda().deploy_mlps_to_secondary_gpus()


def unpack_alpha_map(alpha_vals, ray_mask, width, height):
    alpha_map = np.zeros((height * width), dtype='float32')
    alpha_map[ray_mask] = alpha_vals
    return alpha_map.reshape((height, width))


def unpack_to_image(width, height, ray_mask, bgcolor,
                    rgb, alpha, truth=None):
    
    rgb_image = np.full((height * width, 3), bgcolor, dtype='float32')
    truth_image = np.full((height * width, 3), bgcolor, dtype='float32')

    rgb_image[ray_mask] = rgb
    rgb_image = to_8b_image(rgb_image.reshape((height, width, 3)))

    if truth is not None:
        truth_image[ray_mask] = truth
        truth_image = to_8b_image(truth_image.reshape((height, width, 3)))

    alpha_map = unpack_alpha_map(alpha, ray_mask, width, height)
    alpha_image  = to_8b3ch_image(alpha_map)

    return rgb_image, alpha_image, truth_image


def _freeview(
        data_type='freeview',
        folder_name=None, render_folder_name='freeview', **kwargs):
    cfg.perturb = 0.

    model = load_network()
    test_loader = create_dataloader(data_type, **kwargs)
    writer = ImageWriter(
        output_dir=os.path.join(cfg.logdir, cfg.load_net+cfg.eval_output_tag),
        exp_name=render_folder_name)
    # metrics_writer = MetricsWriter(
    #     output_dir=os.path.join(cfg.logdir, cfg.load_net+cfg.eval_output_tag), 
    #     exp_name=render_folder_name,
    #     dataset=cfg[render_folder_name].dataset)

    model.eval()
    step = 0
    for batch in enumerate(test_loader):
        step += 1
        for k, v in batch.items():
            batch[k] = v[0]

        data = cpu_data_to_gpu(
                    batch,
                    exclude_keys=EXCLUDE_KEYS_TO_GPU)

        with torch.no_grad():
            net_output = model(**data, 
                               iter_val=cfg.eval_iter)
        
        multi_outputs = net_output['multi_outputs']
        width = batch['img_width']
        height = batch['img_height']
        ray_mask = batch['ray_mask']

        if multi_outputs==True:
            rgbs = net_output['rgb']
            alphas = net_output['alpha']
            img_names = [f'{step:06d}_head{i}' for i,_ in enumerate(rgbs)]             
        else:
            rgbs = [net_output['rgb']]
            alphas = [net_output['alpha']]
            img_names = [None]            
        for rgb, alpha, img_name in zip(rgbs, alphas, img_names):
            target_rgbs = batch.get('target_rgbs', None)
            raw_rgbs = batch.get('raw_rgbs', None)
            rgb_img, alpha_img, _ = unpack_to_image(
                width, height, ray_mask, np.array(cfg.bgcolor) / 255.,
                rgb.data.cpu().numpy(),
                alpha.data.cpu().numpy())
            imgs = [rgb_img]
            if cfg.show_truth and target_rgbs is not None:
                raw_rgbs = to_8b_image(raw_rgbs.numpy())
                imgs.append(raw_rgbs)
            if cfg.show_alpha:
                imgs.append(alpha_img)
            img_out = np.concatenate(imgs, axis=1)
            writer.append(img_out, img_name=img_name)
        #metrics_writer.append(name=img_name, pred=rgb_img, target=raw_rgbs)

    writer.finalize()


def run_freeview():
    _freeview(
        data_type='freeview',
        render_folder_name=f"freeview_{cfg.freeview.frame_idx}" \
            if not cfg.render_folder_name else cfg.render_folder_name)


def run_tpose():
    cfg.ignore_non_rigid_motions = True
    _freeview(
        data_type='tpose',
        folder_name='tpose' \
            if not cfg.render_folder_name else cfg.render_folder_name)


def run_novelpose():
    cfg.show_truth = True
    _freeview(
        data_type=f'novelpose',
        pose_id=args.pose_id,
        folder_name=f'novelpose/{args.pose_id}' \
            if not cfg.render_folder_name else cfg.render_folder_name)   

def run_novelview():
    cfg.show_truth = True
    run_movement(render_folder_name='novelview')

def run_novelview_all():
    cfg.show_truth = True
    run_movement(render_folder_name='novelview_all')

def run_novelpose_eval():
    cfg.show_truth = True
    run_movement(render_folder_name='novelpose_eval')

def run_movement(render_folder_name='movement'):
    cfg.perturb = 0.
    cfg.show_truth = True
    model = load_network()
    test_loader = create_dataloader(render_folder_name)
    writer = ImageWriter(
        output_dir=os.path.join(cfg.logdir, cfg.load_net+cfg.eval_output_tag),
        exp_name=render_folder_name)
    metrics_writer = MetricsWriter(
        output_dir=os.path.join(cfg.logdir, cfg.load_net+cfg.eval_output_tag), 
        exp_name=render_folder_name,
        dataset=cfg[render_folder_name].dataset)

    model.eval()
    for idx, batch in enumerate(tqdm(test_loader)):
        for k, v in batch.items():
            batch[k] = v[0]

        data = cpu_data_to_gpu(
                    batch,
                    exclude_keys=EXCLUDE_KEYS_TO_GPU + ['target_rgbs'])

        with torch.no_grad():
            net_output = model(**data, iter_val=cfg.eval_iter)

        rgb = net_output['rgb']
        alpha = net_output['alpha']

        width = batch['img_width']
        height = batch['img_height']
        ray_mask = batch['ray_mask']

        rgb_img, alpha_img, truth_img = \
            unpack_to_image(
                width, height, ray_mask, np.array(cfg.bgcolor)/255.,
                rgb.data.cpu().numpy(),
                alpha.data.cpu().numpy(),
                batch['target_rgbs'])

        imgs = [rgb_img]
        if cfg.show_truth:
            imgs.append(truth_img)
        if cfg.show_alpha:
            imgs.append(alpha_img)


        metrics_writer.append(name=batch['frame_name'], pred=rgb_img, target=truth_img, mask=None)
        img_out = np.concatenate(imgs, axis=1)
        writer.append(img_out, img_name=batch['frame_name'].replace('/','-'))


    writer.finalize()
    metrics_writer.finalize()
    
        
if __name__ == '__main__':
    globals()[f'run_{args.type}']()
