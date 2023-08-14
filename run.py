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

def unpack_weight_map(weight_vals, ray_mask, width, height, weight_mask=None):
    weight_map = np.zeros((height * width, weight_vals.shape[-1]), dtype='float32')
    if weight_mask is not None:
        weight_vals[weight_mask==False] = 0
    weight_map[ray_mask,:] = weight_vals #(N,
    return weight_map.reshape((height, width,-1))

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
        
        multi_outputs = (type(net_output['rgb'])==list)
        width = batch['img_width']
        height = batch['img_height']
        ray_mask = batch['ray_mask']

        if multi_outputs==True:
            rgbs = net_output['rgb']
            alphas = net_output['alpha']
            depths = net_output['depth']
            img_names = [f'{step:06d}_head{i}' for i,_ in enumerate(rgbs)]             
        else:
            rgbs = [net_output['rgb']]
            alphas = [net_output['alpha']]
            depths = [net_output['depth']]
            img_names = [None]            
        for rgb, alpha, depth, img_name in zip(rgbs, alphas, depths, img_names):
            target_rgbs = batch.get('target_rgbs', None)
            raw_rgbs = batch.get('raw_rgbs', None)
            rgb_img, alpha_img, _ = unpack_to_image(
                width, height, ray_mask, np.array(cfg.bgcolor) / 255.,
                rgb.data.cpu().numpy(),)
            depth_img = unpack_alpha_map(alpha_vals=depth, ray_mask=ray_mask, width=width, height=height)
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

def run_train_render():
    run_movement(render_folder_name='train_render')

def run_movement(render_folder_name='movement'):
    cfg.perturb = 0.
    cfg.show_truth = True
    model = load_network()
    test_loader = create_dataloader(render_folder_name)
    multi_outputs = (cfg.multihead.head_num>1 and cfg.test.head_id==-1)
    if multi_outputs:
        writer, metrics_writer = [], []
        for i in range(cfg.multihead.head_num):
            writer.append(ImageWriter(
                output_dir=os.path.join(cfg.logdir, cfg.load_net+cfg.eval_output_tag+f'_h{i}'),
                exp_name=render_folder_name))        
            metrics_writer.append(MetricsWriter(
                output_dir=os.path.join(cfg.logdir, cfg.load_net+cfg.eval_output_tag+f'_h{i}'), 
                exp_name=render_folder_name,
                dataset=cfg[render_folder_name].dataset, 
                lpips_computer=metrics_writer[0].lpips_computer if i!=0 else None))       
    else:
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

        if multi_outputs==True:
            assert (type(net_output['rgb'])==list)
            rgbs = net_output['rgb']
            alphas = net_output['alpha']
            depths = net_output['depth']
            weights_on_rays, xyz_on_rays, rgb_on_rays = net_output['weights_on_rays'],net_output['xyz_on_rays'],net_output['rgb_on_rays']
            cnl_xyzs, cnl_rgbs, cnl_weights = net_output['cnl_xyz'],net_output['cnl_rgb'],net_output['cnl_weight']
            img_names = [f'{idx:06d}_head{i}' for i,_ in enumerate(rgbs)]          
        else:
            rgbs = [net_output['rgb']]
            alphas = [net_output['alpha']]
            depths = [net_output['depth']]
            weights_on_rays, xyz_on_rays, rgb_on_rays = [net_output['weights_on_rays']],[net_output['xyz_on_rays']],[net_output['rgb_on_rays']]
            cnl_xyzs, cnl_rgbs, cnl_weights = [net_output['cnl_xyz']],[net_output['cnl_rgb']], [net_output['cnl_weight']]
            img_names = [None] 

        for hid,(rgb, alpha, depth, cnl_xyz, cnl_rgb, cnl_weight, weights_on_ray, xyz_on_ray, rgb_on_ray, img_name) in \
                enumerate(zip(rgbs, alphas, depths, cnl_xyzs, cnl_rgbs, cnl_weights, weights_on_rays, xyz_on_rays, rgb_on_rays, img_names)):
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

            if multi_outputs:
                metrics_writer[hid].append(name=batch['frame_name'], pred=rgb_img, target=truth_img, mask=None)
                img_out = np.concatenate(imgs, axis=1)
                writer[hid].append(img_out, img_name=batch['frame_name'].replace('/','-'))
            else:
                metrics_writer.append(name=batch['frame_name'], pred=rgb_img, target=truth_img, mask=None)
                img_out = np.concatenate(imgs, axis=1)
                writer.append(img_out, img_name=batch['frame_name'].replace('/','-'))
            
            #reproject to 3D image
            '''
            depth_img = unpack_alpha_map(alpha_vals=depth.data.cpu().numpy(), ray_mask=ray_mask, width=width, height=height)
            weight_img = unpack_weight_map(weight_vals=weight.data.cpu().numpy(), ray_mask=ray_mask, width=width, height=height)
            point_3ds, point_3ds_mask = test_loader.dataset.project_to_world3D(
                depth_img, batch['frame_name'], height, width,
                ray_mask=ray_mask, near=batch['near'], far=batch['far'])
            writer.append_3d(point_3ds, point_3ds_mask, obj_name=batch['frame_name'].replace('/','-'), 
                    weight_img=weight_img, depth_img=depth_img)
            '''
            
            #back to 3d-canonical
            #weight.argmax() -> rgb/density/(x,y,z)
            if cfg.test.save_3d:
                pos_on_image = (ray_mask.view((height, width))).nonzero() #N_rays, 2
                rgb_on_image = batch['target_rgbs'] #N_rays, 3
                writer.save_pkl({'weights_on_rays':weights_on_ray.data.cpu().numpy(), 
                             'rgb_on_rays':rgb_on_ray.data.cpu().numpy(), 
                             'xyz_on_rays':xyz_on_ray.data.cpu().numpy(),
                             'rgb_on_image':rgb_on_image.data.cpu().numpy(),
                             'pos_on_image':pos_on_image.data.cpu().numpy()}, name=batch['frame_name'].replace('/','-')+'-rays.pkl')
                '''
                weight_mask = (cnl_weight>cfg.test.weight_threshold)
                cnl_xyz, cnl_rgb = cnl_xyz[weight_mask].data.cpu().numpy(), cnl_rgb[weight_mask].data.cpu().numpy()
                writer.append_cnl_3d(cnl_xyz, cnl_rgb, obj_name=batch['frame_name'].replace('/','-')+'-cnl')
                '''


    if multi_outputs:
        for w,m in zip(writer, metrics_writer):
            w.finalize()
            m.finalize()
    else:
        writer.finalize()
        metrics_writer.finalize()
    
        
if __name__ == '__main__':
    globals()[f'run_{args.type}']()
