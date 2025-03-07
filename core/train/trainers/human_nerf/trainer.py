import os, wandb


import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from tqdm import tqdm

# import wandb
from third_parties.lpips import LPIPS

from core.train import create_lr_updater
from core.data import create_dataloader
from core.utils.network_util import set_requires_grad
from core.utils.metrics_util import compute_ssim
from core.utils.train_util import cpu_data_to_gpu, Timer
from core.utils.image_util import tile_images, to_8b_image

from configs import cfg
img2mse = lambda x, y : torch.mean((x - y) ** 2)
img2l1 = lambda x, y : torch.mean(torch.abs(x-y))
to8b = lambda x : (255.*np.clip(x,0.,1.)).astype(np.uint8)

EXCLUDE_KEYS_TO_GPU = ['frame_name', 'img_width', 'img_height']


def _unpack_imgs(rgbs, patch_masks, bgcolor, targets, div_indices):
    N_patch = len(div_indices) - 1
    assert patch_masks.shape[0] == N_patch
    assert targets.shape[0] == N_patch

    patch_imgs = bgcolor.expand(targets.shape).clone() # (N_patch, H, W, 3)
    for i in range(N_patch):
        patch_imgs[i, patch_masks[i]] = rgbs[div_indices[i]:div_indices[i+1]]

    return patch_imgs


def scale_for_lpips(image_tensor):
    return image_tensor * 2. - 1.


class Trainer(object):
    def __init__(self, network, optimizer, wandb_run=None):
        print('\n********** Init Trainer ***********')

        network = network.cuda().deploy_mlps_to_secondary_gpus()

        self.network = network

        self.wandb_run = wandb_run
            

        self.optimizer = optimizer
        self.update_lr = create_lr_updater()
        self.use_amp = cfg.use_amp
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        if cfg.resume and Trainer.ckpt_exists(cfg.load_net):
            self.load_ckpt(f'{cfg.load_net}')
        else:
            self.iter = 0
            self.save_ckpt('init')
            self.iter = 1
        self.start_iter = self.iter

        self.timer = Timer()

        if "lpips" in cfg.train.lossweights.keys():
            self.lpips = LPIPS(net='vgg', lpips=cfg.lpips.lpips, layers=cfg.lpips.layers)
            set_requires_grad(self.lpips, requires_grad=False)
            self.lpips = nn.DataParallel(self.lpips).cuda()

        print("Load Progress Dataset ...")
        self.prog_dataloader = create_dataloader(data_type='progress')

        print('************************************')

    def freeze_params(freeze_name):
        print('********** Frozen parameters **********')
        for name, param in self.network.named_parameters():
            if freeze_name in name: 
                param.requires_grad = False
                

    @staticmethod
    def get_ckpt_path(name):
        return os.path.join(cfg.logdir, f'{name}.tar')

    @staticmethod
    def ckpt_exists(name):
        return os.path.exists(Trainer.get_ckpt_path(name))

    ######################################################3
    ## Training 

    def get_img_rebuild_loss(self, loss_names, rgb, target):
        losses = {}

        if "mse" in loss_names:
            losses["mse"] = img2mse(rgb, target)

        if "l1" in loss_names:
            losses["l1"] = img2l1(rgb, target)

        if "lpips" in loss_names and cfg.train.lossweights.lpips>0:
            lpips_loss = self.lpips(scale_for_lpips(rgb.permute(0, 3, 1, 2)), #B,H,W,C 
                                    scale_for_lpips(target.permute(0, 3, 1, 2)))
            losses["lpips"] = torch.mean(lpips_loss)
        else:
            losses["lpips"] = 0 # accelerate computation

        return losses

    def get_loss(self, net_output, 
                 patch_masks, bgcolor, targets, div_indices):

        lossweights = {k:v for k,v in cfg.train.lossweights.items() if v>0 }
        loss_names = list(lossweights.keys())

        rgb = net_output['rgb'] # (,3)
        return_losses = {}
        multi_outputs = (type(rgb)==list)
        if multi_outputs and cfg.multihead.split=='argmin':
            assert type(rgb)==list and len(rgb) == cfg.multihead.head_num, len(rgb)
            criteria_head, selected_losses, unselected_losses = [], [], []
            for hid, rgb_head in enumerate(rgb):
                patch_img = _unpack_imgs(rgb_head, patch_masks, bgcolor, targets, div_indices)
                losses = self.get_img_rebuild_loss(loss_names, 
                        patch_img, 
                        targets)
                selected_losses.append([
                    weight * losses[k] for k, weight in lossweights.items()
                ])
                unselected_losses.append([
                    weight * losses[k] for k, weight in cfg.multihead.argmin_cfg.unselected_lossweights.items()
                ])
                criteria = []
                for k, weight in cfg.multihead.argmin_cfg.selector_criteria.items():
                    if weight>0:
                        if k=='ssim': 
                            patch_img_view = patch_img.permute(1,2,3,0).reshape(patch_img.shape[1],patch_img.shape[2],-1) #H,W,C,B
                            targets_view = targets.permute(1,2,3,0).reshape(targets.shape[1],targets.shape[2],-1) #H,W,C,B
                            ssim = compute_ssim(pred=patch_img_view.detach(), target=targets_view.detach())
                            criteria.append(-1*weight*ssim)
                        else:
                            criteria.append(weight*losses[k])

                criteria_head.append(sum(criteria).item())
                return_losses = {**return_losses,**{loss_names[i]+f'_head{hid}': selected_losses[-1][i] for i in range(len(loss_names))}}
            
            best_head = np.argmin(criteria_head)
            rgb = rgb[best_head]
            train_losses = selected_losses[best_head] # a list
            return_losses = {**return_losses, **{loss_names[i]: train_losses[i] for i in range(len(loss_names))}}
            return_losses['best_head'] = best_head
            for head_id in range(cfg.multihead.head_num):
                if head_id==best_head:
                    continue
                train_losses += unselected_losses[head_id] #already a list

        else:
            losses = self.get_img_rebuild_loss(
                            loss_names, 
                            _unpack_imgs(rgb, patch_masks, bgcolor,
                                        targets, div_indices), 
                            targets)
            train_losses = [
                weight * losses[k] for k, weight in lossweights.items()
            ]
            return_losses = {**return_losses, **{loss_names[i]: train_losses[i] for i in range(len(loss_names))}}


        return sum(train_losses), \
                return_losses

    def train_begin(self, train_dataloader):
        assert train_dataloader.batch_size == 1

        self.network.train()
        cfg.perturb = cfg.train.perturb

    def train_end(self):
        pass

    def train(self, epoch, train_dataloader):
        self.train_begin(train_dataloader=train_dataloader)

        self.timer.begin()
        for batch_idx, batch in enumerate(train_dataloader):
            if self.iter > cfg.train.maxiter:
                break

            self.optimizer.zero_grad()

            # only access the first batch as we process one image one time
            for k, v in batch.items():
                if k=='frame_name_history':
                    batch[k] = [[v2[0] for v2 in v1] for v1 in v]
                else:
                    batch[k] = v[0]

            batch['iter_val'] = torch.full((1,), self.iter)
            data = cpu_data_to_gpu(
                batch, exclude_keys=EXCLUDE_KEYS_TO_GPU+['frame_name_history'])
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.use_amp):
                net_output = self.network(**data)
                train_loss, loss_dict = self.get_loss(
                    net_output=net_output,
                    patch_masks=data['patch_masks'],
                    bgcolor=data['bgcolor'] / 255.,
                    targets=data['target_patches'],
                    div_indices=data['patch_div_indices'])

            # train_loss.backward()
            # self.optimizer.step()

            self.scaler.scale(train_loss).backward() 
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.iter % cfg.train.log_interval == 0:
                loss_str = f"Loss: {train_loss.item():.4f} ["
                for k, v in loss_dict.items():
                    loss_str += f"{k}: {v.item():.4f} "
                loss_str += "]"

                log_str = 'Epoch: {} [Iter {}, {}/{} ({:.0f}%), {}] {}'
                log_str = log_str.format(
                    epoch, self.iter,
                    batch_idx * cfg.train.batch_size, len(train_dataloader.dataset),
                    100. * batch_idx / len(train_dataloader), 
                    self.timer.log(),
                    loss_str)
                print(log_str)
                if self.wandb_run is not None:
                    wandb.log(loss_dict)

            is_reload_model = False
            if ((self.iter in [self.start_iter, 100, 300, 1000, 2500]) or \
                self.iter % cfg.progress.dump_interval == 0):
                is_reload_model = self.progress() #if is_empty_Image keep_iter=1
                is_reload_model = False #do not reproduce image

            if not is_reload_model:
                if self.iter % cfg.train.save_checkpt_interval == 0 or self.iter==self.start_iter:
                    self.save_ckpt('latest')

                if cfg.save_all:
                    if self.iter % cfg.train.save_model_interval == 0:
                        self.save_ckpt(f'iter_{self.iter}')

                self.update_lr(self.optimizer, self.iter)

                self.iter += 1
    
    def finalize(self):
        self.save_ckpt('latest')

    ######################################################3
    ## Progress

    def progress_begin(self):
        self.network.eval()
        cfg.perturb = 0.

    def progress_end(self):
        self.network.train()
        cfg.perturb = cfg.train.perturb

    def progress(self):
        self.progress_begin()

        print('Evaluate Progress Images ...') #TODO quantitative evaluation

        images, images_list = [], [[] for _ in range(cfg.multihead.head_num)] #for multi_head
        is_empty_img = False
        for _, batch in enumerate(tqdm(self.prog_dataloader)):

            # only access the first batch as we process one image one time
            for k, v in batch.items():
                if k=='frame_name_history':
                    batch[k] = [[v2[0] for v2 in v1] for v1 in v]
                else:
                    batch[k] = v[0]

            width = batch['img_width']
            height = batch['img_height']
            ray_mask = batch['ray_mask']


            batch['iter_val'] = torch.full((1,), self.iter)
            data = cpu_data_to_gpu(
                    batch, exclude_keys=EXCLUDE_KEYS_TO_GPU + ['target_rgbs','frame_name_history'])
            with torch.no_grad():
                net_output = self.network(**data)

            multi_outputs = (type(net_output['rgb'])==list)
            if multi_outputs:
                rgbs = [rgb.data.to("cpu").numpy() for rgb in net_output['rgb']]
            else:
                rgbs = [net_output['rgb'].data.to("cpu").numpy()]

            for head_id, rgb in enumerate(rgbs):
                target_rgbs = batch['target_rgbs']
                rendered = np.full(
                            (height * width, 3), np.array(cfg.bgcolor)/255., 
                            dtype='float32')
                truth = np.full(
                            (height * width, 3), np.array(cfg.bgcolor)/255., 
                            dtype='float32')
                            
                rendered[ray_mask] = rgb
                truth[ray_mask] = target_rgbs

                truth = to_8b_image(truth.reshape((height, width, -1)))
                rendered = to_8b_image(rendered.reshape((height, width, -1)))
                if multi_outputs:
                    images_list[head_id].append(np.concatenate([rendered, truth], axis=1))
                else:
                    images.append(np.concatenate([rendered, truth], axis=1))

            # check if we create empty images (only at the begining of training)
            
            if self.iter <= 5000 and \
                np.allclose(rendered, np.array(cfg.bgcolor), atol=5.):
                is_empty_img = True
                #break
            

        if multi_outputs:
            for head_id, images in enumerate(images_list):
                tiled_image = tile_images(images)
                Image.fromarray(tiled_image).save(
                    os.path.join(cfg.logdir, "prog_{:06}_head{:01}.jpg".format(self.iter, head_id)))                
        else: 
            tiled_image = tile_images(images)
            Image.fromarray(tiled_image).save(
                os.path.join(cfg.logdir, "prog_{:06}.jpg".format(self.iter)))

        
        if is_empty_img:
            #print("Produce empty images; reload the init model.")
            print("Produce empty images; ")
            #self.load_ckpt('init')

            
        self.progress_end()

        return is_empty_img


    ######################################################3
    ## Utils

    def save_ckpt(self, name):
        path = Trainer.get_ckpt_path(name)
        print(f"Save checkpoint to {path} ...")

        torch.save({
            'iter': self.iter,
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)

    def load_ckpt(self, name):
        if os.path.isfile(name):
            path = name
        else:
            path = Trainer.get_ckpt_path(name)
        print(f"Load checkpoint from {path} ...")
        
        ckpt = torch.load(path, map_location='cuda:0')
        self.iter = ckpt['iter'] + 1

        self.network.load_state_dict(ckpt['network'], strict=False)
        self.optimizer.load_state_dict(ckpt['optimizer'])
