import os
import pickle

import numpy as np
import cv2
import torch, json, torchvision
import torch.utils.data
from PIL import Image

from core.utils.image_util import load_image, to_3ch_image
from core.utils.body_util import \
    body_pose_to_body_RTs, \
    get_canonical_global_tfms, \
    approx_gaussian_bone_volumes
from core.utils.file_util import list_files, split_path
from core.utils.camera_util import \
    apply_global_tfm_to_camera, \
    get_rays_from_KRT, \
    rays_intersect_3d_bbox
from tools.prepare_zju_mocap.prepare_dataset import get_mask

from configs import cfg


class Dataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            dataset_path,
            source_path=None, 
            keyfilter=None, 
            maxframes=-1,
            bgcolor=None,
            ray_shoot_mode='image',
            skip=1,
            pose_condition_file=None,pose_condition_file_cmlp=None,
            select_views='all',
            **_):

        print('[Dataset Path] ', dataset_path, '[Source Path] ', source_path) 
        self.dataset_path, self.source_path = dataset_path, source_path
        
        if self.source_path is None:
            self.image_dir = os.path.join(dataset_path, 'images') 
        else:
            self.image_dir = source_path

        self.canonical_joints, self.canonical_bbox = \
            self.load_canonical_joints()

        if 'motion_weights_priors' in keyfilter:
            self.motion_weights_priors = \
                approx_gaussian_bone_volumes(
                    self.canonical_joints,   
                    self.canonical_bbox['min_xyz'],
                    self.canonical_bbox['max_xyz'],
                    grid_size=cfg.mweight_volume.volume_size).astype('float32')


        self.cameras = self.load_train_cameras()
        self.mesh_infos = self.load_train_mesh_infos()

        framelist = self.load_train_frames()
        self.framelist_all = framelist
        self.views_all = set([self.get_frame_camera(f)[1] for f in framelist])
        if select_views != 'all':
            framelist = [f for f in framelist if self.get_frame_camera(f)[1] in select_views]
        self.skip = skip
        self.framelist = framelist[::skip]

        if maxframes > 0:
            self.framelist = self.framelist[:maxframes]
        print(f' -- Total Frames: {self.get_total_frames()}')

        
        self.views = self.get_total_views()
        print(f'Views:{self.views}')

        self.keyfilter = keyfilter
        self.bgcolor = bgcolor

        self.ray_shoot_mode = ray_shoot_mode

        if os.path.isfile(cfg.multihead.split):
            self.sample2head_id = json.load(open(cfg.mƒultihead.split,'r'))
        else:
            self.sample2head_id = None

        if os.environ.get('TEST_DIR', '') != '':
            print('Set test_dir to '+os.environ.get('TEST_DIR', '')) 
            self.test_dir = int(os.environ.get('TEST_DIR', ''))
            self.test_dir_camera = self.cameras[[name for name in self.framelist \
                    if self.get_frame_camera(name)[1]==self.test_dir][0]]
            print('Test direction uses camera:', self.test_dir_camera)
        
        if pose_condition_file is not None:
            print(f'Load pose condition from {pose_condition_file}')
            frameid2pose_condition = np.load(pose_condition_file)
            self.pose_condition_list = [frameid2pose_condition[self.get_frame_camera(f)[0]] for f in self.framelist]
        else:
            self.pose_condition_list = None

        if pose_condition_file_cmlp is not None:
            print(f'Load pose condition from {pose_condition_file_cmlp}')
            frameid2pose_condition = np.load(pose_condition_file_cmlp)
            self.pose_condition_list_cmlp = [frameid2pose_condition[self.get_frame_camera(f)[0]] for f in self.framelist]
        else:
            self.pose_condition_list_cmlp = None

    def load_canonical_joints(self):
        cl_joint_path = os.path.join(self.dataset_path, 'canonical_joints.pkl')
        with open(cl_joint_path, 'rb') as f:
            cl_joint_data = pickle.load(f)
        canonical_joints = cl_joint_data['joints'].astype('float32')
        canonical_bbox = self.skeleton_to_bbox(canonical_joints)

        return canonical_joints, canonical_bbox

    def load_train_cameras(self):
        cameras = None
        with open(os.path.join(self.dataset_path, 'cameras.pkl'), 'rb') as f: 
            cameras = pickle.load(f)
        return cameras

    @staticmethod
    def skeleton_to_bbox(skeleton):
        min_xyz = np.min(skeleton, axis=0) - cfg.bbox_offset
        max_xyz = np.max(skeleton, axis=0) + cfg.bbox_offset

        return {
            'min_xyz': min_xyz,
            'max_xyz': max_xyz
        }

    def load_train_mesh_infos(self):
        mesh_infos = None
        with open(os.path.join(self.dataset_path, 'mesh_infos.pkl'), 'rb') as f:   
            mesh_infos = pickle.load(f)

        for frame_name in mesh_infos.keys():
            bbox = self.skeleton_to_bbox(mesh_infos[frame_name]['joints'])
            mesh_infos[frame_name]['bbox'] = bbox

        return mesh_infos

    def get_frame_camera(self, name):
        name = name.split('.')[0] #to remove ext
        if 'frame' in name:
            if '_view_' in name:
                frame, camera = name.split('_view_') #frame_000563, 12
            else:
                frame = name
                camera = 0
            frame_int = int(frame.split('frame_')[1])
            camera_int = int(camera)
        elif 'Camera' in name:
            camera, frame = name.split('/')
            camera_int = int(camera.split('Camera_B')[1])
            frame_int = int(frame)
        else:
            frame_int = int(name)
            camera_int  =0
        return frame_int, camera_int
    
    def get_framename(self, frame_int, camera_int):
        #'Camera_B13/000299.jpg'
        name1 = f'Camera_B{camera_int}/{frame_int:06d}.jpg'
        if name1 not in self.framelist_all:
            raise ValueError
        return name1
        #name2 = 

    def load_train_frames(self):
        if self.source_path is None:
            img_paths = list_files(os.path.join(self.dataset_path, 'images'),
                                exts=['.png'])
            frames = [split_path(ipath)[1] for ipath in img_paths]
        else:
            frames = list(self.mesh_infos.keys()) #OrderedDict
        if cfg.train.selected_frame != 'all':
            assert os.path.isfile(cfg.train.selected_frame)
            selected_frames = [l.strip() for l in open(cfg.train.selected_frame,'r').readlines()]
            if cfg.test.type=='novelview_all': #eval novelview_all
                selected_frames = sorted(list(set([self.get_frame_camera(s)[0] for s in selected_frames])))
                views = set([self.get_frame_camera(f)[1] for f in frames])
                #skip 16 quick dirty solution
                '''
                skip = 16
                frames = []
                for s in selected_frames[::skip]:
                    frames.extend([f'Camera_B{v:1d}/{s:06d}.jpg' for v in views])
                print(frames)
                '''
                frames = [f for f in frames if self.get_frame_camera(f)[0] in selected_frames]
                print(f'number of different frames={len(selected_frames)} in the training set, we test {len(frames)} novel-view images')
            else:
                frames = [f for f in selected_frames if f in frames]
                assert  len(frames)==len(selected_frames), (len(frames), len(selected_frames))
        return frames
    
    def parse_view_from_frame(self, frame_name):
        if 'view' in frame_name:
            #frame_000008_view_00.png
            view = int(frame_name.split('view_')[1][:2])
        elif 'Camera' in frame_name:
            #Camera_B4/000438.jpg
            view = int(frame_name.split('/')[0].split('Camera_B')[1])
        else:
            view = 0 #legacy use Camera 00
        return view

    def get_total_views(self):
        views = [self.parse_view_from_frame(fn) for fn in self.framelist]
        return sorted(list(set(views)))

    def query_dst_skeleton(self, frame_name):
        return {
            'poses': self.mesh_infos[frame_name]['poses'].astype('float32'),
            'dst_tpose_joints': \
                self.mesh_infos[frame_name]['tpose_joints'].astype('float32'),
            'bbox': self.mesh_infos[frame_name]['bbox'].copy(),
            'Rh': self.mesh_infos[frame_name]['Rh'].astype('float32'),
            'Th': self.mesh_infos[frame_name]['Th'].astype('float32')
        }

    @staticmethod
    def select_rays(select_inds, rays_o, rays_d, rays_d_camera, ray_img, near, far):
        rays_o = rays_o[select_inds]
        rays_d = rays_d[select_inds]
        ray_img = ray_img[select_inds]
        rays_d_camera = rays_d_camera[select_inds]
        near = near[select_inds]
        far = far[select_inds]
        return rays_o, rays_d, rays_d_camera, ray_img, near, far
    
    def get_patch_ray_indices(
            self, 
            N_patch, 
            ray_mask, 
            subject_mask, 
            bbox_mask,
            patch_size, 
            H, W):

        assert subject_mask.dtype == np.bool
        assert bbox_mask.dtype == np.bool

        bbox_exclude_subject_mask = np.bitwise_and(
            bbox_mask,
            np.bitwise_not(subject_mask)
        )

        list_ray_indices = []
        list_mask = []
        list_xy_min = []
        list_xy_max = []

        total_rays = 0
        patch_div_indices = [total_rays]
        for _ in range(N_patch):
            # let p = cfg.patch.sample_subject_ratio
            # prob p: we sample on subject area
            # prob (1-p): we sample on non-subject area but still in bbox
            if np.random.rand(1)[0] < cfg.patch.sample_subject_ratio:
                candidate_mask = subject_mask
            else:
                candidate_mask = bbox_exclude_subject_mask

            ray_indices, mask, xy_min, xy_max = \
                self._get_patch_ray_indices(ray_mask, candidate_mask, 
                                            patch_size, H, W)

            assert len(ray_indices.shape) == 1
            total_rays += len(ray_indices) #patch_size e.g. 32*32 or 20*20

            list_ray_indices.append(ray_indices)
            list_mask.append(mask) # mask is shaped as (32,32), only those within the box are True
            list_xy_min.append(xy_min) # cooridante in the (512,512) image
            list_xy_max.append(xy_max) # cooridante in the (512,512) image
            
            patch_div_indices.append(total_rays) #

        select_inds = np.concatenate(list_ray_indices, axis=0) #indices in the cropped bbox!  N_patch*patch_h*patch_w
        patch_info = {
            'mask': np.stack(list_mask, axis=0), #(6,20,20)N_patch, Patch_h, Patch_w #mask for the match (32,32)
            'xy_min': np.stack(list_xy_min, axis=0), #N_patch, xmin, ymin
            'xy_max': np.stack(list_xy_max, axis=0) #N_patch, xmax, ymax in the uncropped (512,512)
        }
        patch_div_indices = np.array(patch_div_indices) 

        return select_inds, patch_info, patch_div_indices


    def _get_patch_ray_indices(
            self, 
            ray_mask, 
            candidate_mask, 
            patch_size, 
            H, W):

        assert len(ray_mask.shape) == 1
        assert ray_mask.dtype == np.bool
        assert candidate_mask.dtype == np.bool

        valid_ys, valid_xs = np.where(candidate_mask)

        # determine patch center
        select_idx = np.random.choice(valid_ys.shape[0], 
                                      size=[1], replace=False)[0]
        center_x = valid_xs[select_idx]
        center_y = valid_ys[select_idx]

        # determine patch boundary
        half_patch_size = patch_size // 2
        x_min = np.clip(a=center_x-half_patch_size, 
                        a_min=0, 
                        a_max=W-patch_size)
        x_max = x_min + patch_size
        y_min = np.clip(a=center_y-half_patch_size,
                        a_min=0,
                        a_max=H-patch_size)
        y_max = y_min + patch_size

        sel_ray_mask = np.zeros_like(candidate_mask)
        sel_ray_mask[y_min:y_max, x_min:x_max] = True
        #####################################################
        ## Below we determine the selected ray indices
        ## and patch valid mask

        sel_ray_mask = sel_ray_mask.reshape(-1)
        inter_mask = np.bitwise_and(sel_ray_mask, ray_mask) #(512*512,)
        select_masked_inds = np.where(inter_mask) #[1-dim,1-dim]

        masked_indices = np.cumsum(ray_mask) - 1 #local ind within the valid area of ray_mask
        select_inds = masked_indices[select_masked_inds]
        
        inter_mask = inter_mask.reshape(H, W)

        return select_inds, \
                inter_mask[y_min:y_max, x_min:x_max], \
                np.array([x_min, y_min]), np.array([x_max, y_max])
    
    def perturb_pixel_according_to_dir(self, img, mask, rays_d):
        scale = rays_d@np.array([1,1,1])
        scale = (scale+2)/2.5
        # print(scale.mean(), scale.min(), scale.max())
        perturbed_img = img.copy()
        perturbed_img = np.clip(perturbed_img*scale[...,None],0,1)
        img = perturbed_img*mask+img*(1-mask)
        return img.astype(np.float32)

    def load_image(self, frame_name, bg_color):
        if self.source_path is None:
            imagepath = os.path.join(self.image_dir, '{}.png'.format(frame_name))
            maskpath = os.path.join(self.dataset_path, 
                                    'masks', 
                                    '{}.png'.format(frame_name))
            alpha_mask = np.array(load_image(maskpath))
            if alpha_mask.max()==1:
                alpha_mask *= 255
        else:
            imagepath = os.path.join(self.image_dir, frame_name)
            alpha_mask = to_3ch_image(get_mask(self.source_path, img_name=frame_name))

        orig_img = np.array(load_image(imagepath))
        
        # undistort image
        if frame_name in self.cameras and 'distortions' in self.cameras[frame_name]:
            K = self.cameras[frame_name]['intrinsics']
            D = self.cameras[frame_name]['distortions']
            orig_img = cv2.undistort(orig_img, K, D)
            alpha_mask = cv2.undistort(alpha_mask, K, D)

        alpha_mask = alpha_mask / 255.

        if cfg.experiments.color_perturbation != 'empty':
            if cfg.experiments.color_perturbation=='per_pixel':
                pass #leave for perturb_pixel_according_to_dir(self, ) 
            elif cfg.experiments.color_perturbation=='per_view':
                _, camera = self.get_frame_camera(frame_name)
                if cfg.experiments.color_perturbation_strength=='strong':
                    if camera==0:
                        #orig_img = np.clip(orig_img*0.8+0.2,0,1)
                        orig_img[:,:,0] =  np.clip(orig_img[:,:,0]*0.8-0.2,0,1)
                    elif camera==6:
                        orig_img[:,:,1] =  np.clip(orig_img[:,:,1]*1.2+0.2,0,1)
                    elif camera==12:
                        orig_img =  orig_img*0.5
                    elif camera==18:
                        pass
                    else:
                        pass
                elif cfg.experiments.color_perturbation_strength=='weak':
                    if camera==0:
                        #orig_img = np.clip(orig_img*0.8+0.2,0,1)
                        orig_img =  np.clip(orig_img*0.6,0,255).astype(np.uint8)
                    elif camera==6:
                        orig_img =  np.clip(orig_img*0.8,0,255).astype(np.uint8)
                    elif camera==12:
                        orig_img =  np.clip(orig_img*1.2,0,255).astype(np.uint8)
                    elif camera==18:
                        pass
                    else:
                        pass                    
            else:
                raise ValueError

        img = alpha_mask * orig_img + (1.0 - alpha_mask) * bg_color[None, None, :]
        if cfg.resize_img_scale != 1.:
            img = cv2.resize(img, None, 
                                fx=cfg.resize_img_scale,
                                fy=cfg.resize_img_scale,
                                interpolation=cv2.INTER_LANCZOS4) #H,W,3
            alpha_mask = cv2.resize(alpha_mask, None, 
                                    fx=cfg.resize_img_scale,
                                    fy=cfg.resize_img_scale,
                                    interpolation=cv2.INTER_LINEAR)                        
        return img, alpha_mask


    def get_total_frames(self):
        return len(self.framelist)


    def project_to_world3D(self, depth_img, frame_name, H, W, ray_mask, near, far):
        K = self.cameras[frame_name]['intrinsics'][:3, :3].copy()
        K[:2] *= cfg.resize_img_scale
        E = self.cameras[frame_name]['extrinsics']
        dst_skel_info = self.query_dst_skeleton(frame_name)
        E = apply_global_tfm_to_camera(
                E=E, 
                Rh=dst_skel_info['Rh'],
                Th=dst_skel_info['Th'])
        R = E[:3, :3]
        T = E[:3, 3]
        i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                        np.arange(H, dtype=np.float32),
                        indexing='xy')
        xy1 = np.stack([i, j, np.ones_like(i)], axis=2) #H,W,3
        pixel_camera = np.dot(xy1, np.linalg.inv(K).T)*depth_img[...,None] #H,W,3 * H,W,1 
        pixel_world = np.dot(pixel_camera - T.ravel(), R)
        near_, far_ = np.ones((H*W),dtype=np.float32)*np.inf, np.zeros((H*W),dtype=np.float32)
        #import ipdb; ipdb.set_trace()
        near_[ray_mask], far_[ray_mask] = near.view(-1), far.view(-1)
        near_, far_ = near_.reshape((H,W)), far_.reshape((H,W))
        valid = (depth_img>near_)*(depth_img<far_)
        return pixel_world, valid
        


    def sample_patch_rays(self, img, H, W,
                          subject_mask, bbox_mask, ray_mask,
                          rays_o, rays_d, rays_d_camera, ray_img, near, far): #bbox_mask (512,512) = reshaped ray_mask (512*512,)

        select_inds, patch_info, patch_div_indices = \
            self.get_patch_ray_indices(
                N_patch=cfg.patch.N_patches, 
                ray_mask=ray_mask, 
                subject_mask=subject_mask, 
                bbox_mask=bbox_mask,
                patch_size=cfg.patch.size, 
                H=H, W=W)

        rays_o, rays_d, rays_d_camera, ray_img, near, far = self.select_rays(
            select_inds, rays_o, rays_d, rays_d_camera, ray_img, near, far)
        
        targets = []
        for i in range(cfg.patch.N_patches):
            x_min, y_min = patch_info['xy_min'][i] 
            x_max, y_max = patch_info['xy_max'][i]
            targets.append(img[y_min:y_max, x_min:x_max])
        target_patches = np.stack(targets, axis=0) # (N_patches, P, P, 3)

        patch_masks = patch_info['mask']  # boolean array (N_patches, P, P)

        return rays_o, rays_d, rays_d_camera, ray_img, near, far, \
                target_patches, patch_masks, patch_div_indices

    def __len__(self):
        return self.get_total_frames()

    def __getitem__(self, idx):
        frame_name = self.framelist[idx]
        results = {
            'frame_name': frame_name,
            'frame_id': self.get_frame_camera(frame_name)[0],
            'dir_idx': torch.tensor([self.views.index(self.parse_view_from_frame(frame_name))], dtype=torch.long)
        }
        
        if cfg.multihead.split == 'view':
            if self.ray_shoot_mode=='image':
                if cfg.test.head_id == -1: #multiple outputs
                    results['head_id'] = int(cfg.test.head_id)
                    '''
                    view_id = self.parse_view_from_frame(frame_name)
                    results['head_id'] = self.views.index(view_id)
                    '''
                else:
                    results['head_id'] = int(cfg.test.head_id)
            elif self.ray_shoot_mode=='patch': #training
                view_id = self.parse_view_from_frame(frame_name)
                results['head_id'] = self.views.index(view_id)
            else:
                raise 
        elif cfg.multihead.split == 'argmin':
            if self.ray_shoot_mode=='image':
                if cfg.test.head_id == -1: #auto
                    #raise ValueError
                    results['head_id'] = int(cfg.test.head_id)
                else:
                    results['head_id'] = int(cfg.test.head_id)
            elif self.ray_shoot_mode=='patch': #training
                results['head_id'] = -1 #all
            else:
                raise   
        elif cfg.multihead.split == 'random':
            if self.ray_shoot_mode=='image':
                if cfg.test.head_id == -1: #auto
                    #raise ValueError
                    results['head_id'] = int(cfg.test.head_id)
                else:
                    results['head_id'] = int(cfg.test.head_id)
            elif self.ray_shoot_mode=='patch': #training
                results['head_id'] = np.random.randint(cfg.multihead.head_num)
            else:
                raise   
        elif os.path.isfile(cfg.multihead.split):
            if self.ray_shoot_mode=='image':
                if cfg.test.head_id == -1: #auto
                    #raise ValueError
                    results['head_id'] = int(cfg.test.head_id)
                else:
                    results['head_id'] = int(cfg.test.head_id)
            elif self.ray_shoot_mode=='patch': #training
                assert self.sample2head_id is not None
                results['head_id'] = self.sample2head_id[frame_name]
            else:
                raise               




        if self.bgcolor is None:
            bgcolor = (np.random.rand(3) * 255.).astype('float32')
        else:
            bgcolor = np.array(self.bgcolor, dtype='float32')

        img, alpha = self.load_image(frame_name, bgcolor)
        img = (img / 255.).astype('float32')

        results['raw_rgbs'] = img
        H, W = img.shape[0:2]

        dst_skel_info = self.query_dst_skeleton(frame_name)
        dst_bbox = dst_skel_info['bbox']
        dst_poses = dst_skel_info['poses']
        dst_tpose_joints = dst_skel_info['dst_tpose_joints']

        assert frame_name in self.cameras
        K = self.cameras[frame_name]['intrinsics'][:3, :3].copy()
        K[:2] *= cfg.resize_img_scale

        E = self.cameras[frame_name]['extrinsics']
        R = E[:3, :3]
        T = E[:3, 3]
        _, rays_d_camera = get_rays_from_KRT(H, W, K, R, T)


        E = apply_global_tfm_to_camera(
                E=E, 
                Rh=dst_skel_info['Rh'],
                Th=dst_skel_info['Th'])
        R = E[:3, :3]
        T = E[:3, 3]
        rays_o, rays_d = get_rays_from_KRT(H, W, K, R, T)

        if cfg.experiments.color_perturbation=='per_pixel':
            if cfg.experiments.color_perturbation_according_to=='camera':
                img = self.perturb_pixel_according_to_dir(img, alpha, rays_d_camera)
            elif cfg.experiments.color_perturbation_according_to=='camera_body':
                img = self.perturb_pixel_according_to_dir(img, alpha, rays_d)
            else:
                raise ValueError

        ray_img = img.reshape(-1, 3) 
        rays_d_camera = rays_d_camera.reshape(-1, 3)
        rays_o = rays_o.reshape(-1, 3) # (H, W, 3) --> (N_rays, 3)
        rays_d = rays_d.reshape(-1, 3)



        # (selected N_samples, ), (selected N_samples, ), (N_samples, )
        near, far, ray_mask = rays_intersect_3d_bbox(dst_bbox, rays_o, rays_d)
        rays_o = rays_o[ray_mask]
        rays_d = rays_d[ray_mask]
        ray_img = ray_img[ray_mask]
        rays_d_camera = rays_d_camera[ray_mask]

        if os.environ.get('TEST_DIR', '') != '':
            K_ = self.test_dir_camera['intrinsics'][:3, :3].copy()
            K_[:2] *= cfg.resize_img_scale

            E_ = self.test_dir_camera['extrinsics']
            E_ = apply_global_tfm_to_camera(
                    E=E_, 
                    Rh=dst_skel_info['Rh'],
                    Th=dst_skel_info['Th'])
            R_ = E_[:3, :3]
            T_ = E_[:3, 3]
            rays_o_, rays_d_ = get_rays_from_KRT(H, W, K_, R_, T_)
            rays_d_ = rays_d_.reshape(-1, 3)
            rays_d_ = rays_d_[ray_mask]
            results.update({'rays_d_':rays_d_})
            results['dir_idx'] = torch.tensor([self.views.index(self.test_dir)], dtype=torch.long)

        near = near[:, None].astype('float32')
        far = far[:, None].astype('float32')

        if self.ray_shoot_mode == 'image':
            pass
        elif self.ray_shoot_mode == 'patch':
            rays_o, rays_d, rays_d_camera, ray_img, near, far, \
            target_patches, patch_masks, patch_div_indices = \
                self.sample_patch_rays(img=img, H=H, W=W,
                                       subject_mask=alpha[:, :, 0] > 0.,
                                       bbox_mask=ray_mask.reshape(H, W),
                                       ray_mask=ray_mask,
                                       rays_o=rays_o, 
                                       rays_d=rays_d, rays_d_camera=rays_d_camera,
                                       ray_img=ray_img, 
                                       near=near, 
                                       far=far)
        else:
            assert False, f"Ivalid Ray Shoot Mode: {self.ray_shoot_mode}"
    
        batch_rays = np.stack([rays_o, rays_d, rays_d_camera], axis=0) #
        if 'rays' in self.keyfilter:
            results.update({
                'img_width': W,
                'img_height': H,
                'ray_mask': ray_mask,
                'rays': batch_rays,
                'near': near,
                'far': far,
                'bgcolor': bgcolor})

            if self.ray_shoot_mode == 'patch':
                results.update({
                    'patch_div_indices': patch_div_indices,
                    'patch_masks': patch_masks,
                    'target_patches': target_patches})

        if 'target_rgbs' in self.keyfilter:
            results['target_rgbs'] = ray_img

        if 'motion_bases' in self.keyfilter:
            dst_Rs, dst_Ts = body_pose_to_body_RTs(
                    dst_poses, dst_tpose_joints
                )
            cnl_gtfms = get_canonical_global_tfms(
                            self.canonical_joints)
            results.update({
                'dst_Rs': dst_Rs,
                'dst_Ts': dst_Ts,
                'cnl_gtfms': cnl_gtfms
            })

            if cfg.rgb_history.last_num > 0: #only support consecutive frames
                frame_id, camera_id = self.get_frame_camera(frame_name)
                dst_Rs_history, dst_Ts_history, dst_posevec_last = [], [], []
                w2c_history = []
                frame_name_history, rgb_history = [], []
                for i in np.arange(1,cfg.rgb_history.last_num+1)*cfg.rgb_history.step:
                    frame_id_last = max(frame_id - i,0)
                    frame_name_last = self.get_framename(frame_id_last, camera_id)
                    #frame_name_last = self.framelist_all[idx*self.skip-i] if idx*self.skip-i>=0 else self.framelist_all[0]
                    dst_skel_info_last = self.query_dst_skeleton(frame_name_last)
                    dst_poses_last, dst_tpose_joints_last = dst_skel_info_last['poses'], dst_skel_info_last['dst_tpose_joints']
                    dst_Rs_last, dst_Ts_last = body_pose_to_body_RTs(
                            dst_poses_last, dst_tpose_joints_last
                        )

                    dst_Rs_history.append(dst_Rs_last)
                    dst_Ts_history.append(dst_Ts_last)
                    dst_posevec_last.append(dst_poses_last[3:] + 1e-2)
                    
                    #multiple camera
                    multiview_w2c = []
                    frame_name_history_multiview, rgb_history_multiview = [], []
                    for cid in self.views_all:
                        frame_name_last = self.get_framename(frame_id_last, cid)
                        frame_name_history_multiview.append(frame_name_last)
                        K_last = self.cameras[frame_name_last]['intrinsics'][:3, :3].copy()
                        K_last[:2] *= cfg.resize_img_scale
                        E_last = self.cameras[frame_name_last]['extrinsics']
                        E_last = apply_global_tfm_to_camera(
                                E=E_last, 
                                Rh=dst_skel_info_last['Rh'],
                                Th=dst_skel_info_last['Th'])
                        multiview_w2c.append(K_last@E_last[:3,:].astype(np.float32))

                        if cfg.rgb_history.precompute_dir != 'empty' and cfg.rgb_history.feature_cfg.layer != -1:
                            feature_path = os.path.join(cfg.rgb_history.precompute_dir, frame_name_last.split('.')[0]+'.bin')
                            feature_last = torch.load(feature_path)
                            rgb_history_multiview.append(feature_last) #(H,W,C)
                        else:
                            img_last, _ = self.load_image(frame_name_last, bgcolor)
                            img_last = (img_last / 255.).astype('float32') #(512,512,(0~1)) #unnormalized
                            rgb_history_multiview.append(img_last)
                    w2c_history.append(np.stack(multiview_w2c, axis=0)) #(num_View, 3,4)
                    frame_name_history.append(frame_name_history_multiview)
                    rgb_history.append(np.stack(rgb_history_multiview,axis=0)) #V,H,W,C

                results.update({
                    'dst_Rs_history': np.stack(dst_Rs_history, axis=0),
                    'dst_Ts_history': np.stack(dst_Ts_history, axis=0),
                    'dst_posevec_history': np.stack(dst_posevec_last, axis=0),
                    'w2c_history': np.stack(w2c_history, axis=0), #(N,num_view,3,4)
                    'frame_name_history': frame_name_history,
                    'rgb_history': np.stack(rgb_history, axis=0) #(N, num_view, 512, 512, 3)
                })

        if 'motion_weights_priors' in self.keyfilter:
            results['motion_weights_priors'] = self.motion_weights_priors.copy()

        # get the bounding box of canonical volume
        if 'cnl_bbox' in self.keyfilter:
            min_xyz = self.canonical_bbox['min_xyz'].astype('float32')
            max_xyz = self.canonical_bbox['max_xyz'].astype('float32')
            results.update({
                'cnl_bbox_min_xyz': min_xyz,
                'cnl_bbox_max_xyz': max_xyz,
                'cnl_bbox_scale_xyz': 2.0 / (max_xyz - min_xyz)
            })
            assert np.all(results['cnl_bbox_scale_xyz'] >= 0)

        if 'dst_posevec_69' in self.keyfilter:
            # 1. ignore global orientation
            # 2. add a small value to avoid all zeros
            dst_posevec_69 = dst_poses[3:] + 1e-2
            results.update({
                'dst_posevec': dst_posevec_69,
            })
        if self.pose_condition_list is not None:
            results['pose_condition'] = self.pose_condition_list[idx]
            if cfg.pose_condition_random_mask != 'empty' and self.ray_shoot_mode=='patch': #during training
                if cfg.pose_condition_random_mask=='second-half':
                    if np.random.rand()<cfg.pose_condition_mask_prob:
                        results['pose_condition'][len(results['pose_condition'])//2:] = 0
                else:
                    raise ValueError
        if self.pose_condition_list_cmlp is not None:
            results['pose_condition_cmlp'] = self.pose_condition_list_cmlp[idx]
        elif 'pose_condition' in results:
            results['pose_condition_cmlp'] = results['pose_condition']
        return results
