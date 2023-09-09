import torch, os
import torch.nn as nn
import torch.nn.functional as F

from core.utils.network_util import MotionBasisComputer
from core.nets.human_nerf.component_factory import \
    load_positional_embedder, \
    load_canonical_mlp, \
    load_mweight_vol_decoder, \
    load_pose_decoder, \
    load_non_rigid_motion_mlp, \
    load_vocab_embedder

from configs import cfg


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        # motion basis computer
        self.motion_basis_computer = MotionBasisComputer(
                                        total_bones=cfg.total_bones)

        # motion weight volume
        self.mweight_vol_decoder = load_mweight_vol_decoder(cfg.mweight_volume.module)(
            embedding_size=cfg.mweight_volume.embedding_size,
            volume_size=cfg.mweight_volume.volume_size,
            total_bones=cfg.total_bones
        )

        # non-rigid motion st positional encoding
        self.get_non_rigid_embedder = \
            load_positional_embedder(cfg.non_rigid_embedder.module)

        # non-rigid motion MLP
        _, non_rigid_pos_embed_size = \
            self.get_non_rigid_embedder(cfg.non_rigid_motion_mlp.multires, 
                                        cfg.non_rigid_motion_mlp.i_embed)
        self.non_rigid_mlp = \
            load_non_rigid_motion_mlp(cfg.non_rigid_motion_mlp.module)(
                pos_embed_size=non_rigid_pos_embed_size,
                condition_code_size=cfg.non_rigid_motion_mlp.condition_code_size,
                mlp_width=cfg.non_rigid_motion_mlp.mlp_width,
                mlp_depth=cfg.non_rigid_motion_mlp.mlp_depth,
                skips=cfg.non_rigid_motion_mlp.skips,
                multihead_enable=cfg.non_rigid_motion_mlp.multihead.enable,
                multihead_depth=cfg.non_rigid_motion_mlp.multihead.head_depth,
                multihead_num=cfg.multihead.head_num,
                last_linear_scale=cfg.non_rigid_motion_mlp.last_linear_scale,
                mlp_depth_plus=cfg.non_rigid_motion_mlp.mlp_depth_plus)
        self.non_rigid_mlp = \
            nn.DataParallel(
                self.non_rigid_mlp,
                device_ids=cfg.secondary_gpus,
                output_device=cfg.secondary_gpus[0])

        # canonical positional encoding
        get_embedder = load_positional_embedder(cfg.embedder.module)
        cnl_pos_embed_fn, cnl_pos_embed_size = \
            get_embedder(cfg.canonical_mlp.multires, 
                         cfg.canonical_mlp.i_embed)
        self.pos_embed_fn = cnl_pos_embed_fn
        
        if cfg.canonical_mlp.view_dir:
            if cfg.canonical_mlp.view_embed == 'mlp':
                get_embedder_dir = load_positional_embedder(cfg.embedder.module)
                self.dir_embed_fn, cnl_dir_embed_size = \
                    get_embedder_dir(cfg.canonical_mlp.multires_dir, 
                                cfg.canonical_mlp.i_embed)
            elif cfg.canonical_mlp.view_embed == 'vocab':
                get_embedder_dir = load_vocab_embedder(cfg.vocab_embedder.module)
                self.dir_embed_fn, cnl_dir_embed_size = \
                    get_embedder_dir(cfg.canonical_mlp.view_vocab_n, 
                                    cfg.canonical_mlp.view_vocab_dim)
            else:
                raise ValueError
        else:
            self.dir_embed_fn, cnl_dir_embed_size = None, -1

        # canonical mlp 
        skips = [4]
        self.cnl_mlp = \
            load_canonical_mlp(cfg.canonical_mlp.module)(
                input_ch=cnl_pos_embed_size, 
                mlp_depth=cfg.canonical_mlp.mlp_depth, 
                mlp_width=cfg.canonical_mlp.mlp_width,
                view_dir=cfg.canonical_mlp.view_dir, 
                input_ch_dir=cnl_dir_embed_size, 
                pose_color=cfg.canonical_mlp.pose_color,
                pose_ch=cfg.canonical_mlp.pose_ch,
                skips=skips,
                multihead_enable=cfg.canonical_mlp.multihead.enable,
                multihead_depth=cfg.canonical_mlp.multihead.head_depth,
                multihead_num=cfg.multihead.head_num,
                last_linear_scale=cfg.canonical_mlp.last_linear_scale,
                mlp_depth_plus=cfg.canonical_mlp.mlp_depth_plus)
        self.cnl_mlp = \
            nn.DataParallel(
                self.cnl_mlp,
                device_ids=cfg.secondary_gpus,
                output_device=cfg.primary_gpus[0])

        # pose decoder MLP
        self.pose_decoder = \
            load_pose_decoder(cfg.pose_decoder.module)(
                embedding_size=cfg.pose_decoder.embedding_size,
                mlp_width=cfg.pose_decoder.mlp_width,
                mlp_depth=cfg.pose_decoder.mlp_depth)
    

    def deploy_mlps_to_secondary_gpus(self):
        self.cnl_mlp = self.cnl_mlp.to(cfg.secondary_gpus[0])
        if self.non_rigid_mlp:
            self.non_rigid_mlp = self.non_rigid_mlp.to(cfg.secondary_gpus[0])

        return self


    def _query_mlp(
            self,
            pos_xyz,
            pos_embed_fn, 
            non_rigid_pos_embed_fn,
            non_rigid_mlp_input,
            dir_xyz, dir_idx, 
            dir_embed_fn, 
            head_id, pose_latent):

        # (N_rays, N_samples, 3) --> (N_rays x N_samples, 3)
        pos_flat = torch.reshape(pos_xyz, [-1, pos_xyz.shape[-1]])
        if cfg.canonical_mlp.view_embed == 'mlp':
            dir_flat = torch.reshape(dir_xyz, [-1, dir_xyz.shape[-1]])
        elif cfg.canonical_mlp.view_embed == 'vocab':
            dir_flat = torch.reshape(dir_idx, [-1,]) #N,
        chunk = cfg.netchunk_per_gpu*len(cfg.secondary_gpus)

        result = self._apply_mlp_kernals(
                        pos_flat=pos_flat,
                        pos_embed_fn=pos_embed_fn,
                        non_rigid_mlp_input=non_rigid_mlp_input,
                        non_rigid_pos_embed_fn=non_rigid_pos_embed_fn,
                        dir_flat=dir_flat, 
                        dir_embed_fn=dir_embed_fn,
                        chunk=chunk, head_id=head_id, pose_latent=pose_latent)

        output = {}

        raws_flat = result['raws']
        xyzs_flat = result['xyzs'] #batch, 3 or [(batch,3), (batch,3)]
        if type(raws_flat)==list:
            assert head_id.min()==-1, head_id
            assert type(xyzs_flat)==list, len(xyzs_flat)==len(raws_flat)
            output['raws'] = [torch.reshape(
                                raws_flat_, 
                                list(pos_xyz.shape[:-1]) + [raws_flat_.shape[-1]]) for raws_flat_ in raws_flat]     
            output['xyzs'] = [torch.reshape(xyzs_flat_, list(pos_xyz.shape[:-1]) + [xyzs_flat_.shape[-1]]) for xyzs_flat_ in xyzs_flat]       
        else:
            output['raws'] = torch.reshape(
                                raws_flat, 
                                list(pos_xyz.shape[:-1]) + [raws_flat.shape[-1]]) 
            output['xyzs'] = torch.reshape(
                                xyzs_flat, 
                                list(pos_xyz.shape[:-1]) + [xyzs_flat.shape[-1]])                                

        return output


    @staticmethod
    def _expand_input(input_data, total_elem):
        assert input_data.shape[0] == 1
        input_size = input_data.shape[1]
        return input_data.expand((total_elem, input_size))


    def _apply_mlp_kernals(
            self, 
            pos_flat,
            pos_embed_fn,
            non_rigid_mlp_input,
            non_rigid_pos_embed_fn,
            dir_flat, 
            dir_embed_fn,
            chunk, head_id, pose_latent):

        if cfg.canonical_mlp.multihead.enable and head_id==-1:
            raws_list = [[] for i in range(cfg.multihead.head_num)]
            xyz_list = [[] for i in range(cfg.multihead.head_num)]
        else:
            raws = []
            xyzs = []
        output = {}
        # iterate ray samples by trunks
        for i in range(0, pos_flat.shape[0], chunk):
            start = i
            end = i + chunk
            if end > pos_flat.shape[0]:
                end = pos_flat.shape[0]
            total_elem = end - start

            xyz, dir_ = pos_flat[start:end], dir_flat[start:end]
            head_id_expanded = self._expand_input(head_id[None, None, ...], total_elem)
            if not cfg.ignore_non_rigid_motions:
                non_rigid_embed_xyz = non_rigid_pos_embed_fn(xyz)
                result = self.non_rigid_mlp(
                    pos_embed=non_rigid_embed_xyz,
                    pos_xyz=xyz,
                    condition_code=self._expand_input(non_rigid_mlp_input, total_elem),
                    head_id=head_id_expanded
                )
                xyz = result['xyz'] #B, 3 or list
            
            if cfg.canonical_mlp.view_dir:
                dir_embed = dir_embed_fn(dir_) #vocab: anyshape
            else:
                dir_embed = None

            if cfg.canonical_mlp.multihead.enable and head_id.min()==-1: #multiple outputs
                if type(xyz) == list:
                    assert len(xyz) == cfg.multihead.head_num
                    for head_id_, xyz_ in enumerate(xyz):
                        xyz_embedded = pos_embed_fn(xyz_) #B*n_head (if argmin), 3*2*10 
                        new_head_id_expanded = torch.ones_like(head_id_expanded)*head_id_
                        xyz_list[head_id_].append(xyz_)
                        raws_list[head_id_] += [self.cnl_mlp(
                                    pos_embed=xyz_embedded, dir_embed=dir_embed,  
                                    head_id=new_head_id_expanded)] #N*num_head, 4                     
                else:
                    xyz_embedded = pos_embed_fn(xyz) #B*n_head (if argmin), 3*2*10 
                    raws_list_ = self.cnl_mlp(
                                pos_embed=xyz_embedded, dir_embed=dir_embed, 
                                head_id=head_id_expanded) #N*num_head, 4  
                    for head_id_, o in enumerate(raws_list_):
                        raws_list[head_id_].append(o)     
                        xyz_list[head_id_].append(xyz)                      
            else:
                xyz_embedded = pos_embed_fn(xyz) #B*n_head (if argmin), 3*2*10
                xyzs.append(xyz)
                raws += [self.cnl_mlp(
                            pos_embed=xyz_embedded, dir_embed=dir_embed, 
                            pose_latent=self._expand_input(pose_latent, total_elem),
                            head_id=head_id_expanded)] #N*num_head, 4
        
        if cfg.canonical_mlp.multihead.enable  and head_id.min()==-1:
            raws_list = [torch.cat(raws, dim=0).to(cfg.primary_gpus[0]) for raws in raws_list] 
            output['raws'] = raws_list
            xyz_list = [torch.cat(xyz, dim=0).to(cfg.primary_gpus[0]) for xyz in xyz_list]
            output['xyzs'] = xyz_list
        else:
            output['raws'] = torch.cat(raws, dim=0).to(cfg.primary_gpus[0]) #N*num_head, 4
            output['xyzs'] = torch.cat(xyzs, dim=0).to(cfg.primary_gpus[0])

        return output


    def _batchify_rays(self, rays_flat, **kwargs):
        all_ret = {}
        multi_outputs = False
        for i in range(0, rays_flat.shape[0], cfg.chunk):
            ret = self._render_rays(rays_flat[i:i+cfg.chunk], **kwargs)
            for k in ret: #rgb, depth
                if type(ret[k])==list:
                    multi_outputs = True
                    if k not in all_ret:
                        if multi_outputs:
                            all_ret[k] = [[] for _ in range(cfg.multihead.head_num)] 
                    for head_id in range(cfg.multihead.head_num): 
                        all_ret[k][head_id].append(ret[k][head_id])                  
                else:
                    if k not in all_ret:
                        all_ret[k] = []
                    all_ret[k].append(ret[k]) 
        if multi_outputs:
            all_ret = {k : [torch.cat(x, 0) for x in all_ret[k]] for k in all_ret} #'rgb':[tensor1-for head0, tensor2]
        else:
            all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret} 
        return all_ret


    @staticmethod
    def _raw2outputs(raw, raw_mask, z_vals, rays_d, xyz, bgcolor=None):
        def _raw2alpha(raw, dists, act_fn=F.relu):
            return 1.0 - torch.exp(-act_fn(raw)*dists)

        dists = z_vals[...,1:] - z_vals[...,:-1]

        infinity_dists = torch.Tensor([1e10])
        infinity_dists = infinity_dists.expand(dists[...,:1].shape).to(dists)
        dists = torch.cat([dists, infinity_dists], dim=-1) 
        dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

        rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
        alpha = _raw2alpha(raw[...,3], dists)  # [N_rays, N_samples]
        alpha = alpha * raw_mask[:, :, 0] #foreground probability

        weights = alpha * torch.cumprod(
            torch.cat([torch.ones((alpha.shape[0], 1)).to(alpha), 
                       1.-alpha + 1e-10], dim=-1), dim=-1)[:, :-1] #[N_rays, n_sample]
        rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, n_samples, 3]

        depth_map = torch.sum(weights * z_vals, -1)
        acc_map = torch.sum(weights, -1)

        rgb_map = rgb_map + (1.-acc_map[...,None]) * bgcolor[None, :]/255.
        
        #xyz [N_rays, n_sample, 3]
        
        
        weights_max, indices = weights.max(dim=1) #N_rays
        indices = indices[:,None,None] #N_rays, 1, 1
        cnl_xyz = torch.gather(xyz, dim=1, index=indices.tile([1,1,xyz.shape[-1]]))
        cnl_rgb = torch.gather(rgb, dim=1, index=indices.tile([1,1,rgb.shape[-1]]))
        return rgb_map, acc_map, weights, depth_map, cnl_xyz.squeeze(1), cnl_rgb.squeeze(1), weights_max, rgb
        


    @staticmethod
    def _sample_motion_fields(
            pts,
            motion_scale_Rs, 
            motion_Ts, 
            motion_weights_vol,
            cnl_bbox_min_xyz, cnl_bbox_scale_xyz,
            output_list):
        orig_shape = list(pts.shape)
        pts = pts.reshape(-1, 3) # [N_rays x N_samples, 3]

        # remove BG channel
        motion_weights = motion_weights_vol[:-1] 

        weights_list = []
        for i in range(motion_weights.size(0)):
            pos = torch.matmul(motion_scale_Rs[i, :, :], pts.T).T + motion_Ts[i, :]
            pos = (pos - cnl_bbox_min_xyz[None, :]) \
                            * cnl_bbox_scale_xyz[None, :] - 1.0 
            weights = F.grid_sample(input=motion_weights[None, i:i+1, :, :, :], 
                                    grid=pos[None, None, None, :, :],           
                                    padding_mode='zeros', align_corners=True)
            weights = weights[0, 0, 0, 0, :, None] 
            weights_list.append(weights) 
        backwarp_motion_weights = torch.cat(weights_list, dim=-1)
        total_bases = backwarp_motion_weights.shape[-1]

        backwarp_motion_weights_sum = torch.sum(backwarp_motion_weights, 
                                                dim=-1, keepdim=True)
        weighted_motion_fields = []
        for i in range(total_bases):
            pos = torch.matmul(motion_scale_Rs[i, :, :], pts.T).T + motion_Ts[i, :]
            weighted_pos = backwarp_motion_weights[:, i:i+1] * pos
            weighted_motion_fields.append(weighted_pos)
        x_skel = torch.sum(
                        torch.stack(weighted_motion_fields, dim=0), dim=0
                        ) / backwarp_motion_weights_sum.clamp(min=0.0001)
        fg_likelihood_mask = backwarp_motion_weights_sum

        x_skel = x_skel.reshape(orig_shape[:2]+[3])
        backwarp_motion_weights = \
            backwarp_motion_weights.reshape(orig_shape[:2]+[total_bases])
        fg_likelihood_mask = fg_likelihood_mask.reshape(orig_shape[:2]+[1])

        results = {}
        
        if 'x_skel' in output_list: # [N_rays x N_samples, 3]
            results['x_skel'] = x_skel
        if 'fg_likelihood_mask' in output_list: # [N_rays x N_samples, 1]
            results['fg_likelihood_mask'] = fg_likelihood_mask
        if 'backward_motion_weights' in output_list:
            results['backward_motion_weights'] = backwarp_motion_weights
        return results


    @staticmethod
    def _unpack_ray_batch(ray_batch):
        rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] 
        bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2]) 
        near, far = bounds[...,0], bounds[...,1] 
        return rays_o, rays_d, near, far


    @staticmethod
    def _get_samples_along_ray(N_rays, near, far):
        t_vals = torch.linspace(0., 1., steps=cfg.N_samples).to(near)
        z_vals = near * (1.-t_vals) + far * (t_vals)
        return z_vals.expand([N_rays, cfg.N_samples]) 


    @staticmethod
    def _stratified_sampling(z_vals):
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        
        t_rand = torch.rand(z_vals.shape).to(z_vals)
        z_vals = lower + (upper - lower) * t_rand

        return z_vals


    def _render_rays(
            self, 
            ray_batch, 
            motion_scale_Rs,
            motion_Ts,
            motion_weights_vol,
            cnl_bbox_min_xyz,
            cnl_bbox_scale_xyz,
            pos_embed_fn,
            non_rigid_pos_embed_fn,
            dir_embed_fn,
            non_rigid_mlp_input=None,
            bgcolor=None, head_id=None,
            pose_latent=None, dir_idx=None, 
            **_):
        
        N_rays = ray_batch.shape[0]
        rays_o, rays_d, near, far = self._unpack_ray_batch(ray_batch)

        z_vals = self._get_samples_along_ray(N_rays, near, far)
        if cfg.perturb > 0.:
            z_vals = self._stratified_sampling(z_vals)

        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] #6144, 128, 3
        if os.environ.get('TEST_DIR', '') != '':
            dir_xyz = torch.nn.functional.normalize(_['rays_d_'].float())[:,None,:] # N,1,3
        else:
            dir_xyz = torch.nn.functional.normalize(rays_d)[:,None,:] # N,1,3
        dir_xyz = torch.tile(dir_xyz, [1,pts.shape[1],1])
        if dir_idx is None:
            dir_idx = torch.zeros([dir_xyz.shape[0]*dir_xyz.shape[1], 1], dtype=torch.long, device=dir_xyz.device)
        else:
            dir_idx = torch.tile(dir_idx[:, None], [dir_xyz.shape[0],pts.shape[1]]) #(1,1)->(N-ray, N-point)
        mv_output = self._sample_motion_fields(
                            pts=pts,
                            motion_scale_Rs=motion_scale_Rs[0], 
                            motion_Ts=motion_Ts[0], 
                            motion_weights_vol=motion_weights_vol,
                            cnl_bbox_min_xyz=cnl_bbox_min_xyz, 
                            cnl_bbox_scale_xyz=cnl_bbox_scale_xyz,
                            output_list=['x_skel', 'fg_likelihood_mask', 'backward_motion_weights'])
        pts_mask = mv_output['fg_likelihood_mask']
        cnl_pts = mv_output['x_skel']
        backward_motion_weights = mv_output['backward_motion_weights']

        query_result = self._query_mlp(
                                pos_xyz=cnl_pts, 
                                non_rigid_mlp_input=non_rigid_mlp_input,
                                pos_embed_fn=pos_embed_fn,
                                non_rigid_pos_embed_fn=non_rigid_pos_embed_fn,
                                dir_embed_fn=dir_embed_fn,
                                dir_xyz=dir_xyz, dir_idx=dir_idx, 
                                head_id=head_id, pose_latent=pose_latent)
        raw = query_result['raws']
        xyz = query_result['xyzs']
        
        if type(raw)==list:
            assert cfg.canonical_mlp.multihead.enable==True and head_id.min()==-1
            rgb_map, acc_map, depth_map = [], [], []
            cnl_xyz, cnl_rgb, cnl_weight = [], [], []
            weights = []
            rgb_on_rays = []
            for raw_head, xyz_head in zip(raw,xyz):
                output_dim = 4
                rgb_map_head, acc_map_head, weights_head, depth_map_head, cnl_xyz_head, cnl_rgb_head, cnl_weight_head, rgb_on_rays_head = \
                    self._raw2outputs(raw_head, pts_mask, z_vals, rays_d, xyz_head, bgcolor)     
                rgb_map.append(rgb_map_head) 
                acc_map.append(acc_map_head)
                depth_map.append(depth_map_head)
                cnl_xyz.append(cnl_xyz_head)
                cnl_rgb.append(cnl_rgb_head)
                cnl_weight.append(cnl_weight_head)
                weights.append(weights_head)
                rgb_on_rays.append(rgb_on_rays_head)
            #multi_outputs = True          
        else:
            rgb_map, acc_map, weights, depth_map, cnl_xyz, cnl_rgb, cnl_weight, rgb_on_rays = \
                self._raw2outputs(raw, pts_mask, z_vals, rays_d, xyz, bgcolor) #[N_rays, 3]
            #multi_outputs = False

        return {'rgb' : rgb_map,  
                'alpha' : acc_map, 
                'depth': depth_map,
                'weights_on_rays': weights,
                'xyz_on_rays': xyz, 'rgb_on_rays': rgb_on_rays, 
                'cnl_xyz':cnl_xyz, 'cnl_rgb':cnl_rgb, 'cnl_weight':cnl_weight,
                'backward_motion_weights': backward_motion_weights}#, multi_outputs


    def _get_motion_base(self, dst_Rs, dst_Ts, cnl_gtfms):
        motion_scale_Rs, motion_Ts = self.motion_basis_computer(
                                        dst_Rs, dst_Ts, cnl_gtfms)

        return motion_scale_Rs, motion_Ts


    @staticmethod
    def _multiply_corrected_Rs(Rs, correct_Rs):
        total_bones = cfg.total_bones - 1
        return torch.matmul(Rs.reshape(-1, 3, 3),
                            correct_Rs.reshape(-1, 3, 3)).reshape(-1, total_bones, 3, 3)

    
    def forward(self,
                rays, 
                dst_Rs, dst_Ts, cnl_gtfms,
                motion_weights_priors,
                dst_posevec=None,
                near=None, far=None,
                iter_val=1e7,
                **kwargs):
        dst_Rs=dst_Rs[None, ...]
        dst_Ts=dst_Ts[None, ...]
        dst_posevec=dst_posevec[None, ...]
        cnl_gtfms=cnl_gtfms[None, ...]
        motion_weights_priors=motion_weights_priors[None, ...]

        # correct body pose
        if iter_val >= cfg.pose_decoder.get('kick_in_iter', 0) and cfg.pose_decoder_off==False:
            pose_out = self.pose_decoder(dst_posevec)
            refined_Rs = pose_out['Rs']
            refined_Ts = pose_out.get('Ts', None)
            
            dst_Rs_no_root = dst_Rs[:, 1:, ...]
            dst_Rs_no_root = self._multiply_corrected_Rs(
                                        dst_Rs_no_root, 
                                        refined_Rs)
            dst_Rs = torch.cat(
                [dst_Rs[:, 0:1, ...], dst_Rs_no_root], dim=1)

            if refined_Ts is not None:
                dst_Ts = dst_Ts + refined_Ts

        non_rigid_pos_embed_fn, _ = \
            self.get_non_rigid_embedder(
                multires=cfg.non_rigid_motion_mlp.multires,                         
                is_identity=cfg.non_rigid_motion_mlp.i_embed,
                iter_val=iter_val,)

        if iter_val < cfg.non_rigid_motion_mlp.kick_in_iter:
            # mask-out non_rigid_mlp_input 
            non_rigid_mlp_input = torch.zeros_like(dst_posevec) * dst_posevec
        else:
            non_rigid_mlp_input = dst_posevec
        kwargs.update({
            "pos_embed_fn": self.pos_embed_fn,
            "dir_embed_fn": self.dir_embed_fn,
            "non_rigid_pos_embed_fn": non_rigid_pos_embed_fn,
            "non_rigid_mlp_input": non_rigid_mlp_input,
            "pose_latent": dst_posevec, 
        })

        motion_scale_Rs, motion_Ts = self._get_motion_base(
                                            dst_Rs=dst_Rs, 
                                            dst_Ts=dst_Ts, 
                                            cnl_gtfms=cnl_gtfms)
        motion_weights_vol = self.mweight_vol_decoder(
            motion_weights_priors=motion_weights_priors)
        motion_weights_vol=motion_weights_vol[0] # remove batch dimension

        kwargs.update({
            'motion_scale_Rs': motion_scale_Rs,
            'motion_Ts': motion_Ts,
            'motion_weights_vol': motion_weights_vol
        })

        rays_o, rays_d = rays
        rays_shape = rays_d.shape 

        rays_o = torch.reshape(rays_o, [-1,3]).float()
        rays_d = torch.reshape(rays_d, [-1,3]).float()
        packed_ray_infos = torch.cat([rays_o, rays_d, near, far], -1)

        all_ret = self._batchify_rays(packed_ray_infos, **kwargs)
        for k in all_ret:
            if type(all_ret[k])==list:
                all_ret[k] = [torch.reshape(x, list(rays_shape[:-1])+list(x.shape[1:])) for x in all_ret[k]] #merge from all gpus
            else:
                k_shape = list(rays_shape[:-1]) + list(all_ret[k].shape[1:])  #3, (num_head?)
                all_ret[k] = torch.reshape(all_ret[k], k_shape)

        return all_ret
