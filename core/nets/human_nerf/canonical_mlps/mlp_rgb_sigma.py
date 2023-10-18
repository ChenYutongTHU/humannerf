import torch, os
import torch.nn as nn

from core.utils.network_util import initseq
from core.nets.human_nerf.multihead import MultiheadMlp
from core.nets.human_nerf.selfattention import SelfAttention, MlpSeq
from core.nets.human_nerf.localize import localize_condition_code
from configs import cfg 

class CanonicalMLP(nn.Module):
    def __init__(self, mlp_depth=8, mlp_width=256, 
                 input_ch=3, skips=None, 
                 view_dir=False, input_ch_dir=3, 
                 pose_ch=69,
                 pose_color='wo',
                 multihead_enable=False, multihead_depth=1, multihead_num=4,
                 mlp_depth_plus=0,
                 last_linear_scale=1, **kwargs):
        super(CanonicalMLP, self).__init__()

        if skips is None:
            skips = [4]

        self.mlp_depth = mlp_depth
        self.mlp_width = mlp_width
        self.input_ch = input_ch
        self.view_dir = view_dir
        self.pose_color = pose_color #'dependent','ao' or 'wo'
        self.pose_ch = pose_ch
        self.input_ch_dir = input_ch_dir
        self.multihead_enable, self.multihead_num = multihead_enable, multihead_num
        self.multihead_depth = multihead_depth
        self.condition_code_dim = cfg.canonical_mlp.condition_code_dim
        if self.condition_code_dim>0:
            if cfg.canonical_mlp.condition_code_encoder.lower() == 'selfattention':
                self.condition_code_encoder = SelfAttention(input_dim=self.condition_code_dim,**cfg.canonical_mlp.selfattention)
                condition_ch = cfg.canonical_mlp.selfattention.output_dim
            elif cfg.canonical_mlp.condition_code_encoder.lower() == 'mlpseq':
                self.condition_code_encoder = MlpSeq(input_dim=self.condition_code_dim,**cfg.canonical_mlp.mlpseq)
                condition_ch = cfg.canonical_mlp.mlpseq.output_dim
            elif cfg.canonical_mlp.condition_code_encoder.lower() == 'none':
                self.condition_code_encoder = torch.nn.Identity()
                condition_ch = self.condition_code_dim
        else:
            self.condition_code_encoder = None
            condition_ch = 0
        
        self.time_input = cfg.canonical_mlp.time_input
        if self.time_input:
            time_ch = cfg.canonical_mlp.time_dim
        else:
            time_ch = 0
        pts_block_mlps = [nn.Linear(input_ch+condition_ch+time_ch, mlp_width), nn.ReLU()]
        layers_to_cat_input = []
        for i in range(mlp_depth+mlp_depth_plus-1):
            if i in skips:
                layers_to_cat_input.append(len(pts_block_mlps))
                pts_block_mlps += [nn.Linear(mlp_width + input_ch, mlp_width), 
                                   nn.ReLU()]
            else:
                if i>=mlp_depth-2:
                    if i==mlp_depth-2:
                        pts_block_mlps += [nn.Linear(mlp_width, mlp_width*last_linear_scale), nn.ReLU()]
                    else:
                        pts_block_mlps += [nn.Linear(mlp_width*last_linear_scale, mlp_width*last_linear_scale), nn.ReLU()]
                else:
                    pts_block_mlps += [nn.Linear(mlp_width, mlp_width), nn.ReLU()]
        self.layers_to_cat_input = layers_to_cat_input

        self.pts_linears = nn.ModuleList(pts_block_mlps)
        initseq(self.pts_linears)

        # output: rgb + sigma (density)
        if self.view_dir or self.pose_color == 'direct':
            assert self.multihead_enable==False, 'Unsupported multihead+view-dependent rgb'
            self.output_linear_density = nn.Sequential(nn.Linear(mlp_width, 1))
            self.output_linear_rgb_1 = nn.Sequential(nn.Linear(mlp_width, mlp_width))
            dim = mlp_width
            if self.view_dir:
                dim += self.input_ch_dir
            if self.pose_color=='direct':
                dim += self.pose_ch
            self.output_linear_rgb_2 = nn.Sequential(
                nn.Linear(dim, mlp_width),
                nn.Linear(mlp_width, 3))
        else:
            if self.multihead_enable==False:
                self.output_linear = nn.Sequential(nn.Linear(mlp_width*last_linear_scale, 4))
                if int(os.environ.get('TUNE_C',0))==1:
                    del self.output_linear
                    self.output_linear = None
                    self.output_linear_density = nn.Sequential(nn.Linear(mlp_width*last_linear_scale, 1))
                    self.output_linear_rgb = nn.Sequential(nn.Linear(mlp_width*last_linear_scale, 3))

            else:
                '''
                output_list = []
                for _ in range(multihead_depth-1):
                    output_list += [nn.Linear(mlp_width, mlp_width), nn.ReLU()]
                output_list += [nn.Linear(mlp_width, 4*self.multihead_num)]
                self.output_linear = nn.Sequential(*output_list)
                '''
                if multihead_depth==1:
                    self.output_linear = nn.Sequential(nn.Linear(mlp_width, 4*self.multihead_num))
                else:
                    self.multihead_mlp = MultiheadMlp(multihead_depth, multihead_num, mlp_width, 4)
                    self.output_linear = None
            if self.output_linear is not None:
                initseq(self.output_linear)
            else:
                pass #multihead_mlp already initialized in MultiheadMlp

        if self.pose_color == 'ao':
            self.output_linear_ao_1 = nn.Sequential(nn.Linear(mlp_width, mlp_width))
            dim = mlp_width+self.pose_ch
            self.output_linear_ao_2 = nn.Sequential(
                nn.Linear(dim, mlp_width),
                nn.Linear(mlp_width, 1)) #output a scalar
            self.ao_activation = torch.nn.Sigmoid()

    def forward(self, pos_embed, dir_embed=None, condition_code=None, head_id=None, pose_latent=None, time_vec_cnl=None, weights=None, **_):
        h = [pos_embed]
        if self.condition_code_dim>0:
            assert condition_code is not None
            condition_code_ = self.condition_code_encoder(condition_code)
            condition_code_ = condition_code_.expand((pos_embed.shape[0],)+condition_code_.shape[1:])
            if cfg.condition_code.type == 'local':
                assert cfg.canonical_mlp.condition_code_encoder.lower() == 'none'
                condition_code_ = localize_condition_code(condition_code_, weights)
            h += [condition_code_]
        if self.time_input:
            h += [time_vec_cnl.expand((pos_embed.shape[0],)+time_vec_cnl.shape[1:])]
        h = torch.cat(h, dim=1)
        for i, _ in enumerate(self.pts_linears):
            if i in self.layers_to_cat_input:
                h = torch.cat([pos_embed, h], dim=-1)
            h = self.pts_linears[i](h)

        if self.view_dir or self.pose_color=='direct':
            assert not self.multihead_enable
            density = self.output_linear_density(h)
            features = [self.output_linear_rgb_1(h)] #N,D
            if self.view_dir:
                features.append(dir_embed)
            if self.pose_color=='direct':
                features.append(pose_latent)

            rgb = self.output_linear_rgb_2(torch.cat(features,dim=1))
            outputs = torch.cat([rgb, density],dim=1) #N, 4
        else:
            if self.multihead_enable:
                head_id = head_id[0,0]
                if head_id == -1:
                    if self.multihead_depth==1:
                        outputs = self.output_linear(h)
                        outputs = [outputs[...,4*hid:4*(hid+1)] for hid in range(self.multihead_num)] #outputs[...,4*head_id:4*(head_id+1)]
                    else:
                        outputs = [self.multihead_mlp(h, hid) for hid in range(self.multihead_num)]              
                else:
                    if self.multihead_depth==1:
                        outputs = self.output_linear(h)
                        outputs = outputs[...,4*head_id:4*(head_id+1)]
                    else:
                        outputs = self.multihead_mlp(h, head_id)
            else:
                if int(os.environ.get('TUNE_C',0))==1:
                    outputs = torch.cat([self.output_linear_rgb(h), self.output_linear_density(h)], axis=1)
                else:
                    outputs = self.output_linear(h)
        
        if self.pose_color == 'ao':
            feature = self.output_linear_ao_1(h)
            ao = self.output_linear_ao_2(torch.cat([feature, pose_latent], axis=1))
            ao = self.ao_activation(ao)
            outputs = torch.cat([rgb*ao, density],dim=1) #N, 4
        return outputs    
        