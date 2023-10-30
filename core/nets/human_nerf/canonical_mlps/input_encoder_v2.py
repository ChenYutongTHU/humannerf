import torch, os
import torch.nn as nn

from core.utils.network_util import initseq
from core.nets.human_nerf.multihead import MultiheadMlp
from core.nets.human_nerf.selfattention import SelfAttention, MlpSeq
from configs import cfg 
import numpy as np
if cfg.canonical_mlp.input_encoder.spatial_localize_cfg.part2joints_file != 'empty':
    PART2JOINTS = np.load(cfg.canonical_mlp.input_encoder.spatial_localize_cfg.part2joints_file)
    PART2JOINTS = torch.tensor(PART2JOINTS).cuda()
N_JOINT = 23


def index_spatial_code(spatial_code, weights, fg_threshold):
    ws = weights.detach() #P, 24.
    ws = torch.argmax(ws, dim=1) #P    
    indices = torch.full_like(ws,-1,dtype=torch.long)
    fg_mask = weights.max(axis=1)[0]>fg_threshold
    indices[fg_mask] = ws[fg_mask]
    indexed_spatial_code = spatial_code[indices,:] #24,D
    return indexed_spatial_code


def spatial_localize_func(condition_code, enable, threshold, fg_threshold=0.2, **kwargs):
    if enable==False:
        return condition_code
    else:
        if threshold == 1:
            condition_code = condition_code[None,...] #1,batch_size(T),23,3
            mask = PART2JOINTS[:,None,:,None] #24,23 -> 24,1,23,1
            condition_code = mask*condition_code #[24,batch_size,23,3]
            condition_code = torch.cat([condition_code, torch.zeros_like(condition_code[0:1,...])], dim=0) #25,batch_size,23,3
        else:
            raise ValueError
            # #pos_xyz P,3; weights P,24; condition_code P,D
            # ws= weights[:,1:].detach() #remove root P,23 
            # dim_per_bone = condition_code.shape[-1]
            # if threshold == -1:
            #     pass
            # else:
            #     ws = torch.where(ws>threshold,1,0)
            # ws = torch.tile(ws[...,None], [1,1,dim_per_bone]) #P,23,dim_per_bone
            # condition_code = ws*condition_code  # [], [B,23,D]
    return condition_code 

class InputEncoder_v2(nn.Module):
    def __init__(self, input_ch, condition_code_dim, seq_len,  #dim per bone
                temporal_enc_method, temporal_enc_cfg_selfattention, temporal_enc_cfg_mlp,
                spatial_localize_cfg,
                spatial_enc_method, spatial_enc_cfg_mlp, 
                fuse_method, fuse_enc_cfg_tmlp,
                **kwargs):
        super(InputEncoder_v2, self).__init__()
        self.seq_len = seq_len
        if temporal_enc_method == 'selfattention':
            self.temporal_encoder = SelfAttention(
                    input_dim=condition_code_dim, #dim per bone
                    max_length=seq_len,
                    **temporal_enc_cfg_selfattention, pe_order='before_fc', in_proj='fc-relu')
            self.temporal_encode = lambda cc: self.temporal_encoder(cc) 
            temporal_dim = self.temporal_encoder.output_dim
        elif temporal_enc_method == 'mlp': #Feed T*D to mlp
            self.temporal_encoder = MlpSeq(
                input_dim=condition_code_dim, seq_len=seq_len, **temporal_enc_cfg_mlp)
            self.temporal_encode = lambda cc: self.temporal_encoder(cc.reshape((cc.shape[0],-1))) #[batchsize,T,Dp] -> [batchsize, T*Dp]
            temporal_dim = self.temporal_encoder.output_dim
        elif temporal_enc_method == 'empty':
            self.temporal_encode = lambda cc: cc.reshape(cc.shape[:1]+(-1,)) #B*23,T,D -> B,23,T*D
            temporal_dim = seq_len*condition_code_dim
        elif temporal_enc_method == 'BT-23-D': #B*23,T,Dp
            #self.temporal_encode = lambda cc: cc.reshape((-1,23,)+cc.shape[1:]).reshape((-1,23,cc.shape[-1])) #BUG
            self.temporal_encode = lambda cc: cc.reshape((-1,23,)+cc.shape[1:]).transpose(1,2).reshape((-1,23,cc.shape[-1]))
            temporal_dim = condition_code_dim
        self.spatial_localize_cfg = spatial_localize_cfg
        self.spatial_localize = lambda tc: spatial_localize_func(tc,**self.spatial_localize_cfg)

        if spatial_enc_method == 'selfattention':
            raise ValueError
        elif spatial_enc_method == 'mlp':
            self.spatial_encoder = MlpSeq(
                input_dim=temporal_dim, seq_len=N_JOINT, **spatial_enc_cfg_mlp
            )
            self.spatial_encode = lambda tc: self.spatial_encoder(tc) #B,23,D 
            spatial_dim = self.spatial_encoder.output_dim
        elif spatial_enc_method == 'empty':
            self.spatial_encode = lambda tc: tc.reshape((tc.shape[0],-1)) #B,23,T*D -> B,23*T*D
            spatial_dim = temporal_dim*N_JOINT
        else:
            raise ValueError

        self.fuse_method = fuse_method
        if fuse_method == 'concat':
            self.output_dim = spatial_dim + input_ch
        elif fuse_method in ['tmlp','tmlp_debug']:
            self.fuse_encoder = MlpSeq(input_dim=spatial_dim, seq_len=seq_len, **fuse_enc_cfg_tmlp)
            if fuse_method == 'tmlp':
                self.output_dim = self.fuse_encoder.output_dim
            else:
                self.output_dim = self.fuse_encoder.output_dim + input_ch
        return

    def forward(self, pos_embed, condition_code, weights=None, gate_weight=1):
        #condition_code, [1,T,23,Dp] 
        condition_code = torch.transpose(condition_code, 1,2) #1,23,T,Dp
        condition_code_ = torch.reshape(condition_code, (-1,)+condition_code.shape[2:]) #Batchsize, T, Dp
        
        temporal_code = self.temporal_encode(condition_code_) #1*23(batch-size), D or B-T,23,D
        temporal_code = temporal_code.reshape((-1, N_JOINT, temporal_code.shape[-1])) #B,23, D(if empty: T*condition_code_dim)
        localized_temporal_code = self.spatial_localize(temporal_code) #24,B,23,D
        spatial_code = self.spatial_encode(localized_temporal_code) #B,23,D -> B,D // (25,B,23,D)-> (25,B,D1)
        if self.fuse_method == 'concat':
            spatial_code = spatial_code.expand((pos_embed.shape[0],-1))
            output = torch.cat([pos_embed, spatial_code*gate_weight], axis=-1) #B,D
        elif self.fuse_method == 'tmlp_debug':
            #spatial_code T first
            #spatial_code = spatial_code.reshape((-1,self.seq_len,spatial_code.shape[-1])) #B,T,D BUG
            #spatial_code = spatial_code.reshape((self.seq_len,-1,spatial_code.shape[-1])).transpose(0,1) #B,T,D DEBUG
            spatial_code = self.fuse_encoder(spatial_code) #25,B(T),D1 -> 25,D2
            spatial_code = index_spatial_code(spatial_code, weights, fg_threshold=self.spatial_localize_cfg.fg_threshold)
            output = torch.cat([pos_embed, spatial_code*gate_weight], axis=-1)
        elif self.fuse_method == 'tmlp':
            spatial_code = spatial_code.reshape((-1,self.seq_len,spatial_code.shape[-1])) #B,T,D
            output = self.fuse_encoder(spatial_code) #B,D2            
        return output