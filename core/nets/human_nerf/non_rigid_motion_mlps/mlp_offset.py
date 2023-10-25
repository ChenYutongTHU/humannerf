import torch
import torch.nn as nn

from core.utils.network_util import initseq
from core.nets.human_nerf.multihead import MultiheadMlp
from core.nets.human_nerf.localize import localize_condition_code
from configs import cfg
class NonRigidMotionMLP(nn.Module):
    def __init__(self,
                 pos_embed_size=3, 
                 condition_code_size=69,
                 mlp_width=128,
                 mlp_depth=6,
                 mlp_depth_plus=0,
                 skips=None, multihead_enable=False, multihead_depth=1, multihead_num=4, 
                 last_linear_scale=1,**kwargs):
        super(NonRigidMotionMLP, self).__init__()

        self.skips = [4] if skips is None else skips
        self.multihead_enable, self.multihead_num, self.multihead_depth = multihead_enable, multihead_num, multihead_depth
        
        block_mlps = [nn.Linear(pos_embed_size+condition_code_size, 
                                mlp_width), nn.ReLU()]
        
        layers_to_cat_inputs = []
        for i in range(1, mlp_depth+mlp_depth_plus):
            if i in self.skips:
                layers_to_cat_inputs.append(len(block_mlps))
                block_mlps += [nn.Linear(mlp_width+pos_embed_size, mlp_width), 
                               nn.ReLU()]
            else:
                if i>=mlp_depth-1:
                    if i==mlp_depth-1:
                        block_mlps += [nn.Linear(mlp_width, mlp_width*last_linear_scale), nn.ReLU()]
                    else:
                        block_mlps += [nn.Linear(mlp_width*last_linear_scale, mlp_width*last_linear_scale), nn.ReLU()]
                else:
                    block_mlps += [nn.Linear(mlp_width, mlp_width), nn.ReLU()]

        if multihead_enable==False:
            block_mlps += [nn.Linear(mlp_width*last_linear_scale, 3)] 
            #last_linear_scale to increase single-head mlp for fair comparison
        else:
            '''
            for _ in  range(multihead_depth-1):
                block_mlps += [nn.Linear(mlp_width, mlp_width), nn.ReLU()]
            block_mlps += [nn.Linear(mlp_width, 3*multihead_num)]
            ''' #BUG
            if multihead_depth==1:
                block_mlps += [nn.Linear(mlp_width, 3*multihead_num)]
            else:
                self.multihead_mlp = MultiheadMlp(
                    multihead_depth, multihead_num, mlp_width, output_channel=3)

        self.block_mlps = nn.ModuleList(block_mlps)
        initseq(self.block_mlps)

        self.layers_to_cat_inputs = layers_to_cat_inputs

        # init the weights of the last layer as very small value
        # -- at the beginning, we hope non-rigid offsets are zeros 
        init_val = 1e-5
        if self.multihead_enable==False or self.multihead_depth==1:
            last_layer = self.block_mlps[-1] 
            last_layer.weight.data.uniform_(-init_val, init_val)
            last_layer.bias.data.zero_()
        else:
            for i in range(multihead_num):
                last_layer = self.multihead_mlp.head[i][-1]
                last_layer.weight.data.uniform_(-init_val, init_val)
                last_layer.bias.data.zero_()                


    def forward(self, pos_embed, pos_xyz, condition_code, viewdirs=None, head_id=None, weights=None, **_):
        condition_code = condition_code.expand((pos_embed.shape[0],-1)) #P,D
        condition_code = localize_condition_code(condition_code, weights)
        h = torch.cat([condition_code, pos_embed], dim=-1)
        if viewdirs is not None:
            h = torch.cat([h, viewdirs], dim=-1)
        for i in range(len(self.block_mlps)):
            if i in self.layers_to_cat_inputs:
                h = torch.cat([h, pos_embed], dim=-1)
            h = self.block_mlps[i](h)
        trans = h
        multi_outputs = False
        if self.multihead_enable:
            head_id = head_id[0,0]
            if head_id==-1: #all head
                #replicate pos_xyz  first
                if self.multihead_depth==1: #legacy #B, num_head*3
                    trans = torch.split(trans, 3, dim=1)
                else:
                    trans_ = []
                    for head_id in range(self.multihead_num):
                        trans_.append(self.multihead_mlp(trans, head_id=head_id))    #B, 3
                    trans = trans_ #[B,3]     
                multi_outputs = True
            else:
                if self.multihead_depth==1: #legacy
                    trans = trans[:,head_id*3:(head_id+1)*3]
                else:
                    trans = self.multihead_mlp(trans, head_id=head_id)
        
        if multi_outputs==True:
            result = {
                'xyz': [pos_xyz + t for t in trans],
                'offsets': trans
            }          
        else:
            result = {
                'xyz': pos_xyz + trans,
                'offsets': trans
            }                   
        return result
