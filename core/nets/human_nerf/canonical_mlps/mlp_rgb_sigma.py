import torch
import torch.nn as nn

from core.utils.network_util import initseq


class CanonicalMLP(nn.Module):
    def __init__(self, mlp_depth=8, mlp_width=256, 
                 input_ch=3, skips=None, view_dir=False, input_ch_dir=3,
                 **_):
        super(CanonicalMLP, self).__init__()

        if skips is None:
            skips = [4]

        self.mlp_depth = mlp_depth
        self.mlp_width = mlp_width
        self.input_ch = input_ch
        self.view_dir = view_dir
        self.input_ch_dir = input_ch_dir
        
        pts_block_mlps = [nn.Linear(input_ch, mlp_width), nn.ReLU()]
        layers_to_cat_input = []
        for i in range(mlp_depth-1):
            if i in skips:
                layers_to_cat_input.append(len(pts_block_mlps))
                pts_block_mlps += [nn.Linear(mlp_width + input_ch, mlp_width), 
                                   nn.ReLU()]
            else:
                pts_block_mlps += [nn.Linear(mlp_width, mlp_width), nn.ReLU()]
        self.layers_to_cat_input = layers_to_cat_input

        self.pts_linears = nn.ModuleList(pts_block_mlps)
        initseq(self.pts_linears)

        # output: rgb + sigma (density)
        if self.view_dir:
            self.output_linear_density = nn.Sequential(nn.Linear(mlp_width, 1))
            self.output_linear_rgb_1 = nn.Sequential(nn.Linear(mlp_width, mlp_width))
            self.output_linear_rgb_2 = nn.Sequential(
                nn.Linear(mlp_width+self.input_ch_dir, mlp_width),
                nn.Linear(mlp_width, 3))
        else:
            self.output_linear = nn.Sequential(nn.Linear(mlp_width, 4))
            initseq(self.output_linear)


    def forward(self, pos_embed, dir_embed=None, **_):
        h = pos_embed
        for i, _ in enumerate(self.pts_linears):
            if i in self.layers_to_cat_input:
                h = torch.cat([pos_embed, h], dim=-1)
            h = self.pts_linears[i](h)

        if self.view_dir:
            density = self.output_linear_density(h)
            feature = self.output_linear_rgb_1(h) #N,D
            feature_dir = torch.cat([feature, dir_embed],dim=1)
            rgb = self.output_linear_rgb_2(feature_dir)
            outputs = torch.cat([rgb, density],dim=1) #N, 4
        else:
            outputs = self.output_linear(h)

        return outputs    
        