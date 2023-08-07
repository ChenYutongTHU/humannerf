import torch.nn as nn
from core.utils.network_util import initseq

class MultiheadMlp(nn.Module):
    def __init__(self, multihead_depth, multihead_num, mlp_width, output_channel):
        super(MultiheadMlp, self).__init__()
        self.multihead_num = multihead_num
        self.head = []
        for i in range(multihead_num):
            mlp_per_head = []
            for _ in  range(multihead_depth-1):
                mlp_per_head += [nn.Linear(mlp_width, mlp_width), nn.ReLU()]
            mlp_per_head += [nn.Linear(mlp_width, output_channel)]
            mlp_per_head = nn.Sequential(*mlp_per_head)
            initseq(mlp_per_head)
            self.head.append(mlp_per_head)
        self.head = nn.ModuleList(self.head)
    def forward(self, x, head_id):
        return self.head[head_id](x)