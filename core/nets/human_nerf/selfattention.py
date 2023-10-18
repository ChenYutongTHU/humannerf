import torch, os
import torch.nn as nn
import math
from core.utils.network_util import initseq
from core.nets.human_nerf.multihead import MultiheadMlp
from core.nets.human_nerf.embedders.fourier import Embedder
from configs import cfg
class MlpSeq(nn.Module):
    def __init__(self, input_dim, seq_len, hidden_dim, output_dim, non_linear, depth=1, ):
        super(MlpSeq, self).__init__()
        module_list = []
        for i in range(depth):
            if i==0:
                module_list.append(nn.Linear(input_dim*seq_len, hidden_dim))
            else:
                module_list.append(nn.Linear(hidden_dim, hidden_dim))
            if non_linear:
                module_list.append(nn.ReLU())
        module_list.append(nn.Linear(hidden_dim, output_dim))
        self.mlp = nn.Sequential(*module_list)

    def forward(self, input_seq):
        #B,T,D
        input_seq = input_seq.reshape([input_seq.shape[0],-1])
        return self.mlp(input_seq)

class SelfAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,
            positional_encoding_type, max_length, pe_order, pe_dim=None, in_proj='fc-relu'):
        super(SelfAttention, self).__init__()
        self.out_proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),nn.ReLU(),nn.Linear(hidden_dim, output_dim))
        self.positional_encoding_type = positional_encoding_type
        self.pe_order = pe_order #after_fc or before_fc
        if self.pe_order == 'after_fc':
            pe_dim = hidden_dim
        elif self.pe_order == 'before_fc':
            pe_dim = pe_dim
            input_dim = input_dim+pe_dim
        if in_proj == 'fc-relu':
            self.in_proj = nn.Sequential(nn.Linear(input_dim, hidden_dim),nn.ReLU())
        elif in_proj == 'fc-relu-fc':
            self.in_proj = nn.Sequential(nn.Linear(input_dim, hidden_dim),nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        if self.positional_encoding_type == 'learnable':
            self.positional_encoding = nn.Embedding(max_length,pe_dim)
        elif self.positional_encoding_type == 'sine':
            embedder = Embedder(include_input=False, input_dims=1, 
            d_model=pe_dim, periodic_fns=[torch.sin, torch.cos], freq_type='transformer')  
            self.positional_encoding = lambda x: embedder.embed(x[:,None])   #L,1          
        elif self.positional_encoding_type == 'sine_fourier':
            embedder2 = Embedder(include_input=False, input_dims=1, 
            max_freq_log2=pe_dim//2-1, num_freqs=pe_dim//2, periodic_fns=[torch.sin, torch.cos], freq_type='fourier')  
            self.positional_encoding = lambda x: embedder2.embed(x[:,None]/(cfg.canonical_mlp.selfattention.max_length-1))   #L,1             
        elif self.positional_encoding_type == 'empty':
            self.positional_encoding = None
        else:
            raise ValueError
        if int(os.environ.get('ATTENTION_OFF',0))==1:
            self.attention = None
        else:
            self.attention = nn.MultiheadAttention(hidden_dim, num_heads=1, dropout=0.2, batch_first=True)

    def forward(self, input_seq): #B,T,D
        if self.positional_encoding_type != 'empty':
            ids = torch.arange(input_seq.shape[1], dtype=torch.long, device=input_seq.device)
            pe = self.positional_encoding(ids)[None,...]
        else:
            pe = 0
        if self.pe_order == 'before_fc' and self.positional_encoding_type != 'empty':
            input_seq = torch.cat([input_seq, pe], axis=-1)
            sa_input = self.in_proj(input_seq)
        else:
            x = self.in_proj(input_seq)
            sa_input = x + pe
        if self.attention is not None:
            attn_output, _ = self.attention(query=sa_input, key=sa_input, value=sa_input)
            attn_output = attn_output[:,0,:] #B,N,D
        else:
            attn_output = torch.mean(sa_input, axis=1)
        output = self.out_proj(attn_output)
        return output