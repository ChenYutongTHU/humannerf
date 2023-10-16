import torch
import torch.nn as nn
from configs import cfg

class NonRigidMotionTransformerEncoder(nn.Module):
    def __init__(self, 
        query_input_dim, **kwargs):
        super(NonRigidMotionTransformerEncoder, self).__init__()
        self.d_model = cfg.non_rigid_motion_transformer_encoder.d_model
        self.query_proj = nn.Linear(query_input_dim, self.d_model)
        self.condition_proj = nn.Linear(cfg.non_rigid_motion_transformer_encoder.condition_input_dim, self.d_model)
        self.time_embedding_type = cfg.non_rigid_motion_transformer_encoder.time_embedding_type
        if self.time_embedding_type == 'learnable':
            #zero for query
            self.time_embedding = nn.Embedding(cfg.non_rigid_motion_transformer_encoder.time_embedding_max_length+1,self.d_model)
        elif self.time_embedding_type == 'sine':
            time_embedding_freq = cfg.non_rigid_motion_transformer_encoder.d_model//2
            assert cfg.non_rigid_motion_transformer_encoder.d_model%2==0
            embedder = Embedder(include_input=False, input_dims=1, 
            max_freq_log2=time_embedding_freq-1, num_freqs=time_embedding_freq,
            periodic_fns=[torch.sin, torch.cos])
            self.time_embedding = lambda x: embedder.embed(x)
        else:
            self.time_embedding = None
        self.joint_number = cfg.non_rigid_motion_transformer_encoder.joint_embedding_max_length
        self.joint_embedding_type = cfg.non_rigid_motion_transformer_encoder.joint_embedding_type
        if self.joint_embedding_type == 'learnable':
            self.joint_embedding = nn.Embedding(self.joint_number,self.d_model)
        else:
            self.joint_embedding = None    
        encoder_layer = nn.TransformerEncoderLayer(
            self.d_model, 
            cfg.non_rigid_motion_transformer_encoder.nhead, 
            cfg.non_rigid_motion_transformer_encoder.dim_feedforward, 
            batch_first=True)
        encoder_norm = nn.LayerNorm(self.d_model, eps=1e-5)
        self.encoder = nn.TransformerEncoder(
            encoder_layer, 
            cfg.non_rigid_motion_transformer_encoder.num_encoder_layers, encoder_norm)
        self.output_mlp = nn.Linear(
            cfg.non_rigid_motion_transformer.d_model, 3)

        # init the weights of the last layer as very small value
        # -- at the beginning, we hope non-rigid offsets are zeros 
        init_val = 1e-5
        self.output_mlp.weight.data.uniform_(-init_val, init_val)
        self.output_mlp.bias.data.zero_()
        return  

    def forward(self, pos_embed, pos_xyz, condition_code, time_ids=None, joint_ids=None, **_):
        query_input = self.query_proj(pos_embed)[:,None,:]#N,1,D
        condition_input = self.condition_proj(condition_code) #B,N,D2
        total_len = condition_code.shape[1]
        num_frames = total_len//self.joint_number
        assert total_len%self.joint_number==0, total_len
        if self.time_embedding is not None:
            if time_ids is None:
                time_ids = torch.arange(num_frames, dtype=torch.long, device=condition_code.device) #N
                time_ids = torch.tile(time_ids[None,...],[self.joint_number,1]).T.reshape(-1)[None,...]
            condition_input += self.time_embedding(time_ids) #B,N -> B,N,D
        if self.joint_embedding is not None:
            if joint_ids is None:
                joint_ids = torch.arange(self.joint_number, dtype=torch.long, device=condition_code.device) #N
                joint_ids = torch.cat([joint_ids for _ in range(num_frames)], dim=0)[None,...] #1,N
            condition_input += self.joint_embedding(joint_ids) #B,N,D
        encoder_input = torch.cat([query_input, condition_input.expand((query_input.shape[0],)+condition_input.shape[1:])], dim=1) #B,N_,D
        output = self.encoder(src=encoder_input)[:,0,:] #B,N,E -> B,E
        trans = self.output_mlp(output) #B,3        
        result = {
            'xyz': pos_xyz + trans,
            'offsets': trans
        }                 
        return result        
'''
class NonRigidMotionTransformer(nn.Module):
    def __init__(self, 
        encoder_input_dim, decoder_input_dim,
        time_embedding_type='learnable', joint_embedding_type='learnable',
        time_embedding_max_length=20, joint_embedding_max_length=23, **kwargs):
        super(NonRigidMotionTransformer, self).__init__()
        self.encoder_proj = nn.Linear(encoder_input_dim, cfg.non_rigid_motion_transformer.d_model)
        self.decoder_proj = nn.Linear(decoder_input_dim, cfg.non_rigid_motion_transformer.d_model)
        self.time_embedding_type = time_embedding_type
        if time_embedding_type == 'learnable':
            self.time_embedding = nn.Embedding(cfg.non_rigid_motion_transformer.time_embedding_max_length,decoder_input_dim)
        elif time_embedding_type == 'sine':
            time_embedding_freq = cfg.non_rigid_motion_transformer.d_model//2
            assert cfg.non_rigid_motion_transformer.d_model%2==0
            embedder = Embedder(include_input=False, input_dims=1, 
            max_freq_log2=time_embedding_freq-1, num_freqs=time_embedding_freq,
            periodic_fns=[torch.sin, torch.cos])
            self.time_embedding = lambda x: embedder.embed(x)
        else:
            self.time_embedding = None
        if joint_embedding_type == 'learnable':
            self.joint_embedding = nn.Embedding(cfg.non_rigid_motion_transformer.joint_embedding_max_length,decoder_input_dim)
        else:
            self.joint_embedding = None
        self.transformer = nn.Transformer(
            d_model=cfg.non_rigid_motion_transformer.d_model, 
            nhead=cfg.non_rigid_motion_transformer.nhead, 
            num_encoder_layers=cfg.non_rigid_motion_transformer.num_encoder_layers, 
            num_decoder_layers=cfg.non_rigid_motion_transformer.num_decoder_layers,
            dim_feedforward=cfg.non_rigid_motion_transformer.dim_feedforward,
            batch_first=True)
        self.output_mlp = nn.Linear(
            cfg.non_rigid_motion_transformer.d_model, 3)
        return

    def forward(self, pos_embed, pos_xyz, condition_code, time_ids, joint_ids, viewdirs=None, head_id=None ,**_):
        decoder_input = self.decoder_proj(pos_embed)
        encoder_input_data = self.encoder_proj(condition_code) #B,N,D2
        encoder_input_time = self.time_embedding(time_ids) #B,N -> B,N,D
        encoder_input_joint = self.joint_embedding(joint_ids) #B,N,D
        encoder_input = encoder_input_data+encoder_input_time+encoder_input_joint
        output = self.transformer(src=encoder_input, tgt=decoder_input,)[:,0,:] #B,N,E
        trans = self.output_mlp(output) #B,3        
        result = {
            'xyz': pos_xyz + trans,
            'offsets': trans
        }                   
        return result
'''