import torch
import torch.nn as nn
from configs import cfg

class NonRigidMotionTransformerEncoder(nn.Module):
    def __init__(self, 
        query_input_dim, **kwargs):
        super(NonRigidMotionTransformerEncoder, self).__init__()
        self.cfg1 = cfg.non_rigid_motion_TStransformer_encoder.encoder1
        self.cfg2 = cfg.non_rigid_motion_TStransformer_encoder.encoder2
        self.attention_order = cfg.non_rigid_motion_TStransformer_encoder.attention_order
        self.query_proj = nn.Linear(query_input_dim, self.cfg2.d_model) #to encoder2
        self.condition_proj = nn.Linear(cfg.non_rigid_motion_TStransformer_encoder.condition_input_dim, self.cfg1.d_model)
        self.condition_proj2 = nn.Linear(self.cfg1.d_model, self.cfg2.d_model)

        encoders_list, embeddings_list = [],[]
        for i, cfg_ in enumerate([self.cfg1, self.cfg2]):
            if cfg_.embedding_type == 'learnable':
                embeddings_list.append(nn.Embedding(cfg_.embedding_max_length+1, cfg_.d_model)) #one for [CLS]
            elif cfg_.embedding_type == 'sine':
                embedding_freq = cfg_.d_model//2
                assert cfg_.d_model%2==0
                embedder = Embedder(include_input=False, input_dims=1, 
                max_freq_log2=embedding_freq-1, num_freqs=embedding_freq,
                periodic_fns=[torch.sin, torch.cos])
                embeddings_list.append(lambda x: embedder.embed(x))
            else:
                raise ValueError
            encoder_layer = nn.TransformerEncoderLayer(cfg_.d_model, cfg_.nhead, cfg_.dim_feedforward, batch_first=True)
            encoder_norm = nn.LayerNorm(cfg_.d_model, eps=1e-5)
            encoders_list.append(nn.TransformerEncoder(
                encoder_layer, 
                cfg_.num_encoder_layers, encoder_norm))
        self.encoder1, self.encoder2 = encoders_list[0], encoders_list[1]
        self.embedding1, self.embedding2 = embeddings_list[0],embeddings_list[1]
        self.output_mlp = nn.Linear(
            cfg.non_rigid_motion_transformer.d_model, 3)

        # init the weights of the last layer as very small value
        # -- at the beginning, we hope non-rigid offsets are zeros 
        init_val = 1e-5
        self.output_mlp.weight.data.uniform_(-init_val, init_val)
        self.output_mlp.bias.data.zero_()
        return  

    def forward(self, pos_embed, pos_xyz, condition_code, **_):
        #condition code (T,num_joints,D)
        B, T, J, D = condition_code.shape
        if self.attention_order == 'TS': #temporal-first
            condition_code = torch.transpose(condition_code, 1, 2) #B,num_joints, T, D

        ids1 = torch.arange(condition_code.shape[2], dtype=torch.long, device=condition_code.device) #T+1, one for query
        embedding1 = self.embedding1(ids1)[None,...] #1,T,D
        output1 = []
        for i in range(condition_code.shape[1]):
            chunk = condition_code[:,i,:,:] #B,T,D
            chunk = self.condition_proj(chunk)
            o1 = self.encoder1(chunk+embedding1)[:,0,:] #B,D
            output1.append(o1)
        output1 = torch.stack(output1, dim=1) #B,J,D
        ids2 = torch.arange(output1.shape[1]+1, dtype=torch.long, device=condition_code.device) #one for query_code
        embedding2 = self.embedding2(ids2)[None,...] #1, J+1, D

        query_input = self.query_proj(pos_embed)[:,None,:] #B,1,D 
        output12 = self.condition_proj2(output1) #B,J,D
        input2 = torch.cat([query_input, output12], dim=1) #B,J+1,D
        input2 = input2 + embedding2
        output2 = self.encoder2(input2)[:,0,:] #B,D
        trans = self.output_mlp(output2) #1,3
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