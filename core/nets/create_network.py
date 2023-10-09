import os, torch
import imp

from configs import cfg

def _query_network():
    module = cfg.network_module
    module_path = module.replace(".", "/") + ".py"
    network = imp.load_source(module, module_path).Network
    return network


def create_network():
    network = _query_network()
    network = network()
    if cfg.modules.pretrained_path != 'empty':
        assert os.path.isfile(cfg.modules.pretrained_path), cfg.modules.pretrained_path
        state_dict = torch.load(cfg.modules.pretrained_path, map_location='cpu')['network']
        if cfg.modules.canonical_mlp.reinit:
            print('Reinitialize canonical_mlp')
            state_dict = {k:v for k,v in state_dict.items() if not 'cnl_mlp' in k}
        if cfg.modules.non_rigid_motion_mlp.reinit:
            print('Reinitialize rigid_motion_mlp')
            state_dict = {k:v for k,v in state_dict.items() if not 'non_rigid_mlp' in k}
        msg = network.load_state_dict(state_dict, strict=False)
        print(msg)
        for name, param in network.named_parameters():
            param.requires_grad = False
            if cfg.modules.canonical_mlp.tune and 'cnl_mlp' in name:
                param.requires_grad = True
            if cfg.modules.non_rigid_motion_mlp.tune and 'non_rigid_mlp' in name:
                param.requires_grad = True
            if cfg.modules.canonical_mlp.tune and 'cnl_mlp' in name:
                param.requires_grad = True            
            if cfg.modules.pose_decoder.tune and 'pose_decoder' in name:
                param.requires_grad = True 
            if cfg.modules.mweight_vol_decoder.tune and 'mweight_vol_decoder' in name:
                param.requires_grad = True 
    return network
