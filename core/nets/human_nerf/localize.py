from configs import cfg 
import torch, numpy as np
if cfg.condition_code.local.part2joints_file != 'empty':
    PART2JOINTS = np.load(cfg.condition_code.local.part2joints_file)
    PART2JOINTS = torch.tensor(PART2JOINTS)


def mask_condition_code(condition_code, mask):
    #condition_code (P,23*D); mask[P,23]
    dim_per_bone = condition_code.shape[1]//23
    mask = torch.tile(mask[...,None], [1,1,dim_per_bone]) #P,23,dim_per_bone
    mask = mask.reshape(mask.shape[0],-1) #P,23*dim_per_bone
    assert mask.shape[1]==condition_code.shape[1]
    condition_code = mask*condition_code  
    return condition_code

def localize_condition_code(condition_code, weights):
    # if condition_code.shape[0]!=weights.shape[0]:
    #     condition_code = condition_code.expand((weights.shape[0],-1,-1))
    if cfg.condition_code.type == 'global':
        pass
    elif cfg.condition_code.type == 'local':
        #pos_xyz P,3; weights P,24; condition_code P,D
        ws= weights[:,1:].detach() #remove root P,23
        if cfg.condition_code.local.threshold == -1:
            pass
        else:
            # import ipdb; ipdb.set_trace()
            ws = torch.where(ws>cfg.condition_code.local.threshold,1,0)
        condition_code = mask_condition_code(condition_code, ws)  
    elif cfg.condition_code.type == 'local_manual':
        ws = weights.detach() #P, 24. The weights indicate which part the points belong to
        ws = torch.argmax(ws, dim=1) #P
        mask = (PART2JOINTS.cuda())[ws] #24,23 -> P,23
        mask = mask*(weights.max(axis=1,keepdims=True)[0]>cfg.condition_code.local.fg_threshold)
        condition_code = mask_condition_code(condition_code, mask)
    else:
        raise ValueError
    return condition_code 