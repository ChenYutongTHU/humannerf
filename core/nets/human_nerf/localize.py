from configs import cfg 
import torch



def localize_condition_code(condition_code, weights):
    if condition_code.shape[0]!=weights.shape[0]:
        condition_code = condition_code.expand((weights.shape[0],-1,-1))
    if cfg.condition_code.type == 'global':
        pass
    elif cfg.condition_code.type == 'local':
        #pos_xyz P,3; weights P,24; condition_code P,D
        ws= weights[:,1:].detach() #remove root P,23
        dim_per_bone = condition_code.shape[1]//ws.shape[1]
        if cfg.condition_code.local.threshold == -1:
            pass
        else:
            # import ipdb; ipdb.set_trace()
            ws = torch.where(ws>cfg.condition_code.local.threshold,1,0)
        ws = torch.tile(ws[...,None], [1,1,dim_per_bone]) #P,23,dim_per_bone
        ws = ws.reshape(ws.shape[0],-1) #P,23*dim_per_bone
        assert ws.shape[1]==condition_code.shape[1]
        condition_code = ws*condition_code  

    return condition_code 