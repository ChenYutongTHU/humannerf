import argparse
import torch
from torchvision import models, transforms
import torch.nn as nn
from configs import cfg

transform = transforms.Compose([
    # transforms.Resize((512, 512)),  # Resize images to an appropriate size
    # transforms.ToTensor(),  # Convert images to Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

net_layer2dim = {
    'resnet34':{-1:3, 2:64,4:64,5:128,6:256,7:512},
}

class RGB_FeatureComputer(nn.Module):
    def __init__(self, 
            net='resnet34', layer=6, precompute=False):
        super(RGB_FeatureComputer, self).__init__()    
        self.net = net
        self.layer = layer
        self.precompute = (precompute or layer==-1) #-1 -> raw
        if precompute:
            self.model = None
        else:
            model = getattr(models, net)(pretrained=True)
            model.eval()
            self.model = torch.nn.Sequential(*(list(model.children())[:layer+1]))
            for param in self.model.parameters():
                param.requires_grad = False
        self.output_dim = net_layer2dim[net][self.layer]

    def compute_and_index_features(self, imgs, indices_hw): #(B,H,W,C), (*,B,2)
        imgs = imgs.permute((0,3,1,2)) #B,C,H,W
        h0, w0 = cfg.height, cfg.width #imgs.shape[-2], imgs.shape[-1]
        assert h0==w0, 'Only support square images'
        orig_shape, orig_dim = indices_hw.shape,indices_hw.dim() 
        tv_num = orig_shape[-2] #N_fg, T*V, 2
        indices_hw = indices_hw.reshape((-1,)+indices_hw.shape[-2:]) #-1,tv_num,2
        p_num = indices_hw.shape[0]
        if self.precompute:
            x = imgs
        else:
            x = transform(imgs)
            with torch.no_grad():
                x = self.model(x)


        # x = torch.load('data/zju/CoreView_387/rgb_features/resnet34/layer-6/Camera_B1/000001.bin',map_location='cuda')
        # x = x.permute((2,0,1))[None,...] #1,C,H,W
        # x = torch.tile(x,[x0.shape[0],1,1,1])
        # # x = x.expand(x0.shape[0],-1,-1,-1)
        hi, wi = x.shape[-2], x.shape[-1] 
        scale_h, scale_w = int(h0/hi), int(w0/wi) #2,4,8,16
        indices_i = (torch.floor(indices_hw[...,0]/scale_h)*wi+torch.floor(indices_hw[...,1]/scale_w)).long() #*,tv_num
        x_flatten = x.reshape(x.shape[:2]+(-1,)) #TV,D,h*w
        x_flatten = x_flatten.permute((2,0,1)).reshape(-1,self.output_dim) #(h*w,tv,D) -> (-1,D)
        # indices_i = indices_i[...,None,None].expand(-1,-1,x_flatten.shape[1],-1) #-1,B,D,1
        # x_flatten = x_flatten[None,...] #,1,B,D,h*w
        # x_indexed = torch.gather(input=x_flatten, index=indices_i, dim=-1) #-1,B,D,1
        # pool_size = self.window_size/scale_h
        mask = torch.zeros([hi*wi, tv_num], device=indices_i.device, dtype=torch.bool)
        mask.scatter_(dim=0, index=indices_i, value=True) #h0*w0,tv_num
        mask_flatten = mask.view(-1) #1024*tv_num
        mask_cumsum_flatten = torch.cumsum(mask_flatten, dim=0)-1 #1024*tv_num
        mask_cumsum = mask_cumsum_flatten.view(mask.shape)
        id_in_list = torch.gather(input=mask_cumsum, index=indices_i, dim=0)
        features_list = x_flatten[mask_flatten]
        # import ipdb; ipdb.set_trace()
        # features_tv = []
        # for tvi in range(tv_num):
        #     features_tv.append(x_flatten[tvi,:,mask[:,tvi]])
        # break
        #features.append(x_indexed[...,0]) #(-1,B,D)
        # x_indexed = torch.gather(input=x_flatten[None,...].expand(indices_i.shape[0],-1,-1,-1), 
        #             index=indices_i[...,None,None].expand(-1,-1,x_flatten.shape[1],-1),
        #             dim=-1) #(1, TV, D, h*w) (N_fg,tv_num,) -> (N_fg, tv,D)
        # features.append(x_indexed)
        #concatenate different layers
        # features_tv = [torch.cat(f, dim=0).T for f in features_tv] #list of D,N -> list of N,D'
        # features = torch.cat(features, dim=-1) #(N_fg, tv,D)
        return features_list, id_in_list
        

        
                
        

