import sys
sys.path.append('/mnt/workspace/chenyutong/code/humannerf')
from core.utils.image_util import load_image
import os, cv2
import argparse
from collections import defaultdict
import numpy as np
import torch
from tqdm import tqdm

class MetricsWriter(object):
    def __init__(self, output_dir, exp_name, dataset):
        os.makedirs(output_dir, exist_ok=True)
        self.per_img_f = open(os.path.join(output_dir, f'{exp_name}-metrics.perimg.txt'),'a')
        self.average_f = open(os.path.join(output_dir, f'{exp_name}-metrics.average.txt'),'a')
        self.per_img_f.writelines(f'========={dataset}==========\n')
        self.average_f.writelines(f'========={dataset}==========\n')
        
        self.name2metrics = {}
        self.metrics2ave = defaultdict(int)

        #self.lpips_computer = LpipsComputer()
        self.metrics_func = {
            "psnr": lambda pred, target, mask: compute_psnr(pred, target, mask).item(),
            #"lpips": lambda pred, target, mask: 1000*self.lpips_computer.compute_lpips(pred=pred.cuda(), target=target.cuda()).item(), 
            #"ssim": lambda pred, target, mask: compute_ssim(pred, target, mask).item()
        }

        self.N = 0

    def normalize(self, img):
        if type(img)==np.ndarray:
            img = torch.tensor(img, dtype=torch.float32)
        if torch.max(img)>2:
            img /= 255
        return img

    def append(self, name, pred, target, mask=None):
        self.N += 1
        assert name not in self.name2metrics, name
        pred, target = self.normalize(pred), self.normalize(target)
        self.per_img_f.writelines(f'{name}: ')
        self.name2metrics[name] = {}

        for k in ["psnr"]:
            self.name2metrics[name][k] = self.metrics_func[k](pred, target, mask)
            self.metrics2ave[k] += self.name2metrics[name][k]
            self.per_img_f.writelines('{}-{:.4f} '.format(k, self.name2metrics[name][k]))
        self.per_img_f.writelines('\n')   

    def finalize(self):
        self.metrics2ave = {k:v/self.N for k,v in self.metrics2ave.items()}      
        for k, v in self.metrics2ave.items():
            self.average_f.writelines(f'{k}:{v:.4f}\n') 

        self.per_img_f.close()
        self.average_f.close()

def compute_psnr_from_mse(mse):
    return -10.0 * torch.log(mse) / np.log(10.0)

def compute_psnr(pred, target, mask=None):
    """Compute psnr value (we assume the maximum pixel value is 1)."""
    # 0~1
    if mask is not None:
        if type(mask)==np.ndarray:
            mask = torch.tensor(mask,dtype=torch.long)
        mask = torch.tile(mask,[1,1,3])
        pred, target = pred*mask, target*mask#pred[mask], target[mask]
    mse = torch.sum((pred - target) ** 2)/torch.sum(mask)
    return compute_psnr_from_mse(mse)

def compute_ssim(pred, target, mask=None):
    """Computes Masked SSIM following the neuralbody paper."""
    # 0~1
    assert pred.shape == target.shape and pred.shape[-1] == 3
    if mask is not None:
        x, y, w, h = cv2.boundingRect(mask.cpu().numpy().astype(np.uint8))
        pred = pred[y : y + h, x : x + w]
        target = target[y : y + h, x : x + w]
    try:
        ssim = structural_similarity(
            pred.cpu().numpy(), target.cpu().numpy(), channel_axis=-1
        )
    except ValueError:
        ssim = structural_similarity(
            pred.cpu().numpy(), target.cpu().numpy(), multichannel=True
        )
    return ssim



def load_preprocess_img(path):
    img = np.array(load_image(path))
    pred_img, gt_img = img[:,:512], img[:,512:]
    return pred_img, gt_img 

def load_preprocess_mask(path):
    mask = np.array(load_image(path))[:,:,0]
    mask = (mask!=2)*(mask!=10)*(mask!=13)*(mask!=14)*(mask!=15) #hair, neck, face, lefthand, righthand
    mask = cv2.resize(mask.astype(np.float32), None, fx=0.5, fy=0.5)
    return mask[...,None] #512,512,1


def main(args):
    output_txt = args.pred_dir+'.maskedpsnr.txt'
    metricswriter = MetricsWriter(
        output_dir = os.path.dirname(args.pred_dir),
        exp_name=os.path.basename(args.pred_dir)+'_masked',
        dataset=os.path.basename(args.pred_dir)
    )
    for imgname in tqdm(os.listdir(args.pred_dir)):
        pred_img, gt_img = load_preprocess_img(os.path.join(args.pred_dir, imgname))
        head_hand_mask = load_preprocess_mask(args.mask_path_format.format(imgname[:-4].split('_')[1]))
        metricswriter.append(imgname, pred_img, gt_img, head_hand_mask)

    metricswriter.finalize()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir',required=True)
    #parser.add_argument('--gt_dir',default='dataset/zju_mocap/387_tava_1view') #we don't need it
    parser.add_argument('--mask_path_format',default='data/zju/CoreView_387/mask_cihp/Camera_B1/{}.png')
    args = parser.parse_args()
    main(args)
