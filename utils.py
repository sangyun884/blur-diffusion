from torchvision import transforms
from PIL import Image
import os
from torch import nn
import torch
import glob
import numpy as np
import torch.nn.functional as F
def tensor2pil(t):
    img = transforms.functional.to_pil_image(denorm(t))
    return img
def denorm(t):
    return t*0.5 + 0.5
def tensor_imsave(t, path, fname, denormalization=False, prt=False):
    # Save tensor as .png file
    check_folder(path)
    if denormalization: t = denorm(t)
    t = torch.clamp(input=t, min=0., max=1.)
    img = transforms.functional.to_pil_image(t.detach().cpu())
    img.save(os.path.join(path,fname))
    if prt:
        print(f"Saved to {os.path.join(path,fname)}")
def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:##Automatic closing using with. rb = read binary
        img = Image.open(f)
        return img.convert('RGB')
def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir
def freeze_model(model):
    for p in model.parameters():
        p.requires_grad = False
class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

def pil_batch_loader(root_path, bsize):
    # Returns the list of PIL.Images at given path. Length of list is bsize.
    flist = sorted(glob.glob(root_path, '*.png'))
    flist = flist[:bsize]
    result = []
    for p in flist:
        img = pil_loader(p)
        result.append(img)
    return result

class PSNR(object):
    def __init__(self, gpu, val_max=1, val_min=0, ycbcr=True):
        super(PSNR,self).__init__()
        self.val_max = val_max
        self.val_min = val_min
        self.gpu = gpu
        self.ycbcr = ycbcr
    
    def __call__(self,x,y):
        """
        if x.is_cuda:
            x = x.detach().cpu()
        if y.is_cuda:
            y = y.detach().cpu()
        """
        assert len(x.size()) == len(y.size())
        with torch.no_grad():
            x_lum = rgb_to_ycbcr(x)[:,0]
            y_lum = rgb_to_ycbcr(y)[:,0]
            # if len(x.size()) == 3:
            #     mse = torch.mean((y-x)**2)
            #     psnr = 20*torch.log10(torch.tensor(self.val_max-self.val_min, dtype=torch.float).cuda(self.gpu)) - 10*torch.log10(mse)
            #     return psnr
            # elif len(x.size()) == 4:
        
            mse = torch.mean((y_lum-x_lum)**2, dim=[1,2])
            psnr = 20*torch.log10(torch.tensor(self.val_max-self.val_min, dtype=torch.float).cuda(self.gpu)) - 10*torch.log10(mse)
            return torch.mean(psnr)

def rgb_to_ycbcr(image):
    r"""Convert an RGB image to YCbCr.
    Args:
        image (torch.Tensor): RGB Image to be converted to YCbCr.
    Returns:
        torch.Tensor: YCbCr version of the image.
    """

    if not torch.is_tensor(image):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(image.shape))

    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]

    delta = .5
    y: torch.Tensor = .299 * r + .587 * g + .114 * b
    cb: torch.Tensor = (b - y) * .564 + delta
    cr: torch.Tensor = (r - y) * .713 + delta
    return torch.stack((y, cb, cr), -3)

def clear(x):
    return x.detach().cpu().squeeze().numpy()

def normalize_np(x):
    x -= x.min()
    x /= x.max()
    return x

def pad(x, pad):
    return F.pad(x, (pad, pad, pad, pad), 'reflect')

def crop(x, pad):
    if pad == 0:
        return x
    else:
        return x[..., pad:-pad, pad:-pad]
