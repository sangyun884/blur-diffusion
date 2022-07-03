import torch
from PIL import Image
from collections import defaultdict
import torch.nn.functional as TF
import torchvision.datasets as dsets
from torchvision import transforms
import numpy as np

import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
import utils
from guided_diffusion.unet import UNetModel
import math
from tensorboardX import SummaryWriter
import os
import json
from collections import namedtuple
import argparse
from torchvision.utils import save_image
from tqdm import tqdm
from blur_diffusion import Deblurring, ForwardBlurIncreasing, gaussian_kernel_1d
from utils import normalize_np, clear
from EMA import EMA
from torch.nn import DataParallel

parser = argparse.ArgumentParser(description='Configs')
parser.add_argument('--gpu', type=str, help='gpu num')
parser.add_argument('--name', type=str, help='Saving directory name')
parser.add_argument('--ckpt', default='', type=str, help='UNet checkpoint')

parser.add_argument('--bsize', default=16, type=int, help='batchsize')
parser.add_argument('--N', default=1000, type=int, help='Max diffusion timesteps')
parser.add_argument('--sig', default=0.3, type=float, help='sigma value for blur kernel')
parser.add_argument('--sig_min', default=0.3, type=float, help='sigma value for blur kernel')
parser.add_argument('--sig_max', default=1.5, type=float, help='sigma value for blur kernel')
parser.add_argument('--noise_schedule', default='linear', type=str, help='Type of noise schedule to use')
parser.add_argument('--betamin', default=0.0001, type=float, help='beta (min). get_score(1) can diverge if this is too low.')
parser.add_argument('--betamax', default=0.02, type=float, help='beta (max)')
parser.add_argument('--res',  type=int, help='resolution')
parser.add_argument('--nc',  type=int, help='channels')
parser.add_argument('--loss_type', type=str, default = 'sm_simple', choices=['sm_simple', 'eps_simple', 'sm_exact'])
parser.add_argument('--f_type', type=str, default = 'linear', choices=['linear', 'log', 'quadratic', 'quartic'])
parser.add_argument('--dropout', default=0, type=float, help='dropout')
parser.add_argument('--freq_feat', action='store_true', help = "concat Utx_i")
opt = parser.parse_args()

ksize = opt.res * 2 - 1
pad = 0
bsize = opt.bsize
beta_min = opt.betamin
beta_max = opt.betamax
device = torch.device(f'cuda:{opt.gpu}')
# define forward blur
kernel = gaussian_kernel_1d(ksize, opt.sig)
blur = Deblurring(kernel, opt.nc, opt.res, device=device)
print("blur.U_small.shape:", blur.U_small.shape)
D_diag = blur.singulars()
fb = ForwardBlurIncreasing(N=opt.N, beta_min=beta_min, beta_max=beta_max, sig=opt.sig, sig_max = opt.sig_max, sig_min = opt.sig_min, D_diag=D_diag,
                    blur=blur, channel=opt.nc, device=device, noise_schedule=opt.noise_schedule, resolution=opt.res, pad=pad, f_type=opt.f_type)
iter = opt.ckpt.split('/')[-1].split('.')[0]
dir = os.path.join('experiments', opt.name, f'inferences-{iter}')
if not os.path.exists(dir):
    os.mkdir(dir)
model = UNetModel(opt.res, opt.nc, 128, opt.nc, blur = blur, dropout=opt.dropout, freq_feat = opt.freq_feat)
model.load_state_dict(torch.load(opt.ckpt))
model.to(device)
model.eval()
print("input_nc", opt.nc, "resolution", opt.res)

num_samples = bsize
id = 0
for _ in tqdm(range(num_samples // bsize)):
    with torch.no_grad():
        i = np.array([opt.N - 1] * bsize)
        i = torch.from_numpy(i).to(device)
        pred = fb.get_x_N([bsize, opt.nc, opt.res, opt.res], i)
        x0_list = []
        for i in reversed(range(1, opt.N)):
            i = np.array([i] * bsize)
            i = torch.from_numpy(i).to(device)
            if opt.loss_type == "sm_simple":
                s = model(pred, i)
            elif opt.loss_type == "eps_simple":
                eps = model(pred, i)
                s = fb.get_score_from_eps(eps, i)
                x0_hat = fb.get_x0_from_eps(pred, eps, i)

                if i[0] % 100 == 0 or i[0] == 1:
                    x0_list.append(x0_hat)
            elif opt.loss_type == "sm_exact":
                s = model(pred, i)
            else:
                raise NotImplementedError
            s = fb.U_I_minus_B_Ut(s, i)
            rms = lambda x: torch.sqrt(torch.mean(x ** 2))
            print(f"rms(s) * fb._beta_i(i) = {rms(s) * fb._beta_i(i)[0]}")
            hf = pred - fb.W(pred, i)
            # Anderson theorem
            pred1 = pred + hf # unsharpening mask filtering
            pred2 = pred1 + s  # # denoising
            if i[0] > 2:
                pred = pred2 + fb.U_I_minus_B_sqrt_Ut(torch.randn_like(pred), i) # inject noise
            else:
                pred = pred2
            print(f"i = {i[0]}, rmse = {torch.sqrt(torch.mean(pred**2))}, mean = {torch.mean(pred)} std = {torch.std(pred)}" )
        grid = torch.cat(x0_list, dim=3) # grid_sample.shape: (bsize, channel, H, W * 12)
        # (batch_size, channel, H, W * 12) -> (channel, H * bsize, W * 12)
        grid = grid.permute(1, 0, 2, 3).contiguous().view(grid.shape[1], -1, grid.shape[3])
        # (bsize, channel, H, W) -> (channel, H, W * bsize)
        save_image(grid, os.path.join(dir, f'xhat.png'))
