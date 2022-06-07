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
from fid import FID
from scipy.integrate import solve_ivp

parser = argparse.ArgumentParser(description='Configs')
parser.add_argument('--gpu', type=str, help='gpu num')
parser.add_argument('--dataset', type=str, help='cifar10 / mnist')
parser.add_argument('--name', type=str, help='Saving directory name')
parser.add_argument('--ckpt', default='', type=str, help='UNet checkpoint')

parser.add_argument('--bsize', default=16, type=int, help='batchsize')
parser.add_argument('--N', default=1000, type=int, help='Max diffusion timesteps')
parser.add_argument('--sig', default=0.4, type=float, help='sigma value for blur kernel')
parser.add_argument('--sig_min', default=0, type=float, help='sigma value for blur kernel')
parser.add_argument('--sig_max', default=0.1, type=float, help='sigma value for blur kernel')
parser.add_argument('--lr', default=0.00005, type=float, help='learning rate')
parser.add_argument('--noise_schedule', default='linear', type=str, help='Type of noise schedule to use')
parser.add_argument('--betamin', default=0.0001, type=float, help='beta (min). get_score(1) can diverge if this is too low.')
parser.add_argument('--betamax', default=0.02, type=float, help='beta (max)')
parser.add_argument('--fromprior', default=True, type=bool, help='start sampling from prior')
parser.add_argument('--gtscore', action='store_true', help='Use ground truth score for reverse diffusion')
parser.add_argument('--max_iter', default=1500000, type=int, help='max iterations')
parser.add_argument('--eval_iter', default=10000, type=int, help='eval iterations')
parser.add_argument('--fid_iter', default=50000, type=int, help='eval iterations')
parser.add_argument('--fid_num_samples', default=10000, type=int, help='eval iterations')
parser.add_argument('--fid_bsize', default=32, type=int, help='eval iterations')
parser.add_argument('--loss_type', type=str, default = 'eps_simple', choices=['sm_simple', 'eps_simple', 'sm_exact', 'std_matching'])
parser.add_argument('--f_type', type=str, default = 'linear', choices=['linear', 'log', 'quadratic', 'cubic', 'quartic', 'triangular'])
parser.add_argument('--dropout', default=0, type=float, help='dropout')

# EMA, save
parser.add_argument('--use_ema', action='store_true',
                    help='use EMA or not')
parser.add_argument('--inference', action='store_true')
parser.add_argument('--freq_feat', action='store_true', help = "concat Utx_i")
parser.add_argument('--ode', action='store_true', help = "ODE fast sampler")
parser.add_argument('--ema_decay', type=float, default=0.9999, help='decay rate for EMA')
parser.add_argument('--save_every', type=int, default=50000, help='How often we wish to save ckpts')
opt = parser.parse_args()

dataset = opt.dataset
device = torch.device(f'cuda:{opt.gpu}')
device = torch.device('cuda')
print("N:", opt.N)
N = opt.N
bsize = opt.bsize
beta_min = opt.betamin
beta_max = opt.betamax
sig = opt.sig

if opt.gtscore:
    opt.fromprior = True
if dataset == 'mnist':
    train_dir = '/home/ubuntu/code/sangyoon/datasets/mnist_train'
    dataset_train = dsets.MNIST(root='/home/ubuntu/code/sangyoon/datasets/mnist_train',
                                train=True,
                                transform=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()]),
                                download=True)
    dataset_test = dsets.MNIST(root='/home/ubuntu/code/sangyoon/datasets/mnist_test',
                               train=False,
                               transform=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()]),
                               download=True)

elif dataset == 'cifar10':
    train_dir = "/home/ubuntu/code/sangyoon/datasets/cifar10_train/png"
    dataset_train = dsets.CIFAR10(root='../datasets/cifar10_train',
                                  train=True,
                                  transform=transforms.Compose([transforms.RandomHorizontalFlip(0.5), transforms.ToTensor()]),
                                  download=True
                                  )
    dataset_test = dsets.CIFAR10(root='../datasets/cifar10_test',
                                 train=False,
                                 transform=transforms.Compose([transforms.RandomHorizontalFlip(0.5), transforms.ToTensor()]),
                                 download=True
                                 )
elif dataset == 'lsun-bedroom':
    res = 64
    train_dir = "/home/ubuntu/code/sangyoon/forward-blur/datasets/bedroom_train/bedroom_train_lmdb/imgs"
    dataset_train = dsets.LSUN(root='../forward-blur/datasets/bedroom_train',
                                 classes=['bedroom_train'],
                                  transform=transforms.Compose([transforms.Resize((res,res)), transforms.ToTensor()])
                                  )
    dataset_test = dsets.LSUN(root='../forward-blur/datasets/bedroom_train',
                                    classes=['bedroom_train'],
                                 transform=transforms.Compose([transforms.Resize((res,res)), transforms.ToTensor()])
                                 )
elif dataset == 'lsun-church':
    res = 64
    train_dir = "/home/ubuntu/code/sangyoon/datasets/church_train/church_outdoor_train_lmdb/imgs"
    dataset_train = dsets.LSUN(root='/home/ubuntu/code/sangyoon/datasets/church_train',
                                 classes=['church_outdoor_train'],
                                  transform=transforms.Compose([transforms.Resize((res,res)), transforms.ToTensor()])
                                  )
    dataset_test = dsets.LSUN(root='/home/ubuntu/code/sangyoon/datasets/church_test',
                                    classes=['church_outdoor_val'],
                                 transform=transforms.Compose([transforms.Resize((res,res)), transforms.ToTensor()])
                                 )

fid_eval = FID(real_dir = train_dir, device = device)
resolution = dataset_train[0][0].shape[-1]
input_nc = dataset_train[0][0].shape[0]
ksize = resolution * 2 - 1
pad = 0

# define forward blur
kernel = gaussian_kernel_1d(ksize, sig)
blur = Deblurring(kernel, input_nc, resolution, device=device)
print("blur.U_small.shape:", blur.U_small.shape)
D_diag = blur.singulars()
fb = ForwardBlurIncreasing(N=N, beta_min=beta_min, beta_max=beta_max, sig=sig, sig_max = opt.sig_max, sig_min = opt.sig_min, D_diag=D_diag,
                    blur=blur, channel=input_nc, device=device, noise_schedule=opt.noise_schedule, resolution=resolution, pad=pad, f_type=opt.f_type)
dir = os.path.join('experiments', opt.name)
writer = SummaryWriter(dir)

# # Bedroom config - ADM
# model_channels = 256
# num_res_blocks = 2
# channel_mult = (1, 1, 2, 2, 4, 4)
# attention_resolutions = [8,16,32]
# out_channels = 6 # mean and varaince
# dropout = 0.1
# resblock_updown = True
# # bsize 256
# # iter -- cat (200K), horse (250K), bedroom (500K)
# lr = 1e-5 # Fine tuning
# model = UNetModel(image_size = resolution, in_channels = input_nc, model_channels = model_channels, out_channels = out_channels,
#                   num_res_blocks=num_res_blocks, channel_mult=channel_mult, attention_resolutions=attention_resolutions,
#                   dropout = dropout, resblock_updown = resblock_updown)

model = UNetModel(resolution, input_nc, 128, input_nc, blur = blur, dropout=opt.dropout, freq_feat = opt.freq_feat)
if not opt.ckpt == '' and os.path.exists(opt.ckpt):
    model.load_state_dict(torch.load(opt.ckpt))
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  model = DataParallel(model)

model.to(device)
print("input_nc", input_nc, "resolution", resolution)

data_loader = torch.utils.data.DataLoader(dataset=dataset_train,
                                          batch_size=bsize,
                                          shuffle=True,
                                          drop_last=True)
data_loader_test = torch.utils.data.DataLoader(dataset=dataset_test,
                                               batch_size=bsize,
                                               shuffle=False,
                                               drop_last=True)
optimizer = optim.Adam(model.parameters(), lr=opt.lr)
if opt.use_ema:
    optimizer = EMA(optimizer, ema_decay=opt.ema_decay)

# forward process visualization
sample = dataset_train[1][0].unsqueeze(0)

x_0 = sample[:4]
x_0 = x_0.to(device)
i = np.array([500] * x_0.shape[0])
i = torch.from_numpy(i).to(device)
fb.sanity(x_0, i)

sample_list = []
for i in range(0, N+1, N//10):
    if i == 0:
        sample_list.append(x_0[0])
        continue
    i = np.array([i] * x_0.shape[0])
    i = torch.from_numpy(i).to(device)
    x_i = fb.get_x_i(x_0, i)
    sample_list.append(x_i[0])
    print(f"x_{i.item()}.std() = {x_i.std()}")
    print(f"x_{i.item()}.mean() = {x_i.mean()}")


grid_sample = torch.cat(sample_list, dim=2)
utils.tensor_imsave(grid_sample, "./" + dir, "forward_process.jpg")
with open(os.path.join(dir, "config.json"), "w") as json_file:
    json.dump(vars(opt), json_file)
import time
meta_iter = 0
for step in range(opt.max_iter):
    if not opt.inference:
        elips = time.time()
        try:
            x_0, _ = train_iter.next()
        except:
            train_iter = iter(data_loader)
            image, _ = train_iter.next()
        """
        training
        """
        assert x_0.shape[-1] == resolution, f"{x_0.shape}"
        i = np.random.uniform(1 / N, 1, size = (x_0.shape[0])) * N
        i = torch.from_numpy(i).to(device).type(torch.long)

        x_0 = x_0.to(device)
        x_i, eps = fb.get_x_i(x_0, i, return_eps = True)

        if opt.loss_type == "sm_simple":
            loss = fb.get_loss_i_simple(model, x_0, x_i, i)
        elif opt.loss_type == "eps_simple":
            loss = fb.get_loss_i_eps_simple(model, x_i, i, eps)
        elif opt.loss_type == "sm_exact":
            loss = fb.get_loss_i_exact(model, x_0, x_i, i)
        elif opt.loss_type == "std_matching":
            loss = fb.get_loss_i_std_matching(model, x_i, i, eps)
        writer.add_scalar('loss_train', loss, step)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if step % 100 == 0:
            print(step, loss)
        # print(f"time: {time.time() - elips}")
    # Calcuate FID
    if step > 240001:
        fid_iter = opt.fid_iter
    else:
        fid_iter = 240000
    if (step % fid_iter == 0 and step > 0):
        id = 0
        if not os.path.exists(os.path.join("./",dir, f"{step}")):
            os.mkdir(os.path.join("./",dir, f"{step}"))
        with torch.no_grad():
            if opt.use_ema:
                optimizer.swap_parameters_with_ema(store_params_in_ema=True)
            model = model.eval()
            for _ in range(opt.fid_num_samples // opt.fid_bsize):
                i = np.array([opt.N - 1] * opt.fid_bsize)
                i = torch.from_numpy(i).to(device)
                pred = fb.get_x_N([opt.fid_bsize, input_nc, resolution, resolution], i)
                for i in reversed(range(1, opt.N)):
                    i = np.array([i] * opt.fid_bsize)
                    i = torch.from_numpy(i).to(device)
                    if opt.loss_type == "sm_simple":
                        s = model(pred, i)
                    elif opt.loss_type == "eps_simple":
                        eps = model(pred, i)
                        s = fb.get_score_from_eps(eps, i)
                    elif opt.loss_type == "sm_exact":
                        s = model(pred, i)
                    elif opt.loss_type == "std_matching":
                        std = model(pred, i)
                        s = fb.get_score_from_std(std, i)
                    else:
                        raise NotImplementedError
                    s = fb.U_I_minus_B_Ut(s, i)
                    rms = lambda x: torch.sqrt(torch.mean(x ** 2))
                    # print(f"rms(s) * fb._beta_i(i) = {rms(s) * fb._beta_i(i)[0]}")
                    hf = pred - fb.W(pred, i)
                    # Anderson theorem
                    pred1 = pred + hf # unsharpening mask filtering
                    pred2 = pred1 + s  # # denoising
                    if i[0] > 2:
                        pred = pred2 + fb.U_I_minus_B_sqrt_Ut(torch.randn_like(pred), i) # inject noise
                    else:
                        pred = pred2
                    # print(f"i = {i[0]}, rmse = {torch.sqrt(torch.mean(pred**2))}, mean = {torch.mean(pred)} std = {torch.std(pred)}" )
                for sample in pred:
                    save_image(sample, os.path.join(dir, f"{step}", f"{id:05d}.png"))
                    id += 1
        if opt.use_ema:
            optimizer.swap_parameters_with_ema(store_params_in_ema=True)
            model = model.train()
        fid = fid_eval(os.path.join(dir, f"{step}"))
        writer.add_scalar('fid', fid, step)
        print(f"step {step}, fid = {fid}")
    if (step % opt.eval_iter == 0 and step > 0) or opt.inference:
        """
        sampling (eval)
        """
        cnt = 0
        loss = 0
        

        with torch.no_grad():
            if opt.use_ema:
                optimizer.swap_parameters_with_ema(store_params_in_ema=True)
            model = model.eval()

            if opt.ode:
                raise NotImplementedError
                def to_flattened_numpy(x):
                    """Flatten a torch tensor `x` and convert it to numpy."""
                    return x.detach().cpu().numpy().reshape((-1,))
                def from_flattened_numpy(x, shape):
                    """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
                    return torch.from_numpy(x.reshape(shape))
                def ode_func(i, y):
                    i = int(i*N)
                    print(f"i = {i}")
                    y = from_flattened_numpy(y, [bsize, input_nc, resolution, resolution]).to(device).type(torch.float32)
                    i = np.array([N - 1] * bsize)
                    i = torch.from_numpy(i).to(device)
                    if opt.loss_type == "sm_simple":
                            s = model(y, i)
                    elif opt.loss_type == "eps_simple":
                        eps = model(y, i)
                        s = fb.get_score_from_eps(eps, i)
                    elif opt.loss_type == "sm_exact":
                        s = model(y, i)
                    elif opt.loss_type == "std_matching":
                        std = model(y, i)
                        s = fb.get_score_from_std(std, i)
                    else:
                        raise NotImplementedError
                    s = fb.U_I_minus_B_Ut(s, i)
                    hf = y - fb.W(y, i)
                    dt = - 1.0 / N
                    drift = (s/2 + hf) / dt
                    drift = to_flattened_numpy(drift)
                    return drift
                x_N = fb.get_x_N([bsize, input_nc, resolution, resolution], N)
                solution = solve_ivp(ode_func, (1, 1e-3), to_flattened_numpy(x_N),
                                     rtol=1e-3, atol=1e-3, method="RK45")
                nfe = solution.nfev
                solution = torch.tensor(solution.y[:, -1]).reshape(x_N.shape).to(device).type(torch.float32)
                
                save_image(solution, "./solution.jpg")
                print(f"nfe = {nfe}")
                raise NotImplementedError
            for x_0, _ in data_loader_test:
                x_0 = x_0.to(device)
                # for v in range(0, 250, 20):
                #     x_0[:, :, v, :] = 0
                if opt.fromprior:
                    i = np.array([N - 1] * x_0.shape[0])
                    i = torch.from_numpy(i).to(device)
                    pred = fb.get_x_N(x_0.shape, i)
                    print(f"pred.std() = {pred.std()}")
                else:
                    i = np.array([N-1] * x_0.shape[0])
                    i = torch.from_numpy(i).to(device)
                    pred = fb.get_x_i(x_0, i)
                preds = [pred]

                for i in reversed(range(1, N)):
                    i = np.array([i] * x_0.shape[0])
                    i = torch.from_numpy(i).to(device)
                    if opt.gtscore:
                        s = fb.get_score_gt(pred, x_0, i)
                    else:
                        if opt.loss_type == "sm_simple":
                            s = model(pred, i)
                        elif opt.loss_type == "eps_simple":
                            eps = model(pred, i)
                            s = fb.get_score_from_eps(eps, i)
                        elif opt.loss_type == "sm_exact":
                            s = model(pred, i)
                        elif opt.loss_type == "std_matching":
                            std = model(pred, i)
                            s = fb.get_score_from_std(std, i)
                        else:
                            raise NotImplementedError
                    s = fb.U_I_minus_B_Ut(s, i)
                    rms = lambda x: torch.sqrt(torch.mean(x ** 2))
                    # print(f"rms(s) * fb._beta_i(i) = {rms(s) * fb._beta_i(i)[0]}")
                    hf = pred - fb.W(pred, i)
                    # Anderson theorem
                    pred1 = pred + hf # unsharpening mask filtering
                    pred2 = pred1 + s  # # denoising
                    if i[0] > 2:
                        pred = pred2 + fb.U_I_minus_B_sqrt_Ut(torch.randn_like(pred), i) # inject noise
                    else:
                        pred = pred2
                    # print(f"i = {i[0]}, rmse = {torch.sqrt(torch.mean(pred**2))}, mean = {torch.mean(pred)} std = {torch.std(pred)}")
                    # assert rms(pred) < 100
                    if (i[0]) % (N // 10) == 0:
                        img = pred[0]
                        preds.append(pred)

                preds.append(pred)
                assert x_0.shape == pred.shape
                # visualize
                grid = torch.cat(preds, dim=3) # grid_sample.shape: (bsize, channel, H, W * 12)
                # (batch_size, channel, H, W * 12) -> (channel, H * bsize, W * 12)
                grid = grid.permute(1, 0, 2, 3).contiguous().view(grid.shape[1], -1, grid.shape[3])
                # (bsize, channel, H, W) -> (channel, H, W * bsize)
                gt = x_0.permute(1, 2, 0, 3).contiguous().view(x_0.shape[1], -1, x_0.shape[3] * x_0.shape[0])
                if cnt <= 2:
                    utils.tensor_imsave(gt, "./" + dir, f"{step}_{cnt}_GT.jpg")
                    utils.tensor_imsave(grid, "./" + dir, f"{step}_{cnt}_pred.jpg")
              
                cnt += 1
                loss += TF.l1_loss(x_0, pred) / 2

                if cnt == 2:
                    break
        print(f"step: {step} loss: {loss}")
        writer.add_scalar('loss_val', loss, meta_iter)
        f = open('./' + str(dir) + '/log.txt', 'a')

        f.write(f"Step: {step} loss: {loss}" + '\n')

        f.close()
        model.train()
        if opt.use_ema:
            optimizer.swap_parameters_with_ema(store_params_in_ema=True)
    if step % opt.save_every == 1:
        if opt.use_ema:
            optimizer.swap_parameters_with_ema(store_params_in_ema=True)
        if torch.cuda.device_count() > 1:
            torch.save(model.module.state_dict(), os.path.join(dir, f"model_{step}.ckpt"))
        else:
            torch.save(model.state_dict(), os.path.join(dir, f"model_{step}.ckpt"))
        if opt.use_ema:
            optimizer.swap_parameters_with_ema(store_params_in_ema=True)
