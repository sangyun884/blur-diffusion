import torch
import torch.nn.functional as F
import time
import utils
import os
import numpy as np

from utils import clear

def gaussian_kernel_1d(kernel_size, sigma):
    assert sigma > 0.00001
    x_cord = torch.arange(kernel_size)
    mean = (kernel_size - 1) / 2.

    # pdf of 1d gaussian not considering normalization constant
    gaussian_kernel = torch.exp(-0.5 * ((x_cord - mean)/sigma)**2)
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    return gaussian_kernel


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Ported from https://github.com/openai/guided-diffusion
    Utility function for cosine noise schedule
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

class ExpSchedule:
    def __init__(self, N, offset = 1e-4):
        self.N = N
        def f(i):
            #return 1 - np.sin((i / N + 1) * np.pi / 2) + offset
            return np.exp(5*i/N - 5) + offset
        idxs = np.arange(N+1)
        self.alphas_bar = 1 - f(idxs) / f(idxs[-1])
        self.alphas_bar_left_shifted = 1 - f(idxs-1) / f(idxs[-1])
        self.alphas = self.alphas_bar / self.alphas_bar_left_shifted
        self.betas = 1 - self.alphas
    def get_betas(self):
        return self.betas
        


# Increase blur strength as i increases
class ForwardBlurIncreasing:
    def __init__(self, N, beta_min, beta_max, sig, sig_min, sig_max, D_diag, blur=None, noise_schedule='linear',
                 channel=3, resolution=32, pad=None, device='cuda:0', f_type = 'linear'):
        # N: total number of discretizations used
        self.device = device
        self.N = N
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.sig = sig
        self.sig_min = sig_min # \sigma_1
        self.sig_max = sig_max # \sigma_N
        
        self.D_diag = D_diag
        # total dimension of image
        self.dim = self.D_diag.shape[0]
        # "deblur" class
        self.blur = blur

        self.resolution = resolution
        self.channel = channel
        self.pad = pad

        self.noise_schedule = noise_schedule
        if noise_schedule == 'linear':
            self.betas = torch.linspace(beta_min, beta_max, N, device=device)
        elif noise_schedule == 'cosine':
            self.betas = torch.tensor(betas_for_alpha_bar(
                N,
                lambda t: np.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2,
            ), device=device, dtype=torch.float)
        elif noise_schedule == 'exp':
            Exp = ExpSchedule(N)
            betas = torch.tensor(Exp.get_betas())
            self.betas = torch.tensor(betas, device=device)
        self.betas = F.pad(self.betas, (1, 0), value=0.0)
        self.alphas = 1 - self.betas
        self.sqrt_alphas = torch.sqrt(self.alphas)
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_bar = torch.cumprod(self.sqrt_alphas, dim=0)
        self.alphas_bar_prev = F.pad(self.alphas_bar, [1, 0], value=1)[:N]

        # def f(i):
        #     a = (self.sig_max**2 - self.sig_min**2) / (self.sig**2 * (self.N**2 - 1))
        #     b = (self.sig_min / self.sig) ** 2 - a
        #     return a * i**2 + b
        def f_linear(i):
            f_N = (self.sig_max / self.sig) ** 2
            f_1 = (self.sig_min / self.sig) ** 2
            return (f_N - f_1)/ (N-1) * (i - 1) + f_1
        
        idxs = torch.tensor(np.arange(N+1), device=device, dtype=torch.float)
        
        def f_log(i):
            log = lambda x: torch.log(x + 1e-6) / (10*np.log(N))
            f_N = (self.sig_max / self.sig) ** 2
            f_1 = (self.sig_min / self.sig) ** 2
            a = (f_N - f_1) / log(torch.tensor(N))
            b = f_1
            return a*log(i) + b
        def f_quadratic(i):
            f_N = (sig_max / sig) ** 2
            f_1 = (sig_min / sig) ** 2
            a = (f_N - f_1) / (N**2 - 1)
            b = f_1 - (f_N - f_1) / (N**2 - 1)
            return a*i**2 + b
        def cubic(i):
            f_N = (sig_max / sig) ** 2
            f_1 = (sig_min / sig) ** 2
            a = (f_N-f_1) / N**3
            b = f_1
            return a*i**3 + b
        def quartic(i):
            f_N = (sig_max / sig) ** 2
            f_1 = (sig_min / sig) ** 2
            a = (f_N-f_1) / N**4
            b = f_1
            return a*i**4 + b
        def triangular(i):
            less_than_N_2 = (i < N/2).type(torch.int)
            return f_linear(i) * less_than_N_2 + f_linear(N-i) * (1-less_than_N_2 )

        if f_type == 'linear':
            f = f_linear
        elif f_type == 'log':
            f = f_log
        elif f_type == 'quadratic':
            f = f_quadratic
        elif f_type == 'cubic':
            f = cubic
        elif f_type == 'quartic':
            f = quartic
        elif f_type == 'triangular':
            f = triangular
        else:
            raise NotImplementedError
        self.fs = f(idxs)
        print("fs: ", self.fs)
        self.fs_cum = torch.cumsum(self.fs, dim=0)
        idxs_long = torch.tensor(np.arange(N+1), device=device, dtype=torch.long)
        def B(i):
            p = (2 * f(i)).unsqueeze(-1).repeat(1, self.D_diag.shape[0])
            print("p: ", p.shape)
            D = self.D_diag.repeat(p.shape[0], 1)
            print("D: ", D.shape)
            return self.alphas[i].unsqueeze(-1) * self.D_diag ** p
        self.Bs = B(idxs_long)
        

        
        self.Bs_bar = F.pad(torch.cumprod(self.Bs[1:], dim=0), (0,0,1, 0), value=0)

        

       
        self.one_minus_Bs_bar = 1 - self.Bs_bar
        self.one_minus_Bs_bar_sqrt = torch.sqrt(self.one_minus_Bs_bar)
        self.Bs_sqrt = torch.sqrt(self.Bs)
        self.Bs_squared = self.Bs ** 2
        self.Bs_bar_sqrt = torch.sqrt(self.Bs_bar)
        
        
        # fname = f'res{resolution}_N{N}_sig{sig}_sigmin_{sig_min}_sigmax_{sig_max}_b_mat_sq-nc{channel}_{noise_schedule}_inc_nopad_rotated.npz'
        # if os.path.isfile(fname):
        #     with np.load(fname, allow_pickle=True) as data:
        #         print("data", list(data.keys()))
        #         data = data['arr'].astype(np.float32)
        #         b_mat_sq = torch.tensor(data, device=device)
        #     self.b_mat_sq = b_mat_sq
        # else:
        #     tic = time.time()
        #     self.b_mat_sq = torch.zeros([N + 1, self.dim]).to(device)
        # self.b_mat = torch.sqrt(self.b_mat_sq)

    def _beta_i(self, i):
        return self.betas[i]
    
    def _alpha_i(self, i):
        return self.alphas[i]

    def _alphas_bar_i(self, i):
        return self.alphas_bar[i]

    def _Bhat_i(self, i):
        return self.b_mat[i, :]

    # Forward blur
    def _Bhat_sq_i(self, i):
        assert i != 0
        return self.b_mat_sq[i, :]


    
    # Forward blur
    def get_mean(self, x0, i):
        mat = self.Bs_bar_sqrt[i]
        mean = self.blur.U(mat * self.blur.Ut(x0))
        return mean

    # Forward blur
    def get_std(self,i, noise):
        mat = self.one_minus_Bs_bar_sqrt[i]
        std = self.blur.U(mat * self.blur.Ut(noise))
        return std

    # Forward blur
    def get_var(self, x0, i, noise=None):
        raise NotImplementedError
        Bhat_i = self._Bhat_sq_i(i)
        noise = noise if noise is not None else torch.randn_like(x0)
        std = self.blur.U(Bhat_i * self.blur.Ut(noise))
        return std

    def W(self, x, i):
        mat = self.Bs_sqrt[i]
        batch = x.shape[0]
        blurred = self.blur.U(mat * self.blur.Ut(x)).view(batch,
                                          self.channel,
                                          self.resolution,
                                          self.resolution)
        return blurred
        

    def W_inv(self, x, i):
        mat = self.Bs_squared[i]
        batch = x.shape[0]
        blurred = self.blur.U(mat * self.blur.Ut(x)).view(batch,
                                          self.channel,
                                          self.resolution,
                                          self.resolution)
        return blurred
    def U_I_minus_B_Ut(self, x, i): # U(I-B)UT
        batch = x.shape[0]
        mat = 1 - self.Bs[i]
        result = self.blur.U(mat * self.blur.Ut(x)).view(batch,
                                          self.channel,
                                          self.resolution,
                                          self.resolution)
        return result
    
    def U_I_minus_B_sqrt_Ut(self, x, i): # U(sqrt(I-B))UT
        batch = x.shape[0]
        mat = torch.sqrt(1 - self.Bs[i])
        result = self.blur.U(mat * self.blur.Ut(x)).view(batch,
                                          self.channel,
                                          self.resolution,
                                          self.resolution)
        return result
    
    # Forward blur
    def get_x_i(self, x0, i, return_eps = False):
        assert 0 not in i
        if len(x0.shape) == 4:
            batch = x0.shape[0]
        # reparam trick
        mean = self.get_mean(x0, i)
        noise = torch.randn_like(x0)
        std = self.get_std(i, noise = noise)
        # crop
        if len(x0.shape) == 4:
            img = (mean + std).view(batch,
                                                self.channel,
                                                self.resolution,
                                                self.resolution)
        else:
            img = (mean + std).view(self.channel,
                                                self.resolution,
                                                self.resolution)
        
        # cropped = cropped + torch.randn_like(cropped) * torch.sqrt(self.betas[i-1])
        if return_eps:
            return img, noise
        return img

    def get_x_N(self, x0_shape, N):
        """
        prior sampling
        """
        if len(x0_shape) == 4:
            batch = x0_shape[0]
        # pad
        x0 = torch.zeros(x0_shape, device=self.device)
        noise = torch.randn_like(x0)
        std = self.get_std(N, noise)
        # crop
        if len(x0_shape) == 4:
            img = std.view(batch,
                                   self.channel,
                                   self.resolution,
                                   self.resolution)
        else:
            img = std.view(self.channel,
                                   self.resolution,
                                   self.resolution)
     
        return img

    
    def get_x0_from_eps(self, xi, eps, i):
        batch = xi.shape[0]
        std = self.get_std(i, noise = eps).view(batch,
                                   self.channel,
                                   self.resolution,
                                   self.resolution)
        mean = xi - std
        return mean / torch.sqrt(self.alphas_bar[i]).view(-1, 1, 1, 1)
        mat = self.Bs_bar_sqrt[i] ** (-1)
        x0 = self.blur.U(mat * self.blur.Ut(mean)).view(batch,
                                   self.channel,
                                   self.resolution,
                                   self.resolution)
        return x0
    
    def get_score_gt(self, xi, x0, i):
        # Return the ground-truth score of x_i
        assert torch.all(i>0)
        batch = x0.shape[0]
        
        
        mean = self.get_mean(x0, i).view(batch,
                                                 self.channel,
                                                 self.resolution,
                                                 self.resolution)

        diff = xi - mean
        mat = (self.one_minus_Bs_bar[i])**(-1)
        score = -self.blur.U(mat * self.blur.Ut(diff)).view(batch,
                                                            self.channel,
                                                            self.resolution,
                                                            self.resolution)                                 
        return score
    def get_score_gt2(self, xi, x0, i):
        # Return the ground-truth score of x_i
        batch = x0.shape[0]
        x0_pad = utils.pad(x0, self.pad)
        # a_iW^ix_0
        mean_pad = self.get_mean(x0_pad, i).view(batch,
                                                 self.channel,
                                                 self.resolution + self.pad * 2,
                                                 self.resolution + self.pad * 2)

        mean = utils.crop(mean_pad, self.pad)
        diff = xi - mean
        diff = utils.pad(diff, self.pad)
        score = self.blur.U(self._Bhat_i(i) ** (-2) * self.blur.Ut(diff)).view(batch,
                                                                               self.channel,
                                                                               self.resolution + self.pad * 2,
                                                                               self.resolution + self.pad * 2)
        score = -utils.crop(score, self.pad) 
        return score
    def sanity(self, x_0, i):
        batch = x_0.shape[0]
        tol = 1e-2
        xi, eps = self.get_x_i(x_0, i, return_eps = True)
        score1 = self.get_score_from_eps(eps, i)
        score2 = self.get_score_gt(xi, x_0, i)
        # utils.tensor_imsave(score1[0], "./", "score1.jpg")
        # utils.tensor_imsave(score2[0], "./", "score2.jpg")
        # rms = lambda x: torch.sqrt(torch.mean(x**2))
        # print(f"rms(score1) = {rms(score1)}, rms(score2) = {rms(score2)}")
        
        # noise = torch.randn_like(x_0)
        # noise_pad = utils.pad(noise, self.pad)
        
        # std = self.blur.U(self._Bhat_i(i) * self.blur.Ut(noise_pad)).view(batch,
        #                                          self.channel,
        #                                          self.resolution + self.pad * 2,
        #                                          self.resolution + self.pad * 2)
        # std = utils.crop(std, self.pad)
        # std = utils.pad(std, self.pad)
        # # crop -> pad makes error
        # score1 = self.blur.U(self._Bhat_i(i) ** (-2) * self.blur.Ut(std)).view(batch,
        #                                                                        self.channel,
        #                                                                        self.resolution + self.pad * 2,
        #                                                                        self.resolution + self.pad * 2)
        # score1 = -utils.crop(score1, self.pad)
        
        # score2 = self.blur.U(self._Bhat_i(i) ** (-1) * self.blur.Ut(noise_pad)).view(batch, self.channel, self.resolution + self.pad * 2, self.resolution + self.pad * 2)
        # score2 = -utils.crop(score2, self.pad)
        MAE = torch.mean(torch.abs(score1 - score2))
        assert MAE < tol, f'MAE = {MAE}'
        print(f"MAE = {MAE}")
        
        # score3 = self.get_score_gt2(xi, x_0, i)
        # print(f"score3 = {score3[:5]}")
        # print(f"score2 = {score2[:5]}")
        
        # MAE = torch.mean(torch.abs(score3 - score2))
        # print(f"MAE = {MAE}, fraction = {MAE/torch.mean(torch.abs(score3))}, norm: {torch.mean(torch.abs(score3))}")
        # assert MAE <1e-5, f'MAE = {MAE}'
        # MAE = torch.mean(torch.abs(self.b_mat_sq[1:] - self.one_minus_Bs_bar[1:]))
        # assert MAE <1e-5, f'MAE = {MAE}'
    def get_score_from_eps(self, eps, i):
        # Return the score of x_i
        batch = eps.shape[0]
        mat = self.one_minus_Bs_bar_sqrt[i]**(-1)
        score = -self.blur.U(mat * self.blur.Ut(eps)).view(batch, self.channel, self.resolution, self.resolution)
        
        return score
    def get_score_from_std(self, std, i):
        # Return the score of x_i
        batch = std.shape[0]
        mat = self.one_minus_Bs_bar[i] ** (-1)
        score = -self.blur.U(mat * self.blur.Ut(std)).view(batch, self.channel, self.resolution, self.resolution)
        return score
    def get_loss_i_exact(self, model, x0, xi, i):
        # Calculate the DSM objective (exact)
        batch = x0.shape[0]
        pred = model(xi, i)
        score = self.get_score_gt(x0, xi, i)
        loss = torch.mean((pred - score) ** 2)
        return loss

    def get_loss_i_simple(self, model, x0, xi, i):
        raise NotImplementedError
        # Calculate the DSM objective (simple)
        batch = x0.shape[0]
        pred = model(xi, i)
        # UB**2U^Ts
        pred = utils.pad(pred, self.pad)
        pred = self.blur.U(self._Bhat_i(i) ** 2 * self.blur.Ut(pred)).view(batch,
                                                                           self.channel,
                                                                           self.resolution + self.pad * 2,
                                                                           self.resolution + self.pad * 2)
        pred = utils.crop(pred, self.pad)
        # ai[UDUT]x0
        x0_pad = utils.pad(x0, self.pad)
        mean = self.get_mean(x0_pad, i).view(batch,
                                             self.channel,
                                             self.resolution + self.pad * 2,
                                             self.resolution + self.pad * 2)
        mean = utils.crop(mean, self.pad)
        loss = torch.mean((pred + xi - mean) ** 2)
        return loss
    def get_loss_i_eps_simple(self, model, x_i, i, eps):
        pred = model(x_i, i)
        loss = torch.mean((pred - eps) ** 2)
        return loss
    def get_loss_i_std_matching(self, model, x_i, i, eps):
        pred = model(x_i, i)
        batch = x_i.shape[0]
        std = self.get_std(i, noise = eps).view(batch,
                                   self.channel,
                                   self.resolution,
                                   self.resolution)
        loss = torch.mean((pred - std) ** 2)
        return loss

class H_functions:
    """
    Ported from https://github.com/bahjat-kawar/ddrm
    A class replacing the SVD of a matrix H, perhaps efficiently.
    All input vectors are of shape (Batch, ...).
    All output vectors are of shape (Batch, DataDimension).
    """

    def V(self, vec):
        """
        Multiplies the input vector by V
        """
        raise NotImplementedError()

    def Vt(self, vec):
        """
        Multiplies the input vector by V transposed
        """
        raise NotImplementedError()

    def U(self, vec):
        """
        Multiplies the input vector by U
        """
        raise NotImplementedError()

    def Ut(self, vec):
        """
        Multiplies the input vector by U transposed
        """
        raise NotImplementedError()

    def singulars(self):
        """
        Returns a vector containing the singular values. The shape of the vector should be the same as the smaller dimension (like U)
        """
        raise NotImplementedError()

    def add_zeros(self, vec):
        """
        Adds trailing zeros to turn a vector from the small dimension (U) to the big dimension (V)
        """
        raise NotImplementedError()

    def H(self, vec):
        """
        Multiplies the input vector by H
        """
        temp = self.Vt(vec)
        singulars = self.singulars()
        return self.U(singulars * temp[:, :singulars.shape[0]])

    def Ht(self, vec):
        """
        Multiplies the input vector by H transposed
        """
        temp = self.Ut(vec)
        singulars = self.singulars()
        return self.V(self.add_zeros(singulars * temp[:, :singulars.shape[0]]))

    def H_pinv(self, vec):
        """
        Multiplies the input vector by the pseudo inverse of H
        """
        temp = self.Ut(vec)
        singulars = self.singulars()
        temp[:, :singulars.shape[0]] = temp[:, :singulars.shape[0]] / singulars
        return self.V(self.add_zeros(temp))


class Deblurring(H_functions):
    def mat_by_img(self, M, v):
        return torch.matmul(M, v.reshape(v.shape[0] * self.channels, self.img_dim,
                                         self.img_dim)).reshape(v.shape[0], self.channels, M.shape[0], self.img_dim)

    def img_by_mat(self, v, M):
        return torch.matmul(v.reshape(v.shape[0] * self.channels, self.img_dim,
                                      self.img_dim), M).reshape(v.shape[0], self.channels, self.img_dim, M.shape[1])

    def __init__(self, kernel, channels, img_dim, device):
        self.img_dim = img_dim
        self.channels = channels
        # build 1D conv matrix
        H_small = torch.zeros(img_dim, img_dim, device=device)
        for i in range(img_dim):
            for j in range(i - kernel.shape[0] // 2, i + kernel.shape[0] // 2):
                if j < 0 or j >= img_dim: continue
                H_small[i, j] = kernel[j - i + kernel.shape[0] // 2]
        # torch.cholesky(H_small) # raise error if H_small is not positive definite
        self.H_small = H_small # positive definite matrix
        # get the evd of the 1D conv
        self.U_small, self.singulars_small, _ = torch.svd(H_small, some=False)
        self.V_small = self.U_small # V should equal U since H is symmetric
        
        ZERO = 3e-2
        # truncate
        self.singulars_small[self.singulars_small < ZERO] = ZERO
        # calculate the singular values of the big matrix using Kronecker product
        self._singulars = torch.matmul(self.singulars_small.reshape(img_dim, 1),
                                       self.singulars_small.reshape(1, img_dim)).reshape(img_dim ** 2)
        self._singulars[self._singulars > 1] = 1
        print("contains zero?", torch.any(self._singulars == 0))
        # raise NotImplementedError
        # sort the big matrix singulars and save the permutation
        self._singulars, self._perm = self._singulars.sort(descending=True)  # , stable=True)

    def V(self, vec):
        # invert the permutation
        temp = torch.zeros(vec.shape[0], self.img_dim ** 2, self.channels, device=vec.device)
        temp[:, self._perm, :] = vec.clone().reshape(vec.shape[0], self.img_dim ** 2, self.channels)
        temp = temp.permute(0, 2, 1)
        # multiply the image by V from the left and by V^T from the right
        out = self.mat_by_img(self.V_small, temp)
        out = self.img_by_mat(out, self.V_small.transpose(0, 1)).reshape(vec.shape[0], -1)
        return out

    def Vt(self, vec):
        # multiply the image by V^T from the left and by V from the right
        temp = self.mat_by_img(self.V_small.transpose(0, 1), vec.clone())
        temp = self.img_by_mat(temp, self.V_small).reshape(vec.shape[0], self.channels, -1)
        # permute the entries according to the singular values
        temp = temp[:, :, self._perm].permute(0, 2, 1)
        return temp.reshape(vec.shape[0], -1)

    def U(self, vec):
        # invert the permutation
        temp = torch.zeros(vec.shape[0], self.img_dim ** 2, self.channels, device=vec.device)
        temp[:, self._perm, :] = vec.clone().reshape(vec.shape[0], self.img_dim ** 2, self.channels)
        temp = temp.permute(0, 2, 1)
        # multiply the image by U from the left and by U^T from the right
        out = self.mat_by_img(self.U_small, temp)
        out = self.img_by_mat(out, self.U_small.transpose(0, 1)).reshape(vec.shape[0], -1)
        return out

    def Ut(self, vec):
        # multiply the image by U^T from the left and by U from the right
        temp = self.mat_by_img(self.U_small.transpose(0, 1), vec.clone())
        temp = self.img_by_mat(temp, self.U_small).reshape(vec.shape[0], self.channels, -1)
        # permute the entries according to the singular values
        temp = temp[:, :, self._perm].permute(0, 2, 1)
        return temp.reshape(vec.shape[0], -1)

    def conv1d_col_matmul(self, x):
        return torch.matmul(self.H_small, x)

    def conv1d_row_matmul(self, x):
        return torch.matmul(x, self.H_small)

    def conv2d_sep_matmul(self, x):
        return torch.matmul(torch.matmul(self.H_small, x), self.H_small)

    def singulars(self):
        if self.channels == 3:
            return self._singulars.repeat(1, 3).reshape(-1)
        else:
            return self._singulars.reshape(-1)

    def update_singulars(self, new_singulars):
        self._singulars = new_singulars

    def add_zeros(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)