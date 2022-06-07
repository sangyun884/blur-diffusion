from pytorch_fid.fid_score import compute_statistics_of_path, calculate_frechet_distance
from pytorch_fid.inception import InceptionV3
import os
class FID(object):
    def __init__(self, real_dir, device, bsize = 128):
        # https://github.com/mseitzer/pytorch-fid
        # If real_dir contains a .npz file, then we'll just use that.
        # Otherwise it's assumed to be a directory.
        # In that case, statistics will be computed, and .npz file will be saved.
        if os.path.exists(os.path.join(real_dir, "inception_statistics.npz")):
            real_dir = os.path.join(real_dir, "inception_statistics.npz")
        self.dims = 2048
        self.bsize = bsize
        self.device = device
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[self.dims]
        self.model = InceptionV3([block_idx]).to(device)
        m, s = compute_statistics_of_path(real_dir, self.model, bsize, self.dims, device)
        self.mu_real = m
        self.sig_real = s
    def __call__(self, fake_dir):
        m, s = compute_statistics_of_path(fake_dir, self.model, self.bsize, self.dims, self.device)
        fid_value = calculate_frechet_distance(m, s, self.mu_real, self.sig_real)
        return fid_value