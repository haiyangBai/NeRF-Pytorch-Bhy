import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

from losses import MSELoss
from metrics import psnr, mse
from datasets.blender import BlenderDataset
from models.nerf import NeRF, PosEmbedding
from models.rendering import render_rays
from models.utils import get_parameters
from torch.utils.data import DataLoader
from utils import get_learning_rate
from datasets.ray_utils import get_ray_directions, get_rays


# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)


def train_nerf():
    device = torch.cuda.is_available()

    # Model
    embeddings = [PosEmbedding(3, 10), PosEmbedding(3, 4)]
    nerf_coarse = NeRF(D=8, W=256, in_channels_xyz=63, in_channels_dir=27, skips=[4])
    nerf_fine = NeRF(D=8, W=256, in_channels_xyz=63, in_channels_dir=27, skips=[4])
    models = [nerf_coarse, nerf_fine]

    # Loss and Optimizer
    cal_loss = MSELoss()
    optimizer = torch.optim.Adam(get_parameters(models), lr=0.01, eps=1e-8, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=16, eta_min=1e-8)

    # Dataset Loader
    train_dataset = BlenderDataset('/home/baihy/datasets/nerf_synthetic/nerf_synthetic/lego', 'train')
    train_dataloader = DataLoader(train_dataset, shuffle=True,num_workers=8,
                                    batch_size=128, pin_memory=True)
    
    # Training
    iters = len(train_dataloader)
    t_start = time.time()
    for i, sample in enumerate(train_dataloader):
        rays, rgbs = sample['rays'], sample['rgbs']
        render_rays_chunks = render_rays(models, embeddings, rays, N_samples=64, use_disp=False, noise_std=0,
                                        N_importance=64, chunk=1024, white_back=True)
        # print(render_rays_chunks['rgb_coarse'].shape, render_rays_chunks['rgb_fine'].shape)
        optimizer.zero_grad()
        loss_total = cal_loss(render_rays_chunks, rgbs)
        loss_fine = img2mse(render_rays_chunks['rgb_fine'], rgbs)
        loss_coarse = img2mse(render_rays_chunks['rgb_coarse'], rgbs)
        t = int(time.time() - t_start)
        total_t = int(t * iters / (i+1))
        per = int((i+1) / iters * 100)

        print(f'[{per}%] [{t}s >> {total_t}s] [{i+1}/{iters}] ',
                f'[total:loss {loss_total.item():.3f}, psnr {psnr(loss_total).item():.3f}] ',
                f'[fine:loss {loss_fine.item():.3f}, psnr {psnr(loss_fine).item():.3f}] ',
                f'[lr {get_learning_rate(optimizer)}]')
        loss_total.backward()
        optimizer.step()
        if int(i) in [1000, 5000, 20000]:
            scheduler.step()
        


if __name__ == '__main__':
    train_nerf()
