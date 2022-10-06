import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import cv2
from tensorboardX import SummaryWriter
from collections import defaultdict
from losses import MSELoss
from tqdm import tqdm
from metrics import psnr, mse
from datasets.blender import BlenderDataset
from models.nerf import NeRF, PosEmbedding
from models.rendering import render_rays
from models.utils import get_parameters, init_weights
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
    nerf_coarse.apply(init_weights)
    nerf_fine.apply(init_weights)
    models = [nerf_coarse, nerf_fine]

    # Loss and Optimizer
    cal_loss = MSELoss()
    optimizer = torch.optim.Adam(get_parameters(models), lr=0.0002, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=16, eta_min=1e-8)

    # Dataset Loader
    chunk = 1024 *32
    epochs = 5
    batch_size = 128

    root_dir = '/home/baihy/datasets/nerf_synthetic/nerf_synthetic/chair'
    train_dataset = BlenderDataset(root_dir, 'train', data_skip=5)
    train_dataloader = DataLoader(train_dataset, shuffle=True,num_workers=8,
                                    batch_size=batch_size, pin_memory=True)
    test_dataset = BlenderDataset(root_dir, 'test')
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=1)

    # logs
    writer = SummaryWriter('log')
    
    
    # Training
    iters = len(train_dataloader)
    t_start = time.time()
    for epoch in range(epochs):
        for it, sample in enumerate(train_dataloader):
            rays, rgbs = sample['rays'], sample['rgbs']
            B = rays.shape[0]

            results = defaultdict(list)
            for i in range(0, B, chunk):
                rendered_rays_chunks = render_rays(models, embeddings, 
                                                rays[i:i+chunk], N_samples=64, 
                                                use_disp=False, noise_std=0,
                                                N_importance=128, chunk=1024*32, 
                                                white_back=True)
                for k, v in rendered_rays_chunks.items():
                        results[k] += [v]

            for k, v in results.items():
                results[k] = torch.cat(v, 0)

            optimizer.zero_grad()
            loss_total = cal_loss(results, rgbs)
            loss_fine = img2mse(results['rgb_fine'], rgbs)
            loss_coarse = img2mse(results['rgb_coarse'], rgbs)
            t = int(time.time() - t_start)
            total_t = int(t * iters / (it+1))
            per = int((it+1) / iters * 100)

            writer.add_scalar('train/loss_total', loss_total.item(), epoch*iters+it)
            writer.add_scalar('train/loss_coarse', loss_coarse.item(), epoch*iters+it)
            writer.add_scalar('train/loss_fine', loss_fine.item(), epoch*iters+it)
            writer.add_scalar('train/psnr', psnr(loss_fine).item(), epoch*iters+it)

            print(f'[{per}%] [{t}s >> {total_t}s] [Epoch {epoch+1}/{epochs}] [{it+1}/{iters}]',
                    f'[total:loss {loss_total.item():.3f}, psnr {psnr(loss_total).item():.3f}]',
                    f'[fine:loss {loss_fine.item():.3f}, psnr {psnr(loss_fine).item():.3f}]',
                    f'[lr {get_learning_rate(optimizer)}]')
            loss_total.backward()
            optimizer.step()
            if int(it) in [1000, 5000, 20000, 50000]:
                scheduler.step()
            
            # Validation
            if False: # (it+1) % 10 == 0:
                sample = next(iter(test_dataloader))
                rays, rgbs = sample['rays'][0], sample['rgbs'][0]
                print('rays shape:', rays.shape, 'rgbs shape:', rgbs.shape)
                # rays, rgbs = rays[0].to(device), rgbs[0].to(device)
                
                BB = rays.shape[0]

                results = defaultdict(list)
                for j in tqdm(range(0, BB, batch_size)):
                    rendered_rays_chunks = render_rays(models, embeddings, 
                                                    rays[j:j+batch_size], N_samples=32, 
                                                    use_disp=False, noise_std=0,
                                                    N_importance=32, chunk=1024*32, 
                                                    white_back=True)
                    for k, v in rendered_rays_chunks.items():
                            results[k] += [v]

                for k, v in results.items():
                    results[k] = torch.cat(v, 0)

                val_loss = img2mse(results['rgb_fine'], rgbs)
                val_psnr = psnr(val_loss)
                writer.add_scalar('val/loss', psnr(val_loss).item(), epoch*iters+it)
                writer.add_scalar('val/psnr', psnr(val_psnr).item(), epoch*iters+it)
                print(f'[{epoch+1}/{epochs}] [{it}/{iters}] [val_loss: {val_loss.item():.3f}] [val_psnr: {psnr.item():.3f}]')
                
                val_img = to8b((results['rgbs'].view(800, 800, 3)))
                gt_img = to8b(rgbs.view(800, 800, 3))
                res_img = np.zeros((800, 1600, 3), dtype=np.uint8)
                res_img[800, :800, 3] = val_img
                res_img[800, 800:, 3] = gt_img

                cv2.imwrite(f'log/img_{epoch*iters+it}.png', res_img)

        

if __name__ == '__main__':
    train_nerf()
