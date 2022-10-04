import numpy as np
import torch
import torch.nn as nn
import json, os
import math, time

from losses import MSELoss
from datasets.blender import BlenderDataset
from models.nerf import PosEmbedding, NeRF
from models.rendering import render_rays
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter


def image_float_to_uint8(img):
    """
    Convert a float image (0.0-1.0) to uint8 (0-255)
    """
    #print(img.shape)
    vmin = np.min(img)
    vmax = np.max(img)
    if vmax - vmin < 1e-10:
        vmax += 1e-10
    img = (img - vmin) / (vmax - vmin)
    img *= 255.0
    return img.astype(np.uint8)


class Trainer():
    def __init__(self, save_dir, gpu, jsonfile, batch_size=2048, check_iter = 10):
        super(Trainer, self).__init__()
        hpampath = os.path.join('jsonfiles', jsonfile)
        with open(hpampath, 'r') as f:
            self.hpams = json.load(f)
        self.device = torch.device('cuda:' + str(gpu))
        self.make_model()
        self.B = batch_size
        self.niter = 0
        self.nepoch = 0
        self.check_iter = check_iter
        self.make_savedir(save_dir)

    def training(self, iter_all):
        start_time = time.time()
        self.loss_calculater = self.get_loss_calculater()
        self.make_dataloader(split='train', batch_size=32, num_workers=8)
        while self.niter < iter_all:
            self.training_single_epoch(start_time)
        self.save_models()
        self.niter += 1

    def training_single_epoch(self, start_time):
        # single epoch here means that it iterates over whole objects
        epoch_start_time = time.time()
        self.set_optimizers()

        epoch_losses = []
        print_every = 1000
        total_iters = len(self.dataloader)
        iter_start_time = time.time()
        for i, sample in enumerate(self.dataloader):

            rays = sample['rays'].to(self.device)   # (batch_size, 8)
            rgbs = sample['rgbs'].to(self.device)   # (batch_size, 3)
            self.optimiter.zero_grad()
            render_rays_chunks = render_rays(self.models, self.embeddings, 
                                             rays, N_samples=64, use_disp=False, 
                                             noise_std=0, N_importance=64, 
                                             chunk=1024, white_back=True)
            loss = self.loss_calculater(render_rays_chunks, rgbs)
            loss.backward()
            self.optimiter.step()
            epoch_losses.append(loss.item())
            
            if i % print_every == 0:
                it = i // 1000
                mean_loss = round(sum(epoch_losses[print_every*(it-1):print_every*it]) / print_every, 4)
                psnr = round(-10*np.log(mean_loss) / np.log(10), 4)
                iter_time = round(time.time() - iter_start_time, 2)
                total_time = round(time.time() - start_time, 2)
                print(f'[TotalT {total_time}s] [Epoch {self.niter+1}/{100}] [Iter {i}/{total_iters}] [Loss {mean_loss}] [PSNR {psnr}] {iter_time}s')
                iter_start_time = time.time()

        mean_loss = round(sum(epoch_losses) / len(epoch_losses), 4)
        psnr = round(-10*np.log(mean_loss) / np.log(10), 4)
        epoch_time = round(time.time() - epoch_start_time, 2)
        total_time = round(time.time() - start_time, 2)
        print(f'TotalT {total_time}s] [Epoch {self.niter+1}/{100} | Loss {mean_loss}|PSNR {psnr}| {epoch_time}s')


        
    def get_loss_calculater(self):
        self.loss_calculater = MSELoss()
        return self.loss_calculater
    
    def set_optimizers(self):
        lr = self.get_learning_rate()
        parameters = []
        for model in self.models:
            parameters += model.parameters()
        self.optimiter = torch.optim.AdamW(parameters, lr=lr)

    def get_learning_rate(self):
        model_lr = self.hpams['lr_schedule']['lr']
        num_model = self.niter // self.hpams['lr_schedule']['interval']
        lr = model_lr * 2**(-num_model)
        return lr

    def make_model(self):
        self.embeddings = [PosEmbedding(3, 10), PosEmbedding(3, 4)]
        nerf_coarse = NeRF(D=8, W=256, in_channels_xyz=63, in_channels_dir=27, skips=[4])
        nerf_fine = NeRF(D=8, W=256, in_channels_xyz=63, in_channels_dir=27, skips=[4])
        # self.models = [nerf_coarse.to(self.device)]
        self.models = [nerf_coarse.to(self.device), nerf_fine.to(self.device)]

    def make_dataloader(self, split='train', batch_size=32, num_workers=1):
        scene_name = self.hpams['data']['scene_name']
        data_path = os.path.join(self.hpams['data']['root_dir'], scene_name)
        dataset = BlenderDataset(data_path, split)
        self.dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
        return self.dataloader

    def make_savedir(self, save_dir):
        self.save_dir = os.path.join('logs', save_dir)
        if not os.path.isdir(self.save_dir):
            os.makedirs(os.path.join(self.save_dir, 'run'))
        self.writer = SummaryWriter(os.path.join(self.save_dir, 'run'))
        hpampath = os.path.join(self.save_dir, 'hpam.json')
        with open(hpampath, 'w') as f:
            json.dump(self.hpams, f, indent=2)

    def save_models(self, iter=None):
        save_dict = {'model_params': self.models.state_dict(),
                     'niter': self.niter,
                     'nepoch': self.nepoch}
        if iter != None:
            torch.save(save_dict, os.path.join(self.save_dir, str(iter)+'.pth'))
        torch.save(save_dict, os.path.join(self.save_dir, 'model.pth'))

    def log_psnr_time(self, loss_per_img, time_spent):
        psnr = -10*np.log(loss_per_img) / np.log(10)
        self.writer.add_scalar('train/psnr', psnr, self.niter)
        self.writer.add_scalar('train/time', time_spent, self.niter)

    def log_regloss(self, loss_reg):
        self.writer.add_scalar('train/loss', loss_reg, global_step=self.niter)

    def log_image(self, generated_img, gt_img):
        H, W = generated_img.shape[:-1]
        ret = torch.zeros(H, W*2, 3)
        ret[:, :W, :] = generated_img
        ret[:, W:, :] = gt_img
        ret = image_float_to_uint8(ret.detach().cpu().numpy())
        self.writer.add_image('train_'+str(self.niter), torch.from_numpy(ret).permute(2,0,1))





if __name__ == '__main__':
    trainer = Trainer('lego', '0', 'blend.json')
    trainer.training(100)