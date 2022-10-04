import torch
import torch.nn as nn
import os

from opt import get_opts
from metrics import psnr
from losses import loss_dict
from utils import *
from torchvision import transforms as T
from datasets import dataset_dict
from torch.utils.data import DataLoader
from collections import defaultdict
from models.nerf import NeRF, PosEmbedding
from models.rendering import render_rays
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


class NeRFSystem(LightningModule):
    def __init__(self, hparams):
        super(NeRFSystem, self).__init__()
        self.save_hyperparameters(hparams)
        
        self.loss = loss_dict[hparams.loss_type]()

        self.embedding_xyz = PosEmbedding(3, 10)
        self.embedding_dir = PosEmbedding(3, 4)
        self.embeddings = [self.embedding_xyz, self.embedding_dir]

        self.nerf_coarse = NeRF(D=8, W=256, in_channels_xyz=63, in_channels_dir=27, skips=[4])
        self.models = [self.nerf_coarse]
        if hparams.N_importance > 0:
            self.nerf_fine = NeRF(D=8, W=256, in_channels_xyz=63, in_channels_dir=27, skips=[4])
            self.models += [self.nerf_fine]
    
    def forward(self, rays):
        """Do batched inference on rays using chunk"""
        B = rays.shape[0]
        results = defaultdict(list)
        chunk = self.hparams.chunk
        for i in range(0, B, chunk):
            rendered_rays_chunks = render_rays(self.models, self.embeddings, 
                                            rays[i:i+chunk], N_samples=64, 
                                            use_disp=False, noise_std=0,
                                             N_importance=64, chunk=1024*32, 
                                             white_back=True)
            for k, v in rendered_rays_chunks.items():
                    results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        return results

    def training_step(self, batch, batch_idx):
        # control training process
        rays = batch['rays']    # (B, 8)
        rgbs = batch['rgbs']    # (B, 3)

        result = self.forward(rays)
        loss = round(self.loss(result, rgbs).item(), 4)
        with torch.no_grad():
            typ = 'fine' if 'rgb_fine' in result else 'coarse'
            psnr_ = round(nn.MSELoss()(result[f'rgb_{typ}'], rgbs).item(), 4)
        self.log('train/loss', loss, prog_bar=True)
        self.log('train/psnr', psnr_, prog_bar=True)
 
    def validation_step(self, batch, batch_idx):
        rays = batch['rays']
        rgbs = batch['rgbs']

 
  
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0002)

    def train_dataloader(self):
        dataset = dataset_dict[self.hparams.dataset_name]
        train_dataset = dataset(self.hparams.root_dir, 'train')
        return DataLoader(train_dataset, batch_size=self.hparams.batch_size,
                          shuffle=True, num_workers=8, pin_memory=True)

    # def val_dataloader(self):
    #     dataset = dataset_dict[self.hparams.dataset_name]
    #     val_dataset = dataset(self.hparams.root_dir, 'val')
    #     return DataLoader(val_dataset, batch_size=1,
    #                       shuffle=False, num_workers=4, pin_memory=True)





def main(hparams):
    model = NeRFSystem(hparams)
    trainer = Trainer(gpus=2)
    trainer.fit(model=model)

 
 



if __name__ == '__main__':
    hparams = get_opts()
    main(hparams)




# class NeRFSystem(LightningModule):
#     def __init__(self, hparams):
#         super(NeRFSystem, self).__init__()
#         self.save_hyperparameters(hparams)

#         # Define loss
#         self.loss = loss_dict[hparams.loss_type]()

#         # Define Embedder 
#         self.embedding_xyz = PosEmbedding(3, 10)
#         self.embedding_dir = PosEmbedding(3, 4)
#         self.embeddings = [self.embedding_xyz, self.embedding_dir]

#         # Define coarse and fine models
#         self.nerf_coarse = NeRF(D=8, W=256, in_channels_xyz=63, in_channels_dir=27, skips=[4])
#         self.models = [self.nerf_coarse]

#         if hparams.N_importance > 0:
#             self.nerf_fine = NeRF(D=8, W=256, in_channels_xyz=63, in_channels_dir=27, skips=[4])
#             self.models += [self.nerf_fine]

#     def get_progress_bar_dict(self):
#         items = super().get_progress_bar_dict()
#         items.pop("v_num", None)
#         return items

#     def forward(self, rays):
#         """Do batched inference on rays using chunk"""
#         B = rays.shape[0]
#         results = defaultdict(list)

#         for i in range(0, B, self.hparams.chunk):
#             render_rays_chunks = render_rays(self.models,
#                                              self.embeddings,
#                                              rays[i: i+self.hparams.chunk],
#                                              self.hparams.N_samples,
#                                              self.hparams.use_disp,
#                                              self.hparams.noise_std,
#                                              self.hparams.N_importance,
#                                              self.hparams.chunk,
#                                              self.train_dataset.white_back)
#             for k, v in render_rays_chunks.items():
#                 results[k] += [v]
        
#         for k, v in results.items():
#             results[k] = torch.cat(v, 0)
#         return results

#     def optimizer_step(self, epoch=None, 
#                     batch_idx=None, 
#                     optimizer=None, 
#                     optimizer_idx=None, 
#                     optimizer_closure=None, 
#                     on_tpu=None, 
#                     using_native_amp=None, 
#                     using_lbfgs=None):
#         if self.hparams.warmup_step > 0 and self.trainer.global_step < self.hparams.warmup_step:
#             lr_scale = min(1., float(self.trainer.global_step + 1) / float(self.hparams.warmup_step))
#             for pg in optimizer.param_groups:
#                 pg['lr'] = lr_scale * self.hparams.lr
#         optimizer.step(closure=optimizer_closure)
#         optimizer.zero_grad()

#     def prepare_data(self):
#         dataset = dataset_dict[self.hparams.dataset_name]
#         kwargs = {'root_dir': self.hparams.root_dir,
#                   'img_wh': self.hparams.img_wh}
#         self.train_dataset = dataset(split='train', **kwargs)
#         if self.hparams.dataset_name == 'blender':
#             self.val_dataset = dataset(split='test', **kwargs)
#         else:
#             self.val_dataset = dataset(split='val', **kwargs)

#     def configure_optimizers(self):
#         self.optimizer = get_optimizer(self.hparams, self.models)
#         scheduler = get_scheduler(self.hparams, self.optimizer)
#         return [self.optimizer], [scheduler]
    
#     def train_dataloader(self):
#         return DataLoader(self.train_dataset, 
#                           shuffle=True,
#                           num_workers=8,
#                           batch_size=self.hparams.batch_size,
#                           pin_memory=True)
    
#     def val_dataloader(self):
#         return DataLoader(self.val_dataset,
#                           shuffle=False,
#                           num_workers=4,
#                           batch_size=1,
#                           pin_memory=True)
    
#     def training_step(self, batch, batch_idx):
#         rays = batch['rays']    # (N_rays, 8)
#         rgbs = batch['rgbs']    # (N_rays, 3)

#         result = self(rays)
#         loss_rgb = self.loss(result, rgbs)

#         with torch.no_grad():
#             typ = 'fine' if 'rgb_fine' in result else 'coarse'
#             psnr_ = psnr(result[f'rgb_{typ}'], rgbs)

#         self.log('train/lr', get_learning_rate(self.optimizer))
#         self.log('train/loss', loss_rgb, prog_bar=True)
#         # print(loss_rgb.item())
#         self.log('train/psnr', psnr_, prog_bar=True)
    

#     def validation_step(self, batch, batch_idx):
#         rays = batch['rays'].squeeze()  # (H*W, 3)
#         rgbs = batch['rgbs'].squeeze()  # (H*W, 3)

#         result = self(rays)
#         log = {}
#         log['val_loss'] = self.loss(result, rgbs)
#         typ = 'fine' if 'rgb_fine' in result else 'coarse'

#         H, W = self.hparams.img_wh
#         img = result[f'rgb_{typ}'].view(H, W, 3).cpu()
#         img = img.permute(2, 0, 1)  # (3, H, W)
#         img_path = os.path.join(f'logs/{self.hparams.exp_name}/video', '%06d.png'%batch_idx)
#         os.makedirs(os.path.dirname(img_path), exist_ok=True)
#         T.ToPILImage()(img).convert('RGB').save(img_path)

#         idx_selected = 0
#         if batch_idx == idx_selected:
#             img = result[f'rgb_{typ}'].view(H, W, 3).permute(2, 0, 1).cpu()
#             img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu()
#             stack = torch.stack([img_gt, img])  # (2, 3, H, W)
#             self.logger.experiment.add_images('val/gt_pred', stack, self.global_step)
#             img_path = os.path.join(f'logs/{self.hparams.exp_name}', f'epoch_{self.current_epoch}.png')
#             T.ToPILImage()(img).convert('RGB').save(img_path)
        
#         log['val_psnr'] = psnr(result[f'rgb_{typ}'], rgbs)
#         torch.cuda.empty_cache()
#         return log

#     def validation_epoch_end(self, outputs):
#         mean_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
#         mean_psnr = torch.stack([x['val_psnr'] for x in outputs]).mean()
#         self.log('val/loss', mean_loss)
#         self.log('val/psnr', mean_psnr, prog_bar=True)

  
# def main(hparams):
#     system = NeRFSystem(hparams)
#     checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(f'logs/{hparams.exp_name}/ckpts',
#                                                                 '{epoch:d}'),
#                                           monitor='val/psnr',
#                                           mode='max',
#                                           save_top_k=5)

#     logger = TensorBoardLogger(save_dir="logs", name=hparams.exp_name)

#     trainer = Trainer(max_epochs=hparams.num_epochs,
#                       checkpoint_callback=checkpoint_callback,
#                       logger=logger,
#                       gpus=hparams.num_gpus,
#                       strategy='ddp' if hparams.num_gpus>1 else None,
#                       benchmark=True)

#     trainer.fit(system, ckpt_path=hparams.ckpt_path)


# if __name__ == '__main__':
#     hparams = get_opts()
#     main(hparams)



        
        



                
