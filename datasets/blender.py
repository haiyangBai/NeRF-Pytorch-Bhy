import torch
import numpy as np
import os
import json

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from .ray_utils import *


trans_t = lambda t: torch.Tensor([
    [1, 0, 0, 0], 
    [0, 1, 0, 0],
    [0, 0, 1, t],
    [0, 0, 0, 1]]).float()

rot_phi = lambda phi: torch.Tensor([
    [1, 0, 0, 0], 
    [0, np.cos(phi), -np.sin(phi), 0],
    [0, np.sin(phi),  np.cos(phi), 0],
    [0, 0, 0, 1]]).float()

rot_theta = lambda th: torch.Tensor([
    [np.cos(th), 0, -np.sin(th), 0],
    [0, 1, 0, 0], 
    [np.sin(th), 0,  np.cos(th), 0],
    [0, 0, 0, 1]]).float()

def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180. * np.pi) @ c2w
    c2w = rot_theta(theta/180 * np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w


class BlenderDataset(Dataset):
    def __init__(self, root_dir, split='train', img_wh=(800, 800)):
        super(BlenderDataset, self).__init__()

        self.root_dir = root_dir
        self.split = split
        self.img_wh = img_wh
        self.white_back = True
        self.read_meta()

    def read_meta(self):
        with open(os.path.join(self.root_dir, f'transforms_{self.split}.json'), 'r') as f:
            self.meta = json.load(f)
        
        w, h = self.img_wh
        # modify focal length to match size self.img_wh
        self.focal = 0.5 * 800 * (w/800) /np.tan(0.5 * self.meta['camera_angle_x'])

        self.near = 2.0
        self.far = 6.0
        self.bounds = np.array([self.near, self.far])

        self.directions = get_ray_directions(h, w, self.focal) # (h, w, 3)
        self.directions = self.directions / torch.norm(self.directions, dim=-1, keepdim=True)

        # get rays and rgb data
        if self.split == 'train':
            self.img_paths = []
            self.poses = []
            self.all_rays = []
            self.all_rgbs = []
            for frame in self.meta['frames']:
                # get poses
                pose = np.array(frame['transform_matrix'])[:3, :4]
                self.poses.append(pose)

                # get rgbs
                img_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")
                self.img_paths.append(img_path)
                img = Image.open(img_path)
                if self.img_wh[0] != img.size[1]:
                    img = img.resize(self.img_wh, Image.Resampling.LANCZOS)
                img = self.transform(img)               # (4, H, W)
                img = img.view(4, -1).permute(1, 0)     # (H*W, 4)
                img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])
                self.all_rgbs.append(img)

                # get rays_o and rays_d
                c2w = torch.FloatTensor(pose)
                rays_o, rays_d = get_rays(self.directions, c2w) # (H*W, 3)
                rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
                self.all_rays.append(torch.cat([rays_o, rays_d,
                                    self.near*torch.ones_like(rays_o[:, :1]),
                                    self.far*torch.ones_like(rays_o[:, :1])], 1))
            
            self.all_rays = torch.cat(self.all_rays, 0) # (len(img_paths)*h*w, 8)
            self.all_rgbs = torch.cat(self.all_rgbs, 0) # (len(img_paths)*h*w, 3)

        elif self.split == 'test':
            # Select 1/10 for fast testing during training
            self.meta['frames'] = self.meta['frames'][::10]
        elif self.split == 'val':
            self.pose_vis = torch.stack([pose_spherical(angle, -30, 4.0)
                        for angle in np.linspace(-180, 180, 1000+1)[:-1]], 0)

    def transform(self, img):
        return T.ToTensor()(img)

    def __len__(self):
        if self.split == 'train':
            return len(self.all_rays)
        elif self.split == 'test':
            return len(self.meta['frames'])
        elif self.split == 'val':
            return self.pose_vis.shape[0]

    def __getitem__(self, idx):
        if self.split == 'train':
            sample = {'rays': self.all_rays[idx],
                      'rgbs': self.all_rgbs[idx]}
        
        elif self.split == 'test':
            frame = self.meta['frames'][idx]
            c2w = torch.FloatTensor(frame['transform_matrix'])[:3, :4]

            img = Image.open(os.path.join(self.root_dir, f"{frame['file_path']}.png"))
            if self.img_wh[0] != img.size[0]:
                img = img.resize(self.img_wh, Image.Resampling.LANCZOS)
            img = self.transform(img)
            valid_mask = (img[-1] > 0).flatten()
            img = img.view(4, -1).permute(1, 0) # (H*W, 4) RGBA
            img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # # (H*W, 3)

            rays_o, rays_d = get_rays(self.directions, c2w) # (H*W, 3)
            rays = torch.cat([rays_o , rays_d,
                            self.near*torch.ones_like(rays_o[:, :1]),
                            self.far*torch.ones_like(rays_o[:, :1])], 1)    # (H*W, 8)

            sample = {'rays': rays, # (H*W, 8)
                      'rgbs': img,  # (H*W, 8)
                      'c2w': c2w,   # (3, 4)
                      'valid_mask': valid_mask}
        
        elif self.split == 'val':
            c2w = torch.FloatTensor(self.pose_vis[idx])[:3, :4]
            rays_o, rays_d = get_rays(self.directions, c2w)
            rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
            rays = torch.cat([rays_o, rays_d,
                            self.near*torch.ones_like(rays_o[:, :1]),
                            self.far*torch.ones_like(rays_o[:, :1])], 1)

            sample = {'rays': rays}
        
        return sample


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    root_dir = '/home/baihy/datasets/nerf_synthetic/nerf_synthetic/lego'
    dataset = BlenderDataset(root_dir)
    dataloader = DataLoader(dataset, 128)

    i = 0
    print(len(dataloader))
    for i, sample in enumerate(dataloader):
        print(sample.keys(), sample['rays'].shape, sample['rgbs'].shape)