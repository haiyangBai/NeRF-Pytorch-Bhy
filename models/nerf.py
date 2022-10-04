import torch
from torch import nn


class PosEmbedding(nn.Module):
    def __init__(self, in_channels, N_freqs, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super(PosEmbedding, self).__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = in_channels * (len(self.funcs) * N_freqs + 1)

        if logscale:
            self.freq_bands = 2 ** torch.linspace(0, N_freqs-1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2**(N_freqs-1), N_freqs)

    def forward(self, x):
        """
        Inputs:
            x: (B, self.in_channels)
        Outputs:
            out: (B, self.out_channels)
        """
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq * x)]
        return torch.cat(out, dim=-1)


class NeRF(nn.Module):
    def __init__(self,
                 D=8,
                 W=256,
                 in_channels_xyz=63,
                 in_channels_dir=27,
                 skips=[4]):
        """
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        in_channels_xyz: number of input channels for xyz
        in_channels_dir: number of input channels fir direction
        skips: add skip connection in the D_th layer 
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.in_channels_xyz = in_channels_xyz
        self.in_channels_dir = in_channels_dir
        self.skips = skips

        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channels_xyz, W)
            elif i in skips:
                layer = nn.Linear(W + in_channels_xyz, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f'xyz_encoding_{i+1}', layer)

        self.xyz_encoding = nn.Sequential(nn.Linear(W, W), nn.ReLU(True))
        self.dir_encoding = nn.Sequential(nn.Linear(W+in_channels_dir, W//2),
                                          nn.ReLU(True))

        self.sigma = nn.Sequential(nn.Linear(W, 1), nn.Softplus())
        self.rgb = nn.Sequential(nn.Linear(W//2, 3), nn.Sigmoid())
    
    def forward(self, xyz, dir, sigma_only=False):
        """
        Encoder input (xyz + dir) to rgb+sigma (not ready to render yet),
        For rendering this ray, please see rendering.py

        Inputs:
            x: (B, self.in_channels_xyz(+self.in_channels_dir))
                the embedded vector of postion and direction
            sigma_only: whether to infer sigma only, if True,
                        x is of shape (B, self.in_channels_xyz)
        Outputs:
            if sigma_only:
                sigma: (B, 1)
            else:
                out: (B, 4)
        """
        input_xyz = xyz

        xyz_ = input_xyz
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], -1)
            xyz_ = getattr(self, f'xyz_encoding_{i+1}')(xyz_)

        sigma = self.sigma(xyz_)
        if sigma_only:
            return sigma
        xyz_encoding = self.xyz_encoding(xyz_)
        dir_encoding = self.dir_encoding(torch.cat([xyz_encoding, dir], -1))
        rgb = self.rgb(dir_encoding)

        out = torch.cat([rgb, sigma], -1)
        return out
        
