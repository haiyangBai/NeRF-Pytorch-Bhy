import torch
from einops import rearrange, reduce, repeat

__all__ = ['rander_rays']


def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):
    """
    Sample @N_importance samples from @bins with distribution defined by @weights.
    Inputs:
        bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
        weights: (N_rays, N_samples_)
        N_importance: the number of samples to draw from the distribution
        det: deterministic or not
        eps: a small number to prevent division by zero
    Outputs:
        samples: the sampled samples
    """
    N_rays, N_samples_ = weights.shape
    weights = weights + eps # prevent division by zero (don't do inplace op!)
    pdf = weights / reduce(weights, 'n1 n2 -> n1 1', 'sum') # (N_rays, N_samples_)
    cdf = torch.cumsum(pdf, -1) # (N_rays, N_samples), cumulative distribution function
    cdf = torch.cat([torch.zeros_like(cdf[: ,:1]), cdf], -1)  # (N_rays, N_samples_+1) 
                                                               # padded to 0~1 inclusive

    if det:
        u = torch.linspace(0, 1, N_importance, device=bins.device)
        u = u.expand(N_rays, N_importance)
    else:
        u = torch.rand(N_rays, N_importance, device=bins.device)
    u = u.contiguous()

    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp_min(inds-1, 0)
    above = torch.clamp_max(inds, N_samples_)

    inds_sampled = rearrange(torch.stack([below, above], -1), 'n1 n2 c -> n1 (n2 c)', c=2)
    cdf_g = rearrange(torch.gather(cdf, 1, inds_sampled), 'n1 (n2 c) -> n1 n2 c', c=2)
    bins_g = rearrange(torch.gather(bins, 1, inds_sampled), 'n1 (n2 c) -> n1 n2 c', c=2)

    denom = cdf_g[...,1]-cdf_g[...,0]
    denom[denom<eps] = 1 # denom equals 0 means a bin has weight 0, in which case it will not be sampled
                         # anyway, therefore any value for it is fine (set to 1 here)

    samples = bins_g[...,0] + (u-cdf_g[...,0])/denom * (bins_g[...,1]-bins_g[...,0])
    return samples


def render_rays(models,
                embeddings,
                rays,
                N_samples,
                use_disp=False,
                noise_std=1,
                N_importance=0,
                chunk=1024*32,
                white_back=False,
                test_time=False):
    """
    Render rays by computing the output of @model applied on @rays.

    Inputs:
        models: lisr of NeRF models (coarse and fine) defined in nerf.py
        embedding: list of embedding models of orgin and directions defined in nerf.py
        rays: (N_rays, 3+3+2), rays origins, directions and near, far depth bounds
        N_sample: number of coarse samples per ray
        use_disp: whether to sample in disparity space (inverse depth)
        perturb: factor to perturb the sampling position on the ray (for only coarse model)
        noise_std: factor to perturb the model's prediection of sigma
        N_importance: number of fine samples per ray
        chunk: chunk size in bached inference
        white_back: whether the background is white (dataset dependent)
        test_time: whether it is test (inference only) or not. If not, it will 
                    not do inference on coarse rgb to save time 
    Outputs:
        result: dict containing fine rgb and depth maps for coarse and fine models
    """
    def inference(model, embeddings, xyz_, dirs, z_vals, weight_only=False):
        """
        Helper function that perform model inference.

        Inputs:
            model: NeRF model (coarse or fine)
            embeddings: xyz and dir embedders
            xyz_: (N_rays, N_samples_, 3) sampled positions
                N_samples_ is the number of sampled points in each ray
                            = N_samples for coarse model.
                            = N_sample + N_importance for fine model
            dirs: (N_rays, 3) rays directions
            z_vals: (N_rays, N_samples_) depths of sampled positions
            weight_only: do inference on sigma only or not

        Outputs:
            if weight_only:
                (N_rays, N_samples_): weights of each sample
            else:
                rgb_final: (N_rays, 3) the final rgb image
                depth_final: (N_rays) depth map
                weights: (N_rays, N_samples_): weights of each sample
        """
        N_rays = xyz_.shape[0]
        N_samples_ = xyz_.shape[1]
        xyz_ = xyz_.view(-1, 3)
        dir_ = dirs.unsqueeze(1).expand(-1, N_samples_, -1).reshape(-1, 3)

        B = xyz_.shape[0]

        # Mapping xyz and dir to (R G B sigma)
        out_chunk = []

        for i in range(0, B, chunk):
            xyz_embedded = embeddings[0](xyz_[i: i+chunk])  # (chunk, 63)
            dir_embedded = embeddings[1](dir_[i: i+chunk])  # (chunk, 27)
            out_chunk += [model(xyz_embedded, dir_embedded, sigma_only=weight_only)]
        out = torch.cat(out_chunk, 0)
        out = out.reshape(N_rays, N_samples_, -1)   # (N_rays, N_samples_, 4)

        # Get rgb and sigma to render
        rgbs = out[..., :3] # (N_rays, N_samples_, 3)
        sigmas = out[..., 3] # (N_rays, N_samples_)

        # Compute weights
        ## deltas
        deltas = z_vals[:, 1:] - z_vals[:, :-1] # (N_rays, N_samples_-1)
        delta_inf = 1e10 * torch.ones_like(deltas[:, :1])
        deltas = torch.cat([deltas, delta_inf], -1) # (N_rays, N_samples_)
        # Multiply each distance by the norm of its corresponding direction ray
        # to convert to real world distance (accounts for non-unit directions).
        deltas = deltas * torch.norm(dirs.unsqueeze(1), dim=-1)

        noise = torch.randn(sigmas.shape, device=sigmas.device) * 0.0

        ## compute alpha by formula (3)
        alphas = 1 - torch.exp(-deltas * torch.nn.Softplus()(sigmas+noise))  # (N_rays, N_samples_)
        alphas_shifted = torch.cat([torch.ones_like(alphas[:, :1]), 1-alphas+1e-10], -1)
        weights = alphas * torch.cumprod(alphas_shifted, -1)[:, :-1]    # (N_rays, N_samples_)
        weights_sum = weights.sum(1)    # (N_rays)

        if weight_only:
            return weights
        
        rgb_final = torch.sum(weights.unsqueeze(-1)*rgbs, -2)   # (N_rays, 3)
        depth_final = torch.sum(weights*z_vals, -1)             # (N_rays)
        
        if white_back:
            rgb_final = rgb_final + 1 - weights_sum.unsqueeze(-1)
        
        return rgb_final, depth_final, weights, rgbs, sigmas


    model_coarse = models[0]

    is_training = model_coarse.training

    N_rays = rays.shape[0]
    rays_o, rays_d = rays[:, :3], rays[:, 3:6]  # both (N_rays, 3)
    near, far = rays[:, 6:7], rays[:, 7:8]      # both (N_rays, 1)

    # Sample depth points
    z_steps = torch.linspace(0, 1, N_samples, device=rays.device)   # (N_samples)
    if not use_disp:
        z_vals = near * (1 - z_steps) + far * z_steps
    else:
        z_vals = 1 / (1/near * (1 - z_steps) + 1/far * z_steps)
    z_vals = z_vals.expand(N_rays, N_samples)

    if is_training:
        z_vals = z_vals + torch.empty_like(z_vals).normal_(0.0, 0.002) * (far-near)
    
    # Obtian coarse samples points origin and direction
    # (N_rays, N_samples, 3)
    xyz_coarse_sampled = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(2)

    rgb_coarse, depth_coarse, weight_coarse, colors_coarse, sigmas_coarse = \
        inference(model_coarse, embeddings, xyz_coarse_sampled, rays_d,
                     z_vals, weight_only=False)

    result = {'rgb_coarse': rgb_coarse, 
              'depth_coarse': depth_coarse, 
              'opactiy_coarse': weight_coarse.sum(1),
              'z_vals_coarse': z_vals,
              'sigam_coarse': sigmas_coarse,
              'weight_coarse': weight_coarse
              }
    
    if N_importance > 0:
        z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:]) # (N_rays, N_samples-1) interval mid points
        z_vals_ = sample_pdf(z_vals_mid, result['weight_coarse'][:, 1:-1].detach(), N_importance)
        z_vals = torch.sort(torch.cat([z_vals, z_vals_], -1), -1)[0]

        xyz_fine_sampled = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(2)

        model_fine = models[1]
        rgb_fine, depth_fine, weight_fine, colors_fine, sigmas_fine = \
        inference(model_fine, embeddings, xyz_fine_sampled, rays_d, 
                        z_vals, weight_only=False)

        result['rgb_fine'] = rgb_fine
        result['depth_fine'] = depth_fine
    
    return result



