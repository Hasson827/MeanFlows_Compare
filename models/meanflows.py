import torch
from einops import rearrange
from functools import partial
import numpy as np

class Normalizer:
    def __init__(self, mode='minmax', mean=None, std=None):
        assert mode in ['minmax', 'mean_std'], "mode must be 'minmax' or 'mean_std'"
        self.mode = mode

        if mode == 'mean_std':
            self.mean = torch.tensor(mean).view(-1, 1, 1)
            self.std = torch.tensor(std).view(-1, 1, 1)

    @classmethod
    def from_list(cls, config):
        mode, mean, std = config
        return cls(mode, mean, std)

    def norm(self, x):
        if self.mode == 'minmax':
            return x * 2 - 1
        elif self.mode == 'mean_std':
            return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def unnorm(self, x):
        if self.mode == 'minmax':
            x = x.clip(-1, 1)
            return (x + 1) * 0.5
        elif self.mode == 'mean_std':
            return x * self.std.to(x.device) + self.mean.to(x.device)

def stopgrad(x):
    return x.detach()

def adaptive_l2_loss(error, gamma=0.25, c=1e-2):
    delta_sq = torch.mean(error ** 2, dim=(1, 2, 3), keepdim=False)
    p = 1.0 - gamma
    w = 1.0 / (delta_sq + c).pow(p)
    result = (stopgrad(w) * delta_sq).mean()
    return result


class MeanFlow:
    def __init__(self, channels, image_size, num_classes, normalizer, flow_ratio, time_dist, cfg_ratio, omega, kappa):
        self.channels = channels
        self.image_size = image_size
        self.num_classes = num_classes
        self.use_cond = (num_classes > 0)
        self.normer = Normalizer.from_list(normalizer)
        self.flow_ratio = flow_ratio
        self.time_dist = time_dist
        self.cfg_ratio = cfg_ratio
        self.w = omega
        self.k = kappa
    
    def sample_t_r(self, batch_size, device):
        if self.time_dist[0] == 'uniform':
            samples = np.random.rand(batch_size, 2).astype(np.float32)
        elif self.time_dist[0] == 'lognorm':
            mu, sigma = self.time_dist[-2], self.time_dist[-1]
            normal_samples = np.random.randn(batch_size, 2).astype(np.float32) * sigma + mu
            samples = 1 / (1 + np.exp(-normal_samples))
        
        t_np = np.maximum(samples[:, 0], samples[:, 1])
        r_np = np.minimum(samples[:, 0], samples[:, 1])
        
        num_selected = int(self.flow_ratio * batch_size)
        indices = np.random.permutation(batch_size)[:num_selected]
        r_np[indices] = t_np[indices]

        t = torch.tensor(t_np, device=device)
        r = torch.tensor(r_np, device=device)
        return t, r
    
    @torch.no_grad()
    def sample(self, model, device, use_cond, n=None):
        if n is None:
            if self.use_cond and use_cond: # 有条件训练&有条件采样
                y = torch.arange(self.num_classes, device=device)
                z = torch.randn(self.num_classes, self.channels, self.image_size, self.image_size, device=device)
                t = torch.ones(self.num_classes, device=device)
                r = torch.zeros(self.num_classes, device=device)
            elif self.use_cond and not use_cond: # 有条件训练&无条件采样
                y = torch.full((10,), self.num_classes, device=device, dtype=torch.long)
                z = torch.randn(10, self.channels, self.image_size, self.image_size, device=device)
                t = torch.ones(10, device=device)
                r = torch.zeros(10, device=device)
            elif self.use_cond is False: # 无条件训练&无条件采样
                y = torch.zeros(10, device=device)
                z = torch.randn(10, self.channels, self.image_size, self.image_size, device=device)
                t = torch.ones(10, device=device)
                r = torch.zeros(10, device=device)
                
            t_ = rearrange(t, 'B -> B 1 1 1').detach().clone()
            r_ = rearrange(r, 'B -> B 1 1 1').detach().clone()
            
            u = model(z, t, r, y)
            z = z - (t_ - r_) * u
            x = self.normer.unnorm(z)
            return x
        else:
            imgs = []
            batch_size = 100
            while len(imgs) < n:
                cur_bs = min(batch_size, n - len(imgs))
                if self.use_cond and use_cond:
                    y = torch.randint(0, self.num_classes, (cur_bs,), device=device)
                elif self.use_cond and not use_cond:
                    y = torch.full((cur_bs,), self.num_classes, device=device, dtype=torch.long)
                else:
                    y = torch.zeros(cur_bs, device=device, dtype=torch.long)
                z = torch.randn(cur_bs, self.channels, self.image_size, self.image_size, device=device)
                t = torch.ones(cur_bs, device=device)
                r = torch.zeros(cur_bs, device=device)
                t_ = rearrange(t, 'B -> B 1 1 1').detach().clone()
                r_ = rearrange(r, 'B -> B 1 1 1').detach().clone()
                u = model(z, t, r, y)
                z = z - (t_ - r_) * u
                x = self.normer.unnorm(z)
                imgs.append(x.cpu())
            return torch.cat(imgs, dim=0)[:n]
    
    def loss(self, model, x, y):
        model = model.module if hasattr(model, 'module') else model
        batch_size = x.shape[0]
        device = x.device
        
        t, r = self.sample_t_r(batch_size, device)
        t_ = rearrange(t, 'B -> B 1 1 1').detach().clone()
        r_ = rearrange(r, 'B -> B 1 1 1').detach().clone()
        
        e = torch.randn_like(x)
        x = self.normer.norm(x)
        z = ((1 - t_) * x + t_ * e)
        v = e - x
        
        if y is None:
            y = torch.zeros(batch_size, device=device, dtype=torch.long)
            v_cfg = v
        else:
            uncond = torch.ones_like(y) * self.num_classes
            cfg_mask = torch.rand_like(y.float()) < self.cfg_ratio
            y = torch.where(cfg_mask, uncond, y)
            with torch.no_grad():
                vt_uncond = model(z, t, t, uncond)
                vt_cond = model(z, t, t, y)
            v_cfg = self.w * v + self.k * vt_cond + (1 - self.w - self.k) * vt_uncond
            cfg_mask = rearrange(cfg_mask, 'B -> B 1 1 1').bool()
            v_cfg = torch.where(cfg_mask, v, v_cfg)
        
        model_partial = partial(model, y=y)
        jvp_args = (
            lambda z, t, r: model_partial(z, t, r),
            (z, t, r), 
            (v_cfg, torch.ones_like(t), torch.zeros_like(r))
        )
        
        try:
            u, dudt = torch.func.jvp(*jvp_args)
        except (ImportError, AttributeError):
            u, dudt = torch.autograd.functional.jvp(*jvp_args, create_graph=True)
        
        u_tgt = v_cfg - (t_ - r_) * dudt
        error = u - stopgrad(u_tgt)
        loss = adaptive_l2_loss(error)
        return loss

class AdditiveMeanFlow(MeanFlow):
    def __init__(self, channels, image_size, num_classes, normalizer, flow_ratio, time_dist, cfg_ratio, omega, kappa):
        super().__init__(channels, image_size, num_classes, normalizer, flow_ratio, time_dist, cfg_ratio, omega, kappa)

    def sample_t_r(self, batch_size, device):
        if self.time_dist[0] == 'uniform':
            samples = np.random.rand(batch_size, 2).astype(np.float32)
        elif self.time_dist[0] == 'lognorm':
            mu, sigma = self.time_dist[-2], self.time_dist[-1]
            normal_samples = np.random.randn(batch_size, 2).astype(np.float32) * sigma + mu
            samples = 1 / (1 + np.exp(-normal_samples))
        
        t_np = np.maximum(samples[:, 0], samples[:, 1])
        r_np = np.minimum(samples[:, 0], samples[:, 1])

        t = torch.tensor(t_np, device=device)
        r = torch.tensor(r_np, device=device)
        return t, r
    
    def loss(self, model, x, y, alpha):
        jvp_loss = super().loss(model, x, y)
        model = model.module if hasattr(model, 'module') else model
        batch_size = x.shape[0]
        device = x.device
        
        t, r = self.sample_t_r(batch_size, device)
        mid = (t + r) / 2
        t_ = rearrange(t, 'B -> B 1 1 1').detach().clone()
        r_ = rearrange(r, 'B -> B 1 1 1').detach().clone()
        mid_ = rearrange(mid, 'B -> B 1 1 1').detach().clone()
        
        e = torch.randn_like(x)
        x = self.normer.norm(x)
        z = (1 - t_) * x + t_ * e
        v = e - x

        model_partial = partial(model, y=y)
        u = model_partial(z, t, r)
        u1 = model_partial(z, t, mid)
        z_mid = z - (t_ - mid_) * u1
        u2 = model_partial(z_mid, mid, r)
        
        u_tgt = (u1 + u2) / 2
        error = u - stopgrad(u_tgt)
        consistency_loss = adaptive_l2_loss(error)
        total_loss = (1-alpha)*jvp_loss + alpha*consistency_loss
        return total_loss