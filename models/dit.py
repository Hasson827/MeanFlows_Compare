import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Mlp, Attention
from einops import repeat, pack, unpack

def modulate(x, scale, shift):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

#! 时间步嵌入模块
class TimestepEmbedder(nn.Module):
    def __init__(self, dim, nfreq=256):
        super().__init__()
        self.nfreq = nfreq
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(nfreq, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )

    def timestep_embedding(self, t, max_period=1e4):
        half_dim = self.nfreq // 2
        device = t.device
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(0, half_dim, dtype=torch.float32, device=device)
            / half_dim
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return emb

    def forward(self, t):
        t = t * 1000
        t_embed_raw = self.timestep_embedding(t)
        t_emb = self.mlp(t_embed_raw)
        return t_emb

#! 标签嵌入模块
class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, dim):
        super().__init__()
        self.embedding = nn.Embedding(num_classes+1, dim)
        self.num_classes = num_classes
    
    def forward(self, labels):
        embeddings = self.embedding(labels)
        return embeddings

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.g

#! DiT模块
class DiTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.attn = Attention(dim, num_heads, qkv_bias=True, qk_norm=True, norm_layer=RMSNorm)
        self.attn.fused_attn = False
        self.mlp = Mlp(
            in_features = dim, 
            hidden_features = int(dim * mlp_ratio), 
            act_layer = lambda: nn.GELU(approximate="tanh"),
            drop = dropout
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), 
            nn.Linear(dim, 6 * dim)
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x, c):
        (shift_msa, scale_msa, gate_msa, 
         shift_mlp, scale_mlp, gate_mlp) = self.adaLN_modulation(c).chunk(6, dim=-1)
        gate_msa = torch.tanh(gate_msa)
        gate_mlp = torch.tanh(gate_mlp)
        x = x + self.dropout(
            gate_msa.unsqueeze(1) * self.attn(
                modulate(self.norm1(x), scale_msa, shift_msa)
            )
        )
        x = x + self.dropout(
            gate_mlp.unsqueeze(1) * self.mlp(
                modulate(self.norm2(x), scale_mlp, shift_mlp)
            )
        )
        return x

#! 最终输出层,将特征映射到输出维度
class FinalLayer(nn.Module):
    def __init__(self, dim, patch_size, out_dim, dropout):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.linear = nn.Linear(dim, patch_size * patch_size * out_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 2 * dim)
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm(x), scale, shift)
        x = self.dropout(x)
        x = self.linear(x)
        return x

class MFDiT(nn.Module):
    def __init__(self, input_size, patch_size, in_channels, dim, depth, num_heads, mlp_ratio, num_classes, dropout):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.num_classes = num_classes
        
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, dim)
        self.t_embedder = TimestepEmbedder(dim)
        self.r_embedder = TimestepEmbedder(dim)
        self.y_embedder = LabelEmbedder(num_classes, dim)
        
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, dim), requires_grad=True)

        self.blocks = nn.ModuleList([DiTBlock(dim, num_heads, mlp_ratio, dropout) for _ in range(depth)])
        self.final_layer = FinalLayer(dim, patch_size, in_channels, dropout)
        
        self.initialize_weights()
    
    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], 
            int(self.x_embedder.num_patches**0.5)
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        w = self.x_embedder.proj.weight.data
        nn.init.normal_(w, std=0.02)
        nn.init.constant_(self.x_embedder.proj.bias, 0)
        
        nn.init.normal_(self.y_embedder.embedding.weight, std=0.02)
        
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.r_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.r_embedder.mlp[2].weight, std=0.02)
        
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
    
    def unpatchify(self, x):
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(-1, h, w, p, p, c)
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs
    
    def forward(self, x, t, r, y):
        x = self.x_embedder(x) + self.pos_embed
        t = self.t_embedder(t)
        r = self.r_embedder(r)
        y = self.y_embedder(y)
        c = t + r + y
        for block in self.blocks:
            x = block(x, c)
        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        return x

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

if __name__ == "__main__":
    # Example usage
    model = MFDiT(input_size=32, patch_size=4, in_channels=3, dim=128, depth=12, num_heads=8, mlp_ratio=4.0, num_classes=10, dropout=0.1)
    x = torch.randn(64, 3, 32, 32)  # Batch of images
    t = torch.ones(64)  # Time steps
    r = torch.zeros(64)  # Random values
    y = torch.randint(0, 10, (64,))  # Class labels
    out = model(x, t, r, y)
    print(out.shape)  # Should output the shape of the reconstructed images
    
    model = MFDiT(input_size=32, patch_size=4, in_channels=3, dim=128, depth=12, num_heads=8, mlp_ratio=4.0, num_classes=0, dropout=0.1)
    y = torch.zeros(64).long()  # No class labels for unconditional training
    out = model(x, t, r, y)
    print(out.shape)  # Should output the shape of the reconstructed images without class labels
    
    number_of_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / (1024*1024)
    print(f"Number of trainable parameters: {number_of_params:.2f}M")