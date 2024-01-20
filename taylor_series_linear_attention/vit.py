from math import sqrt

import torch
from torch import nn, einsum
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce

from taylor_series_linear_attention.attention import (
    TaylorSeriesLinearAttn,
    ChannelFirstTaylorSeriesLinearAttn
)

# sinusoidal pos

def posemb_sincos_2d(
    h, w,
    dim,
    temperature: int = 10000,
    dtype = torch.float32
):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing = "ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"

    dim //= 4
    omega = torch.arange(dim) / (dim - 1)
    omega = temperature ** -omega

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim = 1)
    return pe.type(dtype)

# feed forward related classes

def DepthWiseConv2d(
    self,
    dim_in,
    dim_out,
    kernel_size,
    padding,
    stride = 1,
    bias = True
):
    return nn.Sequential(
        nn.Conv2d(dim_in, dim_in, kernel_size = kernel_size, padding = padding, groups = dim_in, stride = stride, bias = bias),
        nn.Conv2d(dim_in, dim_out, kernel_size = 1, bias = bias)
    )

class FeedForward(Module):
    def __init__(
        self,
        dim,
        hidden_dim,
        dropout = 0.
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Conv2d(dim, hidden_dim, 1),
            nn.Hardswish(),
            DepthWiseConv2d(hidden_dim, hidden_dim, 3, padding = 1),
            nn.Hardswish(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_dim, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        h = w = int(sqrt(x.shape[-2]))
        x = rearrange(x, 'b (h w) c -> b c h w', h = h, w = w)
        x = self.net(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x

class Transformer(Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout = 0.
    ):
        super().__init__()

        self.layers = ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                TaylorSeriesLinearAttn(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

# main class

class ViT(Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        channels = 3,
        dim_head = 8,
        dropout = 0.,
        emb_dropout = 0.
    ):
        super().__init__()
        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.register_buffer('pos_embedding', posemb_sincos_2d(
            h = image_size // patch_size,
            w = image_size // patch_size,
            dim = dim,
        ), persistent = False)

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.mlp_head = nn.Sequential(
            Reduce('b n d -> b d', 'mean'),
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        x = x + self.pos_embedding
        x = self.dropout(x)

        x = self.transformer(x)

        return self.mlp_head(x)
