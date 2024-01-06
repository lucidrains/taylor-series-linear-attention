import torch
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torch import nn, einsum, Tensor

from einops import rearrange, pack, unpack
from einops.layers.torch import Rearrange

from typing import Optional
from torchtyping import TensorType

from rotary_embedding_torch import RotaryEmbedding

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# they use 2nd taylor expansion for exp(x)
# https://arxiv.org/abs/2209.04881
# in a linear attention formulation

def second_taylor_expansion(x: Tensor):
    dtype, device, dim = x.dtype, x.device, x.shape[-1]

    x, ps = pack([x], '* d')

    lead_dims = x.shape[0]

    # exp(qk) = 1 + qk + (qk)^2 / 2

    x0 = x.new_ones((lead_dims,))
    x1 = x
    x2 = einsum('... i, ... j -> ... i j', x, x) * (0.5 ** 0.5)

    # concat - dimension D now becomes (1 + D + D ^2)
    # in paper, they had to heavily reduce the attention head dimension to make this work

    out, _ = pack([x0, x1, x2], 'b *')
    out, = unpack(out, ps, '* d')
    return out

# main class

class TaylorSeriesLinearAttn(Module):
    def __init__(
        self,
        dim,
        *,
        dim_head = 16,
        heads = 8,
        one_headed_kv = False,
        rotary_emb = False
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        dim_inner = dim_head * heads

        kv_heads = heads if not one_headed_kv else 1
        dim_kv_inner = dim_head * (heads if not one_headed_kv else 1)

        self.rotary_emb = RotaryEmbedding(dim_head) if rotary_emb else None

        self.one_headed_kv = one_headed_kv

        self.to_q = nn.Sequential(
            nn.Linear(dim, dim_inner, bias = False),
            Rearrange('b n (h d) -> b h n d', h = heads)
        )

        self.to_kv = nn.Sequential(
            nn.Linear(dim, dim_kv_inner * 2, bias = False),
            Rearrange('b n (kv h d) -> kv b h n d', kv = 2, h = kv_heads)
        )

        self.to_out = nn.Sequential(
            Rearrange('b h n d -> b n (h d)'),
            nn.Linear(dim_inner, dim, bias = False)
        )

    def forward(
        self,
        x:          TensorType['batch', 'seq', 'dim'],
        mask:       Optional[TensorType['batch', 'seq']] = None,
        context:    Optional[TensorType['batch', 'target_seq', 'dim']] = None,
        eps: float = 1e-5
    ):
        """
        einops
        b - batch
        h - heads
        d - query / key head dimension
        e - value head dimension
        n - source query sequence length
        m - target key / value sequence length
        """
        is_cross_attn = exists(context)
        assert not (exists(self.rotary_emb) and is_cross_attn), 'rotary embedding does not work with cross attention'

        q = self.to_q(x)
        k, v = self.to_kv(default(context, x))

        # maybe rotary

        if exists(self.rotary_emb):
            q, k = map(self.rotary_emb.rotate_queries_or_keys, (q, k))

        # scale

        q = q * self.scale

        # 2nd taylor expansion for exp(qk)

        q, k = map(second_taylor_expansion, (q, k))

        # masking

        if exists(mask):
            mask = rearrange(mask, 'b n -> b 1 n 1')
            k = k.masked_fill(~mask, 0.)
            v = v.masked_fill(~mask, 0.)

        # linear attention

        if self.one_headed_kv:
            k, v = map(lambda t: rearrange(t, 'b 1 n d -> b n d'), (k, v))

            kv = einsum('b n d, b n e -> b d e', k, v)
            qk_inv = 1. / einsum('b h n d, b m d -> b h n', q, k).clamp(min = eps)
            out = einsum('b h n d, b d e, b h n -> b h n e', q, kv, qk_inv)

        else:
            kv = einsum('b h n d, b h n e -> b h d e', k, v)
            qk_inv = 1. / einsum('b h n d, b h m d -> b h n', q, k).clamp(min = eps)
            out = einsum('b h n d, b h d e, b h n -> b h n e', q, kv, qk_inv)

        # combine heads

        return self.to_out(out)
