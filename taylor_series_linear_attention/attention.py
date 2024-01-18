from functools import partial

import torch
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torch import nn, einsum, Tensor

from einops import rearrange, pack, unpack
from einops.layers.torch import Rearrange

from typing import Optional
from torchtyping import TensorType

from rotary_embedding_torch import RotaryEmbedding

import importlib

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def shift(t):
    t, t_shift = t.chunk(2, dim = -1)
    t_shift = F.pad(t_shift, (0, 0, 1, -1), value = 0.)
    return torch.cat((t, t_shift), dim = -1)

# prenorm

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self.gamma * F.normalize(x, dim = -1) * self.scale

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
        causal = False,
        one_headed_kv = False,
        rotary_emb = False,
        combine_heads = True,
        gate_value_heads = False,
        prenorm = False,
        shift_tokens = False
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        dim_inner = dim_head * heads

        self.shift_tokens = shift_tokens
        self.norm = RMSNorm(dim) if prenorm else nn.Identity()

        self.heads = heads
        self.dim_hidden = dim_inner

        self.causal = causal
        self.causal_linear_attn_fn = None

        if causal:
            if not exists(importlib.util.find_spec('fast_transformers')):
                print('pytorch-fast-transformers must be installed. `pip install pytorch-fast-transformers` first')
                exit()

            from fast_transformers.causal_product import CausalDotProduct
            self.causal_linear_attn_fn = CausalDotProduct.apply

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

        self.to_v_gates = nn.Sequential(
            nn.Linear(dim, heads, bias = False),
            nn.Sigmoid(),
            Rearrange('b n h -> b h n 1')
        ) if gate_value_heads else None

        self.merge_heads = Rearrange('b h n d -> b n (h d)')
        self.to_out = nn.Identity()

        if combine_heads:
            self.to_out = nn.Linear(dim_inner, dim, bias = False)

    def forward(
        self,
        x:          TensorType['batch', 'seq', 'dim', float],
        mask:       Optional[TensorType['batch', 'seq', bool]] = None,
        context:    Optional[TensorType['batch', 'target_seq', 'dim', float]] = None,
        eps: float = 1e-5,
        cache = None,
        return_cache = False
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
        orig_input, seq_len, is_cross_attn = x, x.shape[-2], exists(context)
        assert not (exists(self.rotary_emb) and is_cross_attn), 'rotary embedding does not work with cross attention'

        # token shift from rwkv

        if self.shift_tokens:
            if exists(cache):
                _, last_token, *_ = cache
                x, ps = pack([last_token, x], 'b * d')

            x = shift(x)

            if exists(cache):
                _, x = unpack(x, ps, 'b * d')

        # pre rmsnorm

        normed = self.norm(x)

        # queries, keys, values

        q = self.to_q(normed)
        k, v = self.to_kv(default(context, normed))

        # maybe rotary

        if exists(self.rotary_emb):
            rotate_fn = self.rotary_emb.rotate_queries_or_keys

            if exists(cache):
                cache_length, *_ = cache
                rotate_fn = partial(rotate_fn, offset = cache_length)

            q, k = map(rotate_fn, (q, k))

        # scale

        q = q * self.scale

        # 2nd taylor expansion for exp(qk)

        q, k = map(second_taylor_expansion, (q, k))

        # linear attention

        if self.causal:
            assert not exists(mask), 'masking does not make sense for autoregressive linear attention'
            assert not is_cross_attn, 'causal does not make sense with cross attention'

            if self.one_headed_kv:
                k, v = map(lambda t: repeat(t, 'b 1 n d -> b h n d', h = self.heads), (k, v))

            if exists(cache):
                assert seq_len == 1
                old_seq_len, _, kv_cumsum_cache, k_cumsum_cache = cache

                kv = einsum('b h n d, b h n e -> b h d e', k, v)

                kv_cumsum = kv + kv_cumsum_cache
                k_cumsum = k + k_cumsum_cache

                num = einsum('b h n d, b h d e -> b h n e', q, kv_cumsum)
                den = einsum('... n d, ... n d -> ... n', q, k_cumsum)
                den = rearrange(den, '... -> ... 1')

                out = num / den.clamp(min = eps)

                if return_cache:
                    new_cache = (old_seq_len + 1, orig_input, kv_cumsum, k_cumsum)

            else:

                num = self.causal_linear_attn_fn(q, k, v)

                k_cumsum = k.cumsum(dim = -2)
                den = einsum('... n d, ... n d -> ... n', q, k_cumsum)
                den = rearrange(den, '... -> ... 1')

                out = num / den.clamp(min = eps)

                if return_cache:
                    new_kv_cache = einsum('b h n d, b h n e -> b h d e', k, v)
                    new_k_cumsum_cache = k_cumsum[..., -1:, :]
                    new_cache = (seq_len, orig_input[:, -1:], new_kv_cache, new_k_cumsum_cache)

        else:
            assert not return_cache, 'cache is only needed for autoregressive'

            if exists(mask):
                mask = rearrange(mask, 'b n -> b 1 n 1')
                k = k.masked_fill(~mask, 0.)
                v = v.masked_fill(~mask, 0.)

            if self.one_headed_kv:
                k, v = map(lambda t: rearrange(t, 'b 1 n d -> b n d'), (k, v))

                kv = einsum('b n d, b n e -> b d e', k, v)
                qk_inv = 1. / einsum('b h n d, b m d -> b h n', q, k).clamp(min = eps)
                out = einsum('b h n d, b d e, b h n -> b h n e', q, kv, qk_inv)

            else:
                kv = einsum('b h n d, b h n e -> b h d e', k, v)
                qk_inv = 1. / einsum('b h n d, b h m d -> b h n', q, k).clamp(min = eps)
                out = einsum('b h n d, b h d e, b h n -> b h n e', q, kv, qk_inv)

        # gate value heads

        if exists(self.to_v_gates):
            out = out * self.to_v_gates(x)

        # merge heads

        out = self.merge_heads(out)

        # maybe combine heads

        out = self.to_out(out)

        if not return_cache:
            return out

        return out, new_cache

# adapted for images and video

class ChannelFirstTaylorSeriesLinearAttn(Module):
    def __init__(
        self,
        *args,
        **kwargs
    ):
        super().__init__()
        self.attn = TaylorSeriesLinearAttn(*args, **kwargs)

    def forward(
        self,
        x: Tensor
    ):
        x = rearrange(x, 'b c ... -> b ... c')
        x, ps = pack([x], 'b * c')

        out = self.attn(x)

        out, = unpack(out, ps, 'b * c')
        return rearrange(out, 'b ... c -> b c ...')
