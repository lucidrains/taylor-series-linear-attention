from __future__ import annotations

import torch
from torch import einsum, is_tensor, nn
from torch.nn import Linear, Module, ModuleList

from einops.layers.torch import Rearrange

from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb

from taylor_series_linear_attention.attention import RMSNorm, second_taylor_expansion
from taylor_series_linear_attention.tensor_typing import Float, Bool

# helper functions

def exists(val):
    return val is not None

def first(t):
    return t[0]

def cast_tuple(t, length = 1):
    return (t,) * length if is_tensor(t) or not isinstance(t, tuple) else t

def resolve_context_mask(mask, context_mask):
    context_mask = cast_tuple(context_mask, 2) if exists(context_mask) else (None, None)
    context_mask1, context_mask2 = context_mask

    # by default, the query mask also masks keys, mirroring the softmax poly attention

    if not exists(context_mask1) and exists(mask):
        context_mask1 = mask

    if not exists(context_mask2) and exists(mask):
        context_mask2 = mask

    return context_mask1, context_mask2

# taylor series poly attention (order 2)
# kernel attention instantiation of order-2 poly attention
# https://arxiv.org/abs/2602.02422
#
# the taylor feature map (2nd order expansion of exp(qk)) is used on both
# passes by design, expanding the head dimension from d to 1 + d + d^2
# https://arxiv.org/abs/2312.04927
#
# memory is O(b h (n d^2 + d^3)) - linear in the sequence length n, with the
# quadratic cost pushed into the attention head dimension, so d should be kept
# small. with remove_even_power_dups, the d^2 terms are cut roughly in half

def taylor_series_poly_attention(
    q1: Float['b h n d'],
    q2: Float['b h n d'],
    q3: Float['b h n d'],
    v3: Float['b h n e'],
    remove_even_power_dups = False,
    include_pass1_mass = True,
    mask: Bool['b n'] | None = None,
    context_mask: Bool['b n'] | tuple[Bool['b n'], Bool['b n']] | None = None,
    eps = 1e-6
) -> Float['b h n e']:
    """
    einops
    b - batch
    h - heads
    n - source sequence length
    d - query / key head dimension
    e - value head dimension

    pass 1: kernel attention from q2 to q3, aggregating the values v3
    pass 2: kernel attention from q1 to q2, aggregating the pass-1 messages

    both passes weighted by their mass, giving the kernel analog of
    order-2 softmax poly attention

    mask        - padding mask for the queries (True = valid), also masks the keys by default
    context_mask - padding mask(s) for the keys only, a single mask or a tuple of 2,
                  masking the keys of pass 1 (q2) and pass 2 (q3) respectively
    """
    assert not exists(mask) or mask.dtype == torch.bool

    context_mask1, context_mask2 = resolve_context_mask(mask, context_mask)

    phi1, phi2, phi3 = (second_taylor_expansion(t, remove_even_power_dups) for t in (q1, q2, q3))

    if exists(context_mask1):
        phi2 = phi2 * context_mask1.float()[:, None, :, None]

    if exists(context_mask2):
        phi3 = phi3 * context_mask2.float()[:, None, :, None]

    # pass 1 - kernel attention from q2 to q3 with values v3

    S3 = einsum('b h n d, b h n e -> b h d e', phi3, v3)
    z3 = phi3.sum(dim = -2)

    s23 = einsum('b h n d, b h d -> b h n', phi2, z3)
    msg_un = einsum('b h n d, b h d e -> b h n e', phi2, S3)

    # pass 2 - kernel attention from q1 to q2, weighted by the pass-1 mass

    if include_pass1_mass:
        S12 = einsum('b h n d, b h n e -> b h d e', phi2, msg_un)
        z12 = (s23[..., None] * phi2).sum(dim = -2)
    else:
        S12 = einsum('b h n d, b h n e -> b h d e', phi2, msg_un / s23.clamp_min(eps).unsqueeze(-1))
        z12 = phi2.sum(dim = -2)

    num = einsum('b h n d, b h d e -> b h n e', phi1, S12)
    den = einsum('b h n d, b h d -> b h n', phi1, z12)

    out = num / (den[..., None] + eps)

    # fully masked query rows (all keys masked) should output zero

    out = torch.where((den != 0.)[..., None], out, torch.zeros_like(out))

    if exists(mask):
        out = out.masked_fill(~mask[:, None, :, None], 0.)

    return out

# main class

class TaylorSeriesPolyAttention(Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 64,
        remove_even_power_dups = False,
        include_pass1_mass = True,
        prenorm = False,
        multiply_root_value = False,
        use_root_value_as_attn_gate = True,
        attn_gate = False,
        use_rotary_embed = True,
        causal = False,
        eps = 1e-6
    ):
        super().__init__()

        assert not causal, 'autoregressive variant not yet supported'

        self.norm = RMSNorm(dim) if prenorm else nn.Identity()

        self.remove_even_power_dups = remove_even_power_dups
        self.include_pass1_mass = include_pass1_mass
        self.eps = eps

        self.multiply_root_value = multiply_root_value
        self.use_root_value_as_attn_gate = use_root_value_as_attn_gate
        self.attn_gate = attn_gate

        dim_inner = dim_head * heads

        q_split = 2 if attn_gate else 1
        self.split_q = Rearrange('b n (split h d) -> split b h n d', split = q_split, h = heads)

        self.has_root_v = multiply_root_value
        kv1_split = 2 if self.has_root_v else 1
        self.split_kv1 = Rearrange('b n (split h d) -> split b h n d', split = kv1_split, h = heads)

        self.split_kv = Rearrange('b n (split h d) -> split b h n d', split = 2, h = heads)

        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        self.to_q = Linear(dim, dim_inner * q_split, bias = False)
        self.to_kvs = ModuleList([
            Linear(dim, dim_inner * kv1_split, bias = False),
            Linear(dim, dim_inner * 2, bias = False)
        ])

        self.q_norms = ModuleList([RMSNorm(dim_head) for _ in range(3)])

        self.rotary_emb = RotaryEmbedding(dim_head) if use_rotary_embed else None

        self.to_out = Linear(dim_inner, dim)

    def forward(
        self,
        x: Float['b n d'],
        mask: Bool['b n'] | None = None,
        context_mask: Bool['b n'] | tuple[Bool['b n'], Bool['b n']] | None = None,
        rotary_pos_emb = None
    ) -> Float['b n d']:
        """
        einops
        b - batch
        h - heads
        n - source sequence length
        d - feature dimension

        mask      - padding mask for the queries (True = valid), also masks the keys by default
        context_mask - padding mask(s) for the keys only, a single mask or a tuple of 2,
                    masking the keys of pass 1 (q2) and pass 2 (q3) respectively
        """
        orig_x = x
        x = self.norm(x)

        q_and_maybe_gates = self.split_q(self.to_q(x))

        if self.attn_gate:
            q1, gates = q_and_maybe_gates
        else:
            q1 = first(q_and_maybe_gates)

        # kvs

        kv1 = self.split_kv1(self.to_kvs[0](orig_x))
        kv2 = self.split_kv(self.to_kvs[1](orig_x))

        q2 = first(kv1)
        v2 = kv1[1] if self.has_root_v else None
        q3, v3 = kv2

        # qk rmsnorm

        q1, q2, q3 = (norm(q) for norm, q in zip(self.q_norms, (q1, q2, q3)))

        # rotary for relative positions - applied to the queries and keys only,
        # never the values. enabled by default

        if exists(rotary_pos_emb):
            q1, q2, q3 = (apply_rotary_emb(rotary_pos_emb, q) for q in (q1, q2, q3))
        elif exists(self.rotary_emb):
            rotate_fn = self.rotary_emb.rotate_queries_or_keys

            q1, q2, q3 = map(rotate_fn, (q1, q2, q3))

        out = taylor_series_poly_attention(
            q1, q2, q3, v3,
            remove_even_power_dups = self.remove_even_power_dups,
            include_pass1_mass = self.include_pass1_mass,
            mask = mask,
            context_mask = context_mask,
            eps = self.eps
        )

        # elementwise multiply root values

        if self.multiply_root_value:
            if self.use_root_value_as_attn_gate:
                v2 = v2.sigmoid()

            out = out * v2

        # attention gate

        if self.attn_gate:
            out = out * gates.sigmoid()

        # combine heads

        out = self.to_out(self.merge_heads(out))

        # masked query rows should output zero

        if exists(mask):
            out = out.masked_fill(~mask[:, :, None], 0.)

        return out
