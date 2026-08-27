import pytest
import torch

from taylor_series_linear_attention import (
  TaylorSeriesLinearAttn,
  TaylorSeriesPolyAttention,
  ChannelFirstTaylorSeriesLinearAttn
)

from taylor_series_linear_attention.attention import (
  RMSNorm,
  second_taylor_expansion
)

# helpers

def exists(val):
    return val is not None

# device

def get_device():
    return torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

DEVICE = get_device()

# taylor series (order-1) linear attention

def test_taylor_series_linear_attn():
    attn = TaylorSeriesLinearAttn(dim = 512, dim_head = 16, heads = 16).to(DEVICE)

    x = torch.randn(1, 1024, 512, device = DEVICE)
    mask = torch.ones((1, 1024), dtype = torch.bool, device = DEVICE)

    out = attn(x, mask = mask)

    assert x.shape == out.shape

def test_taylor_series_linear_attn_cross_attention():
    attn = TaylorSeriesLinearAttn(dim = 512, dim_head = 16, heads = 16).to(DEVICE)

    x = torch.randn(1, 512, 512, device = DEVICE)
    context = torch.randn(1, 1024, 512, device = DEVICE)
    context_mask = torch.ones((1, 1024), dtype = torch.bool, device = DEVICE)

    out = attn(x, context = context, mask = context_mask)

    assert x.shape == out.shape

def test_taylor_series_linear_attn_one_headed_kv():
    attn = TaylorSeriesLinearAttn(dim = 512, dim_head = 16, heads = 16, one_headed_kv = True).to(DEVICE)

    x = torch.randn(1, 1024, 512, device = DEVICE)

    out = attn(x)

    assert x.shape == out.shape

def test_taylor_series_linear_attn_remove_even_power_dups():
    attn = TaylorSeriesLinearAttn(dim = 512, dim_head = 16, heads = 16, remove_even_power_dups = True).to(DEVICE)

    x = torch.randn(1, 256, 512, device = DEVICE)

    out = attn(x)

    assert x.shape == out.shape

def test_taylor_series_linear_attn_gradient_flows():
    attn = TaylorSeriesLinearAttn(dim = 512, dim_head = 16, heads = 16).to(DEVICE)

    x = torch.randn(1, 64, 512, device = DEVICE)

    out = attn(x)
    out.sum().backward()

    assert all(exists(p.grad) and p.grad.abs().sum() > 0. for p in attn.parameters())

# taylor series (order-2) poly attention

def test_taylor_series_poly_attention():
    attn = TaylorSeriesPolyAttention(dim = 512, heads = 16, dim_head = 32, prenorm = True).to(DEVICE)

    x = torch.randn(1, 1024, 512, device = DEVICE)

    out = attn(x)

    assert x.shape == out.shape

def test_taylor_series_poly_attention_gradient_flows():
    attn = TaylorSeriesPolyAttention(dim = 512, heads = 8, dim_head = 32, prenorm = True, multiply_root_value = True).to(DEVICE)

    x = torch.randn(1, 128, 512, device = DEVICE)

    out = attn(x)
    out.sum().backward()

    # rotary freqs are recomputed from arange, so exclude them

    assert all(exists(p.grad) and p.grad.abs().sum() > 0. for n, p in attn.named_parameters() if 'rotary_emb' not in n)

def test_taylor_series_poly_attention_padding_mask():
    attn = TaylorSeriesPolyAttention(dim = 512, heads = 8, dim_head = 32, prenorm = True).to(DEVICE)

    x = torch.randn(2, 16, 512, device = DEVICE)
    mask = torch.ones((2, 16), dtype = torch.bool, device = DEVICE)
    mask[0, -4:] = False

    out = attn(x, mask = mask)

    # padded query rows should output zero, valid rows should be unaffected by padding

    assert (out[0, -4:].abs() < 1e-6).all()
    assert out[0, 0].abs().sum() > 0.

def test_taylor_series_poly_attention_no_rotary():
    attn = TaylorSeriesPolyAttention(dim = 512, heads = 8, dim_head = 32, use_rotary_embed = False).to(DEVICE)

    x = torch.randn(1, 128, 512, device = DEVICE)

    out = attn(x)

    assert x.shape == out.shape

def test_taylor_series_poly_attention_remove_even_power_dups():
    attn = TaylorSeriesPolyAttention(dim = 512, heads = 8, dim_head = 32, remove_even_power_dups = True).to(DEVICE)

    x = torch.randn(1, 128, 512, device = DEVICE)

    out = attn(x)

    assert x.shape == out.shape

def test_taylor_series_poly_attention_causal_not_supported():
    with pytest.raises(AssertionError):
        TaylorSeriesPolyAttention(dim = 512, causal = True)

# channel first

def test_channel_first_taylor_series_linear_attn():
    attn = ChannelFirstTaylorSeriesLinearAttn(dim = 512, dim_head = 16, heads = 16).to(DEVICE)

    image = torch.randn(1, 512, 8, 8, 8, device = DEVICE)

    out = attn(image)

    assert out.shape == image.shape

# taylor expansion

def test_second_taylor_expansion():
    x = torch.randn(2, 4, 8, 16, device = DEVICE)

    out = second_taylor_expansion(x)

    assert out.shape[-1] == 1 + 16 + 16 ** 2

def test_second_taylor_expansion_remove_even_power_dups():
    x = torch.randn(2, 4, 8, 16, device = DEVICE)

    out = second_taylor_expansion(x, remove_even_power_dups = True)

    assert out.shape[-1] == 1 + 16 + 16 * 17 // 2

# rmsnorm

def test_rmsnorm():
    norm = RMSNorm(64).to(DEVICE)

    x = torch.randn(2, 16, 64, device = DEVICE)

    out = norm(x)

    assert out.shape == x.shape
