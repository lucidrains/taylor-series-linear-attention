# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "einops",
#     "fire",
#     "rotary-embedding-torch",
# ]
# ///

# demonstration of higher-order (order-2) linear attention with the taylor series
# feature map (zoology, https://arxiv.org/abs/2312.04927) on the function
# composition task - a single layer solves the two-hop lookup f1(f2(x))
#
#   - TaylorSeriesPolyAttention (order-2 kernel attention + taylor feature map, 1 layer)  <-- the linear variant, always a single layer
#   - PolyAttention            (order-2 softmax attention, 1 layer)    (reference)
#   - LinearAttention          (order-1 kernel attention, 4 layers)    (baseline, depth configurable)
#   - SelfAttention            (order-1 softmax attention, 2 layers)   (reference)
#
# run:
#   python train_function_composition.py

import fire
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Module, ModuleList, RMSNorm
from torch.optim import Adam

from einops import rearrange

from taylor_series_linear_attention import TaylorSeriesPolyAttention
from taylor_series_linear_attention.attention import second_taylor_expansion
from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# data

def function_composition(
    seq_len,
    batch_size = 32,
    num_classes = 10,
    x = None,
    composition_depth = 2,
    device = 'cpu'
):
    inputs = torch.zeros((batch_size, seq_len, 2), dtype = torch.long, device = device)

    x_vals = torch.full((batch_size,), x, device = device) if exists(x) else torch.randint(0, num_classes, (batch_size,), device = device)
    funcs = torch.randint(0, num_classes, (batch_size, composition_depth, num_classes), device = device)

    targets = x_vals.clone()
    for step in range(composition_depth):
        targets = funcs[torch.arange(batch_size, device = device), step, targets]

    total_pos = min(composition_depth * num_classes, seq_len)

    if total_pos > 0:
        step_idx = torch.arange(total_pos, device = device) // num_classes
        pos_idx = torch.arange(total_pos, device = device) % num_classes

        inputs[:, :total_pos, 0] = funcs[torch.arange(batch_size, device = device)[:, None], step_idx, pos_idx]
        inputs[:, :total_pos, 1] = pos_idx

    if seq_len > 0:
        inputs[:, -1, 0] = x_vals
        inputs[:, -1, 1] = 0

    return inputs, targets

# attention layers

class SelfAttention(Module):
    def __init__(self, dim, heads = 8, dim_head = 64, causal = False):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5

        dim_inner = dim_head * heads

        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias = False)
        self.to_out = nn.Linear(dim_inner, dim)

    def forward(self, x, mask = None):
        b, n, _ = x.shape

        q, k, v = rearrange(self.to_qkv(x), 'b n (qkv h d) -> qkv b h n d', qkv = 3, h = self.heads)

        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        mask_value = -torch.finfo(sim.dtype).max

        i, j = sim.shape[-2:]
        causal_mask = torch.ones((i, j), device = sim.device, dtype = torch.bool).triu(1)
        sim = sim.masked_fill(causal_mask, mask_value)

        if exists(mask):
            sim = sim.masked_fill(~mask[:, None, None, :], mask_value)

        attn = sim.softmax(dim = -1)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)

        return self.to_out(rearrange(out, 'b h n d -> b n (h d)'))

class PolyAttention(Module):
    # order-2 softmax poly attention (reference)

    def __init__(self, dim, heads = 8, dim_head = 64, causal = False):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5

        dim_inner = dim_head * heads

        self.to_q1 = nn.Linear(dim, dim_inner, bias = False)
        self.to_q2 = nn.Linear(dim, dim_inner, bias = False)
        self.to_q3 = nn.Linear(dim, dim_inner, bias = False)
        self.to_v3 = nn.Linear(dim, dim_inner, bias = False)

        self.to_out = nn.Linear(dim_inner, dim)

    def forward(self, x, mask = None):
        b, n, _ = x.shape

        q1, q2, q3, v3 = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads),
            (self.to_q1(x), self.to_q2(x), self.to_q3(x), self.to_v3(x))
        )

        mask_value = -torch.finfo(x.dtype).max

        # pass 1 - softmax attention from q2 to q3, aggregating v3

        sim = torch.einsum('b h i d, b h j d -> b h i j', q2, q3) * self.scale

        if exists(mask):
            sim = sim.masked_fill(~mask[:, None, None, :], mask_value)

        attn = sim.softmax(dim = -1)
        msg = torch.einsum('b h i j, b h j d -> b h i d', attn, v3)

        # pass 2 - softmax attention from q1 to q2, aggregating the messages

        sim = torch.einsum('b h i d, b h j d -> b h i j', q1, q2) * self.scale

        if exists(mask):
            sim = sim.masked_fill(~mask[:, None, None, :], mask_value)

        attn = sim.softmax(dim = -1)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, msg)

        return self.to_out(rearrange(out, 'b h n d -> b n (h d)'))

class LinearAttention(Module):
    # order-1 kernel attention (Katharopoulos et al.), the standard linear attention

    def __init__(self, dim, heads = 8, dim_head = 64, feature_map = 'elu', use_rotary_embed = True, eps = 1e-6):
        super().__init__()
        self.heads = heads
        self.feature_map = feature_map
        self.eps = eps

        dim_inner = dim_head * heads

        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias = False)
        self.q_norm = RMSNorm(dim_head)
        self.k_norm = RMSNorm(dim_head)
        self.rotary_emb = RotaryEmbedding(dim_head) if use_rotary_embed else None
        self.to_out = nn.Linear(dim_inner, dim)

    def _feature_map(self, t):
        if self.feature_map == 'identity':
            return t

        if self.feature_map == 'elu':
            return F.elu(t) + 1.

        if self.feature_map == 'relu':
            return F.relu(t)

        if self.feature_map == 'taylor':
            return second_taylor_expansion(t)

        raise ValueError(f'unknown feature map {self.feature_map}')

    def forward(self, x, mask = None):
        b, n, _ = x.shape

        q, k, v = rearrange(self.to_qkv(x), 'b n (qkv h d) -> qkv b h n d', qkv = 3, h = self.heads)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if exists(self.rotary_emb):
            freqs = self.rotary_emb(torch.arange(n, device = x.device))
            q = apply_rotary_emb(freqs, q)
            k = apply_rotary_emb(freqs, k)

        q = self._feature_map(q)
        k = self._feature_map(k)

        if exists(mask):
            k = k.masked_fill(~mask[:, None, :, None], 0.)
            v = v.masked_fill(~mask[:, None, :, None], 0.)

        S = torch.einsum('b h n d, b h n e -> b h d e', k, v)
        z = k.sum(dim = -2)

        num = torch.einsum('b h n d, b h d e -> b h n e', q, S)
        den = torch.einsum('b h n d, b h d -> b h n', q, z)

        out = num / (den[..., None] + self.eps)

        return self.to_out(rearrange(out, 'b h n d -> b n (h d)'))

class Block(Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 64,
        attn_type = 'linear_poly',
        causal = False,
        feature_map = 'elu',
        include_pass1_mass = True,
        use_rotary_embed = True
    ):
        super().__init__()

        if attn_type == 'linear_poly':
            self.attn = TaylorSeriesPolyAttention(dim, heads = heads, dim_head = dim_head, include_pass1_mass = include_pass1_mass, prenorm = True, use_rotary_embed = use_rotary_embed, multiply_root_value = True, use_root_value_as_attn_gate = True)
        elif attn_type == 'poly':
            self.attn = PolyAttention(dim, heads = heads, dim_head = dim_head, causal = causal)
        elif attn_type == 'linear':
            self.attn = LinearAttention(dim, heads = heads, dim_head = dim_head, feature_map = feature_map, use_rotary_embed = use_rotary_embed)
        elif attn_type == 'softmax':
            self.attn = SelfAttention(dim, heads = heads, dim_head = dim_head, causal = causal)
        else:
            raise ValueError(f'unknown attention type {attn_type}')

        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        x = self.attn(self.norm1(x)) + x
        return self.ffn(self.norm2(x)) + x

class Model(Module):
    def __init__(
        self,
        vocab_size,
        seq_len,
        dim = 128,
        heads = 4,
        dim_head = 32,
        layers = 6,
        attn_type = 'linear_poly',
        causal = False,
        feature_map = 'elu',
        include_pass1_mass = True,
        use_rotary_embed = False
    ):
        super().__init__()
        self.embedding = nn.Linear(vocab_size, dim)
        self.pos_enc = nn.Embedding(seq_len, dim)

        self.blocks = ModuleList([
            Block(dim, heads = heads, dim_head = dim_head, attn_type = attn_type, causal = causal, feature_map = feature_map, include_pass1_mass = include_pass1_mass, use_rotary_embed = use_rotary_embed)
            for _ in range(layers)
        ])

        self.norm = RMSNorm(dim)
        self.to_logits = nn.Linear(dim, vocab_size)

    def forward(self, x_one_hot):
        pos = torch.arange(x_one_hot.shape[1], device = x_one_hot.device)
        x = self.embedding(x_one_hot) + self.pos_enc(pos)

        for block in self.blocks:
            x = block(x)

        return self.to_logits(self.norm(x))

# training

def train_model(
    model_name,
    attn_type,
    layers,
    epochs,
    batch_size,
    lr,
    device,
    num_classes = 10,
    composition_depth = 2,
    seed = 42,
    feature_map = 'elu',
    include_pass1_mass = True,
    use_rotary_embed = True
):
    torch.manual_seed(seed)

    assert attn_type != 'linear_poly' or layers == 1, 'linear poly attention is a single layer by construction'

    seq_len = composition_depth * num_classes + 1
    vocab_size = 3 + num_classes + num_classes

    model = Model(
        vocab_size = vocab_size, seq_len = seq_len, dim = 128, heads = 4, dim_head = 32,
        layers = layers, attn_type = attn_type, feature_map = feature_map,
        include_pass1_mass = include_pass1_mass, use_rotary_embed = use_rotary_embed
    ).to(device)

    optimizer = Adam(model.parameters(), lr = lr)

    print(f"\ntraining {model_name} ({attn_type}, {layers} layers) for up to {epochs} epochs...")

    best_acc = 0.0
    epoch_hit_1 = None

    for epoch in range(epochs):
        inputs, targets = function_composition(
            seq_len, batch_size = batch_size, num_classes = num_classes,
            composition_depth = composition_depth, device = device
        )

        b_in = rearrange(F.one_hot(inputs, num_classes = num_classes), 'b n f c -> b n (f c)').float()
        b_in = F.pad(b_in, (3, 0))
        b_in[:, :num_classes, 0] = 1.
        b_in[:, num_classes:-1, 1] = 1.
        b_in[:, -1, 2] = 1.

        optimizer.zero_grad()
        logits = model(b_in)
        loss = F.cross_entropy(logits[:, -1], targets)
        loss.backward()
        optimizer.step()

        acc = (logits[:, -1].argmax(dim = -1) == targets).float().mean().item()
        best_acc = max(best_acc, acc)

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1:04d} | Loss: {loss.item():.4f} | Accuracy: {acc:.4f}")

        if acc >= 1.:
            epoch_hit_1 = default(epoch_hit_1, epoch + 1)
            break

    print(f"Best Accuracy for {model_name}: {best_acc:.4f}" + (f" (hit 1.0 at epoch {epoch_hit_1})" if epoch_hit_1 else ""))

    return best_acc, epoch_hit_1

def main(
    epochs: int = 1500,
    batch_size: int = 256,
    lr: float = 1e-3,
    num_classes: int = 10,
    composition_depth: int = 2,
    seed: int = 42,
    feature_map: str = 'taylor',
    include_pass1_mass: bool = True,
    linear_layers: int = 4,
    device: str = 'mps'
):
    if not torch.backends.mps.is_available() and device == 'mps':
        device = 'cpu'

    device = torch.device(device)

    print(f"running on {device} | feature_map = {feature_map} (order-1 baseline) | include_pass1_mass = {include_pass1_mass}")

    train_fn = partial(
        train_model,
        epochs = epochs, batch_size = batch_size, lr = lr,
        num_classes = num_classes, composition_depth = composition_depth,
        seed = seed, device = device, feature_map = feature_map,
        include_pass1_mass = include_pass1_mass
    )

    # linear poly attention is a single layer by design

    acc_linear_poly, epoch_linear_poly = train_fn("TaylorSeriesPolyAttention (1 layer)", attn_type = 'linear_poly', layers = 1)
    acc_poly, epoch_poly = train_fn("PolyAttention (1 layer)", attn_type = 'poly', layers = 1)
    acc_linear, epoch_linear = train_fn(f"LinearAttention ({linear_layers} layers)", attn_type = 'linear', layers = linear_layers)
    acc_base, epoch_base = train_fn("SelfAttention (2 layers)", attn_type = 'softmax', layers = 2)

    epoch_str = lambda e: str(e) if e else "never"

    print(f"""
Final Best Accuracies ({feature_map} feature map)
  TaylorSeriesPolyAttention (1 layer) : {acc_linear_poly:.4f} (hit 1.0 at epoch {epoch_str(epoch_linear_poly)})
  PolyAttention           (1 layer)   : {acc_poly:.4f} (hit 1.0 at epoch {epoch_str(epoch_poly)})
  LinearAttention         ({linear_layers} layers): {acc_linear:.4f} (hit 1.0 at epoch {epoch_str(epoch_linear)})
  SelfAttention           (2 layers)  : {acc_base:.4f} (hit 1.0 at epoch {epoch_str(epoch_base)})
""")

if __name__ == '__main__':
    fire.Fire(main)
