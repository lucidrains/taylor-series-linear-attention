import torch
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torch import nn, einsum

from einops import rearrange
from einops.layers.torch import Rearrange

# functions

def exists(v):
    return v is not None

# main class

class TaylorSeriesLinearAttn(Module):
    def __init__(
        self,
        dim,
        *,
        dim_head = 64,
        heads = 8
    ):
        super().__init__()

    def forward(self, x):
        return x
