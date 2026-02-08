"""
RMSNorm layer implementation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """
        Initialize an RMSNorm layer.
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.in_type = dtype
        # Initialize the scale parameter
        self.gamma = nn.Parameter(torch.empty(d_model, device=device, dtype=dtype))
        std = math.sqrt(2 / (d_model + d_model))
        nn.init.trunc_normal_(self.gamma, std=std, a=-3 * std, b=3 * std)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape (batch_size, sequence_length, d_model) and
        return a tensor of the same shape.
        """
        X = X.to(torch.float32)
        rms_x = torch.sqrt(torch.mean(X ** 2, dim=-1, keepdim=True) + self.eps)
        return einsum(X / rms_x, self.gamma, "... d_model, d_model -> ... d_model").to(self.in_type)
