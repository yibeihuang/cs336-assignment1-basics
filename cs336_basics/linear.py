"""Linear layer implementation.
"""

import math
import torch
import torch.nn as nn
from einops import einsum

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        """
        Initialize a linear layer.

        Args:
            in_features (int): final dimension of the input features.
            out_features (int): final dimension of the output features.
            device (torch.device, optional): Device to store the parameters on.
            dtype (torch.dtype, optional): Data type of the parameters.
        """
        super().__init__()
        self.weights = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        std = math.sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(self.weights, std=std, a=-3 * std, b=3 * std)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Apply the linear transformation to the input tensor. y = Wx
        """
        return einsum(self.weights, X, "d_out d_in, ... d_in -> ... d_out")
