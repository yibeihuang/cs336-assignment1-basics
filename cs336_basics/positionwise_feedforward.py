"""
Positionwise Feedforward layer implementation.

FFN(x) = SwiGLU(x, W1, W2, W3) = W2(SiLU(W1x) ⊙ W3x)
where x ∈ Rdmodel , W1, W3 ∈ Rdff×dmodel , W2 ∈ Rdmodel×dff , and canonically, dff = 83 dmod
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum

class PositionwiseFeedforward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        """
        Initialize a PositionwiseFeedforward layer.

        Args:
            d_model (int): The dimensionality of the input and output.
            d_ff (int): The dimensionality of the feedforward layer.
            device (torch.device, optional): The device to store the parameters on.
            dtype (torch.dtype, optional): The data type of the parameters.
        """
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype
        self.w1 = nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))
        self.w2 = nn.Parameter(torch.empty(d_model, d_ff, device=device, dtype=dtype))
        self.w3 = nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))
        std = 2 / (d_model + d_ff)
        nn.init.trunc_normal_(self.w1, std=std, a=-3 * std, b=3 * std)
        nn.init.trunc_normal_(self.w2, std=std, a=-3 * std, b=3 * std)
        nn.init.trunc_normal_(self.w3, std=std, a=-3 * std, b=3 * std)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Apply the positionwise feedforward layer to the input tensor.
        FFN(x) = SwiGLU(x, W1, W2, W3) = W2(SiLU(W1x) ⊙ W3x)
        """
        # part 1:W1x
        swish_input = einsum(self.w1, X, "d_ff d_model, ... d_model -> ... d_ff")
        # part 2: SiLU(x) = x * sigmoid(x) = x / (1 + e^-x)
        silu_output = einsum(swish_input, torch.sigmoid(swish_input), "... d_ff, ... d_ff -> ... d_ff")
        # part 3: W3x
        right_val = einsum(self.w3, X, "d_ff d_model, ... d_model -> ... d_ff")
        # W2(SiLU(W1x) ⊙ W3x)
        return einsum(
            self.w2,
            einsum(silu_output, right_val, "... d_ff, ... d_ff -> ... d_ff"), "d_model d_ff, ... d_ff -> ... d_model")
