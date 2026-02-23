'''
The full transformer block implementation.
'''

import torch
import torch.nn as nn
from cs336_basics.multihead_self_attention import MultiheadSelfAttention
from cs336_basics.positionwise_feedforward import PositionwiseFeedforward
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.rope import RoPE

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta: float, max_seq_len: int):
        super().__init__()
        self.theta = theta
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)
        self.ffn = PositionwiseFeedforward(d_model, d_ff)
        self.rope = RoPE(theta, d_model // num_heads, max_seq_len)
        self.attn = MultiheadSelfAttention(d_model, num_heads)

    def forward(self, X: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Process an input tensor of shape (batch_size, seq_len, d_model) and
        return a tensor of the same shape.
        """
        batch_size, seq_len, d_model = X.shape
        token_positions = torch.arange(seq_len, device=X.device)
        attn_out = self.attn.forward_with_rope(self.ln1(X), self.rope, token_positions, mask)
        residual = X + attn_out
        return residual + self.ffn(self.ln2(residual))
