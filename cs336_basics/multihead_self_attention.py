'''
Multi-head self-attention implementation.
'''

import torch
import torch.nn as nn
from einops import einsum, rearrange
from cs336_basics.utils import scaled_dot_product_attention
from cs336_basics.rope import RoPE


class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        """
        Initialize a multi-head self-attention layer.
        Weights are stored in concatenated format (d_model, d_model) to match reference.
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        self.q_proj = nn.Parameter(torch.empty(d_model, d_model))
        self.k_proj = nn.Parameter(torch.empty(d_model, d_model))
        self.v_proj = nn.Parameter(torch.empty(d_model, d_model))
        self.o_proj = nn.Parameter(torch.empty(d_model, d_model))
        std = (2 / (d_model + d_model)) ** 0.5
        nn.init.trunc_normal_(self.q_proj, std=std, a=-3 * std, b=3 * std)
        nn.init.trunc_normal_(self.k_proj, std=std, a=-3 * std, b=3 * std)
        nn.init.trunc_normal_(self.v_proj, std=std, a=-3 * std, b=3 * std)
        nn.init.trunc_normal_(self.o_proj, std=std, a=-3 * std, b=3 * std)

    def forward(self, X: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Process an input tensor of shape (batch_size, seq_len, d_model) and
        return a tensor of the same shape.
        """
        # Reshape to per-head: (d_model, d_model) -> (num_heads, d_k, d_model)
        q_w = rearrange(self.q_proj, "(heads d_k) d_model -> heads d_k d_model", heads=self.num_heads)
        k_w = rearrange(self.k_proj, "(heads d_k) d_model -> heads d_k d_model", heads=self.num_heads)
        v_w = rearrange(self.v_proj, "(heads d_v) d_model -> heads d_v d_model", heads=self.num_heads)
        o_w = rearrange(self.o_proj, "d_model (heads d_v) -> heads d_model d_v", heads=self.num_heads)

        q = einsum(q_w, X, "heads d_k d_model, batch seq d_model -> batch heads seq d_k")
        k = einsum(k_w, X, "heads d_k d_model, batch seq d_model -> batch heads seq d_k")
        v = einsum(v_w, X, "heads d_v d_model, batch seq d_model -> batch heads seq d_v")

        if mask is None:
            # Causal mask: 1 = attend, 0 = mask out (fill with -inf)
            mask = torch.tril(torch.ones(X.shape[1], X.shape[1], device=X.device))

        attn_out = scaled_dot_product_attention(q, k, v, mask)
        return einsum(o_w, attn_out, "heads d_model d_v, batch heads seq d_v -> batch seq d_model")

    def forward_with_rope(
        self,
        X: torch.Tensor,
        rope: RoPE,
        token_positions: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward with RoPE applied to Q and K before attention."""
        q_w = rearrange(self.q_proj, "(heads d_k) d_model -> heads d_k d_model", heads=self.num_heads)
        k_w = rearrange(self.k_proj, "(heads d_k) d_model -> heads d_k d_model", heads=self.num_heads)
        v_w = rearrange(self.v_proj, "(heads d_v) d_model -> heads d_v d_model", heads=self.num_heads)
        o_w = rearrange(self.o_proj, "d_model (heads d_v) -> heads d_model d_v", heads=self.num_heads)

        q = einsum(q_w, X, "heads d_k d_model, batch seq d_model -> batch heads seq d_k")
        k = einsum(k_w, X, "heads d_k d_model, batch seq d_model -> batch heads seq d_k")
        v = einsum(v_w, X, "heads d_v d_model, batch seq d_model -> batch heads seq d_v")

        # Apply RoPE: reshape to (batch*heads, seq, d_k) for RoPE, then reshape back
        batch, heads, seq, d_k = q.shape
        q_flat = rearrange(q, "batch heads seq d_k -> (batch heads) seq d_k")
        k_flat = rearrange(k, "batch heads seq d_k -> (batch heads) seq d_k")
        pos_expanded = token_positions.expand(batch * heads, -1)
        q = rearrange(rope(q_flat, pos_expanded), "(batch heads) seq d_k -> batch heads seq d_k", heads=heads)
        k = rearrange(rope(k_flat, pos_expanded), "(batch heads) seq d_k -> batch heads seq d_k", heads=heads)

        if mask is None:
            mask = torch.tril(torch.ones(seq, seq, device=X.device))
        attn_out = scaled_dot_product_attention(q, k, v, mask)
        return einsum(o_w, attn_out, "heads d_model d_v, batch heads seq d_v -> batch seq d_model")