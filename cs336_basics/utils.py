"""
Utility functions for the model.
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
"""

import math
import torch
from einops import einsum

def softmax(X: torch.Tensor, i: int) -> torch.Tensor:
    """
    Apply softmax to the input tensor along the specified dimension i.
    To avoid exp(v_i) from becoming inf, we will substract the max value from the input tensor.
    softmax(v)_i = exp(v_i - max_values) / sum(exp(v_j - max_values) for j in range(X.shape[dim]))
    Args:
        X (torch.Tensor): The input tensor.
        i (int): The dimension to apply softmax to.
    Returns:
        torch.Tensor: The softmaxed tensor.
    """
    max_values = torch.max(X, dim=i, keepdim=True).values
    exp_values = torch.exp(X - max_values)
    sum_exp_values = torch.sum(exp_values, dim=i, keepdim=True)
    return exp_values / sum_exp_values


def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """
    Apply scaled dot-product attention to the input tensors. QK^T / sqrt(d_k) with optional mask.

    Args:
        Q (torch.Tensor): The query tensor of shape (batch_size, ..., seq_len, d_k).
        K (torch.Tensor): The key tensor of shape (batch_size, ..., seq_len, d_k).
        V (torch.Tensor): The value tensor of shape (batch_size, ..., seq_len, d_v).
        mask (torch.Tensor | None): The mask tensor of shape (batch_size, ..., seq_len, seq_len).
    Returns:
        torch.Tensor: The output tensor of shape (batch_size, ..., seq_len, d_v).
    """
    d_k = Q.shape[-1]
    scores = einsum(Q, K, "batch ... queries d_k, batch ... keys d_k -> batch ... queries keys") / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -float("inf"))
    return einsum(softmax(scores, i=-1), V, "batch ... queries keys, batch ... keys d_v -> batch ... queries d_v")