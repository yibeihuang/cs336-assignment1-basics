"""
Utility functions for the model.
"""

import torch

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