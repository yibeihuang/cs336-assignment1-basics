"""
RoPE implementation.

Here is the logical flow of data in a RoPE-based Transformer (like Llama):
1. Input: Token IDs from your BPE Tokenizer.
    2. Embedding: Look up the vectors (no positions added here!)
    3. Linear Projections: Generate (Q, K, V) tensors.
    4. RoPE Application: Rotate (Q, K) based on their position index.
    5. Attention: Calculate scores using the rotated (Q, K).
    6. Output: Multiply by (V) and continue to the FFN.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        Initialize a RoPE layer.

        Args:
            theta (float): The RoPE parameter.
            d_k (int): The dimension of key and query vectors.
            max_seq_len (int): The maximum sequence length that will be inputted.
            device (torch.device, optional): The device to store the parameters on.
        """
        super().__init__()
        self.theta = theta
        self.max_seq_len = max_seq_len
        self.d_k = d_k
        self.device = device
        self.register_buffer("rotation_matrix", self._create_rotation_matrix(), persistent=False)

    def _create_rotation_matrix(self) -> torch.Tensor:
        """
        Create a rotation matrix for RoPE.
        For each position m and dimension pair k, rotate by angle m * theta^(-2k/d_k).
        """
        rotation_matrix = torch.eye(self.d_k, device=self.device).unsqueeze(0).expand(
            self.max_seq_len, -1, -1
        ).clone()
        for m in range(self.max_seq_len):
            for k in range(self.d_k // 2):
                inv_freq = self.theta ** (-2 * k / self.d_k)
                angle = m * inv_freq
                c, s = math.cos(angle), math.sin(angle)
                rotation_matrix[m, 2 * k, 2 * k] = c
                rotation_matrix[m, 2 * k, 2 * k + 1] = -s
                rotation_matrix[m, 2 * k + 1, 2 * k] = s
                rotation_matrix[m, 2 * k + 1, 2 * k + 1] = c
        return rotation_matrix

    def forward(self, X: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape.
        Args:
            X (torch.Tensor): The input tensor of shape (..., seq_len, d_k).
            token_positions (torch.Tensor): The token positions of shape (..., seq_len).
        Returns:
            torch.Tensor: The output tensor of shape (..., seq_len, d_k).
        """
        return einsum(
            self.rotation_matrix[token_positions], X,
            "... seq d_out d_in, ... seq d_in -> ... seq d_out"
        )
