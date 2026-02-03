"""
Embedding layer implementation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        """
        Initialize an embedding layer.

        Args:
            num_embeddings (int): The number of embeddings in the vocabulary.
            embedding_dim (int): The dimension of the embeddings.
            device (torch.device, optional): Device to store the parameters on.
            dtype (torch.dtype, optional): Data type of the parameters.
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        std = math.sqrt(2 / (num_embeddings + embedding_dim))
        nn.init.trunc_normal_(self.embeddings, std=std, a=-3 * std, b=3 * std)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Apply the embedding layer to the input tensor.
        """
        x_one_hot = F.one_hot(X, num_classes=self.num_embeddings).float()
        return einsum(x_one_hot, self.embeddings, "... d_vocab, d_vocab d_model -> ... d_model")
