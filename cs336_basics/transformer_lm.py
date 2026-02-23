'''
Transformer language model implementation.
'''

import torch
import torch.nn as nn
from cs336_basics.transformer_block import TransformerBlock
from cs336_basics.embedding import Embedding
from cs336_basics.linear import Linear
from cs336_basics.rmsnorm import RMSNorm

class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, num_layers: int, d_model: int, num_heads: int, d_ff: int, theta: float, max_seq_len: int):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, theta, max_seq_len) for _ in range(num_layers)])
        self.linear = Linear(d_model, vocab_size)
        self.ln = RMSNorm(d_model)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape (batch_size, seq_len) and
        return a tensor of the same shape.
        """
        batch_size, seq_len = X.shape
        X = self.embedding(X)
        for transformer_block in self.transformer_blocks:
            X = transformer_block(X)
        normalized_output = self.ln(X)
        return self.linear(normalized_output)