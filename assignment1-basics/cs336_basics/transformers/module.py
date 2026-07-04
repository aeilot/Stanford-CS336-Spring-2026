import jaxtyping
import numpy as np
import torch
from torch import nn


class Linear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        # Initialize Weights
        self.weights = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        # Initialize Truncated Normal Distribution for Weights
        std = np.sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(self.weights, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Check if input tensor has the correct shape
        if x.shape[-1] != self.in_features:
            raise ValueError(f"Input tensor must have shape (*, {self.in_features}), but got {x.shape}")

        # Perform linear transformation
        return torch.matmul(x, self.weights.T)


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        # Initialize Embedding Weights
        self.weights = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
        # Initialize Truncated Normal Distribution for Embedding Weights
        nn.init.trunc_normal_(self.weights, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # Check if token_ids are within the valid range
        if not torch.all((token_ids >= 0) & (token_ids < self.num_embeddings)):
            raise ValueError(f"token_ids must be in the range [0, {self.num_embeddings - 1}]")

        # Perform embedding lookup
        return self.weights[token_ids]
