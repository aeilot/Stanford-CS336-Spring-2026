import math

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


# Pre-Norm Transformer
# Moving the LayerNorm to the input of the sublayer instead of the output.
# This is known as Pre-Norm and can help with training stability in deep transformers.


class RMSNorm(nn.Module):
    def __init__(
        self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype

        # Initialize Scale Parameter
        self.scale = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cast to Float32 to prevent overflow during computation
        in_dtype = x.dtype
        x = x.to(torch.float32)

        # Keepdim: True to maintain the original shape for broadcasting
        # ALWAYS REMEMBER
        # The rule of broadcasting: First we align from the left, and match from the right.
        # If the dims are not equal and one of them is 1, we can broadcast the smaller dim to match the larger dim.
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)

        result = x / rms * self.scale

        return result.to(in_dtype)


class SiLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(
        self, d_model: int, d_ff: int = 0, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = 8 / 3 * d_model  # 8/3 is the recommended expansion factor for FFN in transformers
        # Round to nearest 64 multiple
        if d_ff < 64:
            d_ff = int(8 * d_model / 3)
            d_ff = 64 * math.ceil(d_ff / 64)
        self.d_ff = d_ff
        self.up_proj = Linear(self.d_model, self.d_ff, device=device, dtype=dtype)
        self.down_proj = Linear(self.d_ff, self.d_model, device=device, dtype=dtype)
        self.gate_proj = Linear(self.d_model, self.d_ff, device=device, dtype=dtype)
        self.act_fn = SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class FFN(SwiGLU):
    pass

