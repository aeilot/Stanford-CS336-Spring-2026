import math
import token

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
    d_ff: int

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        if d_ff is None:
            self.d_ff = 64 * math.ceil((8 * d_model / 3) / 64)
            # 8/3 is the recommended expansion factor for FFN in transformers
            # Round up to the next multiple of 64
        else:
            self.d_ff = d_ff
        self.up_proj = Linear(self.d_model, self.d_ff, device=device, dtype=dtype)
        self.down_proj = Linear(self.d_ff, self.d_model, device=device, dtype=dtype)
        self.gate_proj = Linear(self.d_model, self.d_ff, device=device, dtype=dtype)
        self.act_fn = SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class FFN(SwiGLU):
    pass


class RotaryPositionalEmbedding(nn.Module):
    cos_cached: torch.Tensor
    sin_cached: torch.Tensor

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        super().__init__()
        self.theta = theta

        # For every 2k-1 and 2k, we have the same frequency.
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device) / d_k))
        # t starts from 0, so that the first position has no rotation applied.
        t = torch.arange(max_seq_len, device=device).float()
        # Outer Prod to get the table of angle theta for every i and k
        freqs = torch.outer(t, inv_freq)
        # Cache the cos and sin values for efficiency
        # persistent = False means that these buffers will not be saved in the state_dict, which is useful for large models where you don't want to save these precomputed values.
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # Get the cos and sin values for the token positions
        # token_positions shape: (..., seq_len)
        # cos/sin_cached shape: (seq_len, d_k/2)
        # cos/sin shape: (..., seq_len, d_k/2)
        # x shape: (..., seq_len, d_k)
        cos = self.cos_cached[token_positions, :]
        sin = self.sin_cached[token_positions, :]
        cos = cos.to(device=x.device, dtype=x.dtype)
        sin = sin.to(device=x.device, dtype=x.dtype)
        # cos -sin
        # sin cos
        # x1, x2 shape: (..., seq_len, d_k/2)
        x1, x2 = x[..., ::2], x[..., 1::2]
        # No need to compute the whole R matrix
        # x1' = x1 * cos - x2 * sin
        # x2' = x1 * sin + x2 * cos
        x_rotated = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
        # flatten (starting from) -2 dim
        return x_rotated.flatten(-2)


# Scaled Dot-Product Attention


class Softmax(nn.Module):
    def forward(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        # Subtract the max for numerical stability
        x = x - x.max(dim=dim, keepdim=True).values
        exp_x = torch.exp(x)
        return exp_x / exp_x.sum(dim=dim, keepdim=True)


class ScaledDotProductAttention(nn.Module):
    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # q, k, v shape: (batch_size, ..., seq_len, d_k)
        d_k = q.size(-1)
        # Compute the dot product between queries and keys
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        # Apply the mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        # Apply softmax to get attention weights
        attn_weights = Softmax()(scores, dim=-1)
        # Compute the weighted sum of values
        output = torch.matmul(attn_weights, v)
        return output


class CausalMHA(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, dtype: torch.dtype | None = None, device: torch.device | None = None
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.dtype = dtype
        self.device = device

        # Linear layers for query, key, and value
        self.q_linear = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_linear = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_linear = Linear(d_model, d_model, device=device, dtype=dtype)
        # Output linear layer
        self.out_linear = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.q_linear(x)
        v = self.v_linear(x)
        k = self.k_linear(x)

        # Reshape q, k, v for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        rope = RotaryPositionalEmbedding(theta=10000, d_k=self.d_k, max_seq_len=seq_len, device=self.device)
        mask = torch.tril(torch.ones((seq_len, seq_len), device=self.device))

        # RoPE should be applied for each head separately
        if token_positions is not None:
            q = rope(q, token_positions)
            k = rope(k, token_positions)

        attn_output = ScaledDotProductAttention()(q, k, v, mask=mask)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        output = self.out_linear(attn_output)

        return output
