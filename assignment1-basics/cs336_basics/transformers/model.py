import math

import torch
from torch import nn

from cs336_basics.transformers.module import (
    Embedding,
    Linear,
    RMSNorm,
    RotaryPositionalEmbedding,
    Softmax,
    TransformerBlock,
)


class TransformerLM(nn.Module):
    rope_theta: float

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        rope_theta: float = 10000.0,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.rope_theta = rope_theta
        self.d_ff = d_ff if d_ff is not None else 64 * math.ceil((8 * d_model / 3) / 64)

        self.embedding = Embedding(vocab_size, d_model, self.device, self.dtype)
        self.rope = RotaryPositionalEmbedding(
            rope_theta, self.d_model // self.num_heads, max_seq_len=self.context_length, device=self.device
        )

        self.transformer_blocks = [
            TransformerBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                d_ff=self.d_ff,
                rope=self.rope,
                max_seq_len=self.context_length,
                device=self.device,
                dtype=self.dtype,
            )
            for _ in range(self.num_layers)
        ]

        self.norm = RMSNorm(self.d_model, device=self.device, dtype=self.dtype)
        self.output_layer = Linear(self.d_model, self.vocab_size, device=self.device, dtype=self.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.norm(x)
        x = self.output_layer(x)
        return x
