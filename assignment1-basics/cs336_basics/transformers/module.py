import jaxtyping
import numpy as np
import torch


class Linear(torch.nn.Module):
    def __init__(
        self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        # Initialize Weights
        self.weights = torch.nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        # Initialize Truncated Normal Distribution for Weights
        std = np.sqrt(2 / (in_features + out_features))
        torch.nn.init.trunc_normal_(self.weights, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Check if input tensor has the correct shape
        if x.shape[-1] != self.in_features:
            raise ValueError(f"Input tensor must have shape (*, {self.in_features}), but got {x.shape}")

        # Perform linear transformation
        return torch.matmul(x, self.weights.T)
