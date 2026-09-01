import typing
import torch
import numpy as np

def load_data(x: np.ndarray, batch_size: int, context_length: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load data into batches for training.

    Args:
        x (np.ndarray): Input data.
        batch_size (int): Size of each batch.
        context_length (int): Length of the context window.
        device (torch.device): Device to load the data onto.
    
    Returns:
        tuple of torch.Tensor: Batches of input data.
    """
    x_torch = torch.from_numpy(x)

    starts = torch.randint(
        0,
        len(x_torch) - context_length,
        (batch_size,),
    )
    
    offsets = torch.arange(context_length)

    x_batch = x_torch[starts[:, None] + offsets]
    y_batch = x_torch[starts[:, None] + offsets + 1]

    return x_batch.to(device), y_batch.to(device)