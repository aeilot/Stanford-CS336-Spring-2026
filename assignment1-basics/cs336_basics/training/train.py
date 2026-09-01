import typing
import torch
import numpy as np
from torch import nn
import os

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

    # Add the final dim, turn the offsets into a 2D tensor, and use broadcasting to create the batch
    x_batch = x_torch[starts[:, None] + offsets]
    y_batch = x_torch[starts[:, None] + offsets + 1]

    # Move only the current batch to the device to avoid storing the entire dataset in GPU memory.
    return x_batch.to(device), y_batch.to(device)

# Use mmap mode to load data. Put only partial data on the GPU.

# Checkpointing
def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]) -> None:
    """
    Save a checkpoint of the model and optimizer state.

    Args:
        model (nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        iteration (int): The current training iteration.
        out (str | os.PathLike | typing.BinaryIO | typing.IO[bytes]): The output file path or file-like object.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration,
    }
    torch.save(checkpoint, out)

def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], model: nn.Module, optimizer: torch.optim.Optimizer) -> int:
    """
    Load a checkpoint of the model and optimizer state.

    Args:
        src (str | os.PathLike | typing.BinaryIO | typing.IO[bytes]): The source file path or file-like object.
        model (nn.Module): The model to load the state into.
        optimizer (torch.optim.Optimizer): The optimizer to load the state into.

    Returns:
        int: The iteration number from the checkpoint.
    """
    checkpoint = torch.load(src)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint['iteration']
