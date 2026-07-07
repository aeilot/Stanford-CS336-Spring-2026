import torch
from torch import nn

from cs336_basics.transformers.module import Softmax


# Defining CrossEntropy for Transformers LM
# logits: (batch_size, vocab_size)
# targets: (batch_size)
# For Transformers, we can flatten the seq_len dim and get the targeted shape for this loss function.
class CrossEntropyLoss(nn.Module):
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # CrossEntropy = lse - target_logits
        lse = torch.logsumexp(logits, dim=-1)
        target_logits = logits[torch.arange(logits.shape[0]), targets]
        return (lse - target_logits).mean()
