import torch


# Defining CrossEntropy for Transformers LM
# logits: (batch_size, vocab_size)
# targets: (batch_size)
# For Transformers, we can flatten the seq_len dim and get the targeted shape for this loss function.
def CrossEntropyLoss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # CrossEntropy = lse - target_logits
    # We should reduce the use of exp and log to avoid numerical instability.
    # logits can be very large, which can lead to overflow when taking the exp.
    lse = torch.logsumexp(logits, dim=-1)
    target_logits = logits[torch.arange(logits.shape[0]), targets]
    return (lse - target_logits).mean()


def Perplexity(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    loss = CrossEntropyLoss(logits, targets)
    return torch.exp(loss)
