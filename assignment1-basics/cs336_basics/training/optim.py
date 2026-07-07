import math
from collections.abc import Iterable
from typing import Callable, Optional

import torch


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(  # type: ignore[override]
        self, closure: Optional[Callable[[], float]] = None
    ) -> float | None:
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 0)  # Get iteration number from the state, or 0.
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place.
                state["t"] = t + 1  # Increment iteration number.

        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.95,
        weight_decay: float = 0.01,
        eps: float = 1e-8,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")

        if eps < 0:
            raise ValueError(f"Invalid epsilon value: {eps}")

        if not 0 <= beta1 < 1:
            raise ValueError(f"Invalid beta1: {beta1}")

        if not 0 <= beta2 < 1:
            raise ValueError(f"Invalid beta2: {beta2}")

        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            weight_decay=weight_decay,
            eps=eps,
        )
        super().__init__(params, defaults)

    def step(  # type: ignore[override]
        self, closure: Optional[Callable[[], float]] = None
    ) -> float | None:
        loss = None if closure is None else closure()

        for group in self.param_groups:
            # Get hyperparameters for this parameter group.
            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 1)
                grad = p.grad.data

                cur_lr = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)  # Compute the learning rate for this step.
                p.data -= weight_decay * lr * p.data  # Apply weight decay.

                momentum = state.get("momentum", torch.zeros_like(p.data))
                momentum = beta1 * momentum + (1 - beta1) * grad  # Update momentum.

                var = state.get("var", torch.zeros_like(p.data))
                var = beta2 * var + (1 - beta2) * grad**2

                p.data -= cur_lr * momentum / (torch.sqrt(var) + eps)  # Update weight tensor in-place.

                state["t"] = t + 1  # Increment iteration number.

        return loss


# LLaMA Cosine Learning Rate Schedule
# First linearly increase the learning rate from 0 to a_max over T_w warmup steps,
# then decay it to a_min over T_c - T_w steps using a cosine schedule,
# and finally keep it at a_min for the remaining steps.
#
# t: current step
# a_max: maximum learning rate
# a_min: minimum learning rate
# T_w: the number of warm-up iterations
# T_c: the final iteration of cosine annealing
#
def CosineLearningRateSchedule(t: int, a_max: float, a_min: float, T_w: int, T_c: int) -> float:
    if t < T_w:
        return t / T_w * a_max
    elif T_w <= t and t <= T_c:
        return a_min + 0.5 * (a_max - a_min) * (1 + math.cos(math.pi * (t - T_w) / (T_c - T_w)))
    else:
        return a_min


# To avoid exploding gradients
# Enforce a limit on the norm of the gradient after each backward pass before taking an optimizer step
# NOTE: We take the gradient for all parameters, and computer its L2 norm.
def GradientClipping(params: Iterable[torch.nn.Parameter], max_norm: float):
    eps = 1e-6
    grad_sum = 0.0
    for p in params:
        if p.grad is None:
            continue
        grad_sum += (p.grad.data**2).sum().item()
    grad_norm = math.sqrt(grad_sum)
    clip_coef = max_norm / (grad_norm + eps)
    if grad_norm > max_norm:
        for p in params:
            if p.grad is None:
                continue
            p.grad.data.mul_(clip_coef)


if __name__ == "__main__":
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=0.01)
    for t in range(100):
        opt.zero_grad()  # Reset the gradients for all learnable parameters.
        loss = (weights**2).mean()  # Compute a scalar loss value.
        print(loss.cpu().item())
        loss.backward()  # Run backward pass, which computes gradients.
        opt.step()  # Run optimizer step.
